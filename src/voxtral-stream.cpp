// ============================================================================
// Internal streaming session skeleton (v1) — implementation.
//
// Compatibility path: full buffered execution at finish.
// Will be replaced incrementally by Mel/encoder/decoder stages.
//
// See src/voxtral-stream.h and docs/architecture/streaming-runtime.md.
// ============================================================================

#include "voxtral-stream.h"

#include <cmath>
#include <cstdint>
#include <deque>
#include <exception>
#include <new>
#include <string>
#include <utility>
#include <vector>

// ----------------------------------------------------------------------------
// Bounds
// ----------------------------------------------------------------------------
namespace {

// Never grow a single feed's copy past what a float vector can address; also a
// sanity ceiling that keeps overflow arithmetic well-defined.
constexpr uint64_t kMaxFeedSamples = 64ull * 1024 * 1024;   // 64M samples / feed

// Default hard bound for the event queue. v1 emits only a handful of lifecycle
// events per session, so this is never reached in practice; it is a backstop
// against a future decoder emitting unboundedly. It is a TRUE maximum: a push
// into a full queue is dropped and recorded, never grown past.
constexpr size_t kMaxEvents = 4096;

// ----------------------------------------------------------------------------
// Context factory / free — indirected through file-local pointers so the
// model-free unit tests can substitute them (see the test seam). Production
// always uses the real factory.
// ----------------------------------------------------------------------------
voxtral_stream_context_factory_fn g_context_factory = &voxtral_init_from_model;
voxtral_stream_context_free_fn    g_context_free    = &voxtral_free;

} // namespace

// ----------------------------------------------------------------------------
// The stream object.
// ----------------------------------------------------------------------------
struct voxtral_stream {
    // Execution engine. Owned by this stream iff `owns_context` (created from a
    // model). May be null for lifecycle-only streams (model == nullptr).
    voxtral_context     * ctx          = nullptr;
    bool                  owns_context = false;
    voxtral_stream_params params;

    voxtral_stream_state  state       = voxtral_stream_state::created;
    voxtral_status        last_status = voxtral_status::ok;
    std::string           last_error;

    // Non-blocking reentrancy/concurrency guard (see the threading contract).
    // Set while a mutating entry point runs on this stream.
    bool in_operation = false;

    bool cancel_requested = false;
    bool cancelled_emitted = false;

    // Canonical PCM: float32 mono in nominal [-1, 1]. Append-only in v1.
    std::vector<float> pcm;
    // Reserved for the incremental frontend: the first unconsumed sample index.
    // Stays 0 in v1 (the whole buffer is handed to the batch path at finish).
    size_t   pcm_read_offset      = 0;

    uint64_t total_samples_received = 0;
    uint64_t total_samples_consumed = 0;
    uint64_t feed_calls             = 0;
    uint64_t inference_runs         = 0;

    std::vector<int32_t> tokens;
    std::string          transcript;

    std::deque<voxtral_stream_event> events;
    size_t max_events        = kMaxEvents;   // hard bound; test-overridable
    bool   events_overflowed = false;

    // Test seam: invoked once inside finish() while state == finishing.
    voxtral_stream_finishing_hook_fn finishing_hook = nullptr;
    void *                           finishing_hook_user = nullptr;
};

// ----------------------------------------------------------------------------
// Small helpers
// ----------------------------------------------------------------------------
namespace {

void set_error(voxtral_stream * s, voxtral_status status, const std::string & msg) {
    s->last_status = status;
    s->last_error  = msg;
}

void clear_error(voxtral_stream * s) {
    s->last_status = voxtral_status::ok;
    s->last_error.clear();
}

double samples_to_ms(uint64_t samples, int32_t sample_rate) {
    if (sample_rate <= 0) return 0.0;
    // Derived on demand from the 64-bit count — no floating-point accumulation.
    return static_cast<double>(samples) * 1000.0 / static_cast<double>(sample_rate);
}

// Non-blocking reentrancy guard. On construction it tries to engage; if another
// operation is already in progress on the same stream, engage() fails and the
// caller must return `busy`. Never blocks, so cancel()/reset() cannot create a
// false impression of interrupting an in-flight finish().
struct op_guard {
    voxtral_stream * s;
    bool engaged = false;
    explicit op_guard(voxtral_stream * str) : s(str) {}
    bool engage() {
        if (s->in_operation) return false;
        s->in_operation = true;
        engaged = true;
        return true;
    }
    ~op_guard() { if (engaged) s->in_operation = false; }
};

// Strictly bounded push. Returns false without enqueuing (and records the
// overflow loudly: overflow flag + last_error) when the queue is at its bound.
// Never grows past max_events; never silently drops.
bool push_event(voxtral_stream * s, voxtral_stream_event ev) {
    if (s->events.size() >= s->max_events) {
        s->events_overflowed = true;
        set_error(s, voxtral_status::limit_exceeded,
                  std::string("event queue full (bound ") + std::to_string(s->max_events) +
                  "): dropped " + voxtral_stream_event_name(ev.type) + " event");
        return false;
    }
    s->events.push_back(std::move(ev));
    return true;
}

void emit_final_and_completed(voxtral_stream * s) {
    const double t_ms = samples_to_ms(s->total_samples_received, s->params.sample_rate);
    {
        voxtral_stream_event ev;
        ev.type       = voxtral_stream_event_type::final_text;
        ev.text       = s->transcript;   // owned copy
        ev.t_audio_ms = t_ms;
        push_event(s, std::move(ev));
    }
    {
        voxtral_stream_event ev;
        ev.type       = voxtral_stream_event_type::completed;
        ev.t_audio_ms = t_ms;
        push_event(s, std::move(ev));
    }
}

void emit_error(voxtral_stream * s, voxtral_status status, const std::string & msg) {
    voxtral_stream_event ev;
    ev.type       = voxtral_stream_event_type::error;
    ev.text       = msg;
    ev.error_code = static_cast<int32_t>(status);
    ev.t_audio_ms = samples_to_ms(s->total_samples_received, s->params.sample_rate);
    push_event(s, std::move(ev));
}

bool feed_allowed(voxtral_stream_state st) {
    return st == voxtral_stream_state::created || st == voxtral_stream_state::running;
}

// Common core for both feed variants: guard reentrancy, validate state/args,
// size guards, then append via the caller-provided converter. `convert(dst)`
// must append exactly `count` floats to `dst`.
template <typename Convert>
voxtral_status feed_common(voxtral_stream * s, const void * ptr, size_t count, Convert convert) {
    if (!s) return voxtral_status::invalid_argument;

    op_guard guard(s);
    if (!guard.engage()) {
        // Concurrent/reentrant misuse. Do not mutate shared state (the returned
        // status is enough); see the threading contract.
        return voxtral_status::busy;
    }

    if (!feed_allowed(s->state)) {
        set_error(s, voxtral_status::invalid_state,
                  std::string("feed not allowed in state ") + voxtral_stream_state_name(s->state));
        return voxtral_status::invalid_state;
    }

    if (ptr == nullptr && count != 0) {
        set_error(s, voxtral_status::invalid_argument, "null samples with non-zero count");
        return voxtral_status::invalid_argument;
    }

    // Zero-length feed: successful no-op. Does not change state or audio position.
    if (count == 0) {
        clear_error(s);
        s->feed_calls++;
        return voxtral_status::ok;
    }

    if (static_cast<uint64_t>(count) > kMaxFeedSamples) {
        set_error(s, voxtral_status::invalid_argument, "feed exceeds per-call sample limit");
        return voxtral_status::invalid_argument;
    }

    const uint64_t received = s->total_samples_received;
    // Overflow-safe cumulative bound check. This is the temporary full-buffer
    // compatibility limit, not an allocation failure -> limit_exceeded.
    if (count > s->params.max_total_samples ||
        received > s->params.max_total_samples - static_cast<uint64_t>(count)) {
        set_error(s, voxtral_status::limit_exceeded,
                  "accumulated PCM exceeds max_total_samples: temporary full-buffer "
                  "compatibility limit that keeps decode below the decoder KV window "
                  "(prevents reaching the unsafe kv_cache_shift_left rollover path)");
        return voxtral_status::limit_exceeded;
    }

    try {
        s->pcm.reserve(s->pcm.size() + count);
    } catch (const std::exception & e) {
        // A genuine allocation failure (distinct from the compatibility limit).
        set_error(s, voxtral_status::out_of_memory, std::string("pcm reserve failed: ") + e.what());
        return voxtral_status::out_of_memory;
    }

    if (!convert(s->pcm)) {
        // Converter rejected the payload (e.g. non-finite float). It must not
        // have mutated the buffer; state is unchanged.
        return s->last_status;
    }

    s->total_samples_received += static_cast<uint64_t>(count);
    s->feed_calls++;
    if (s->state == voxtral_stream_state::created) {
        s->state = voxtral_stream_state::running;
    }
    clear_error(s);
    return voxtral_status::ok;
}

} // namespace

// ============================================================================
// Params validation
// ============================================================================
voxtral_status voxtral_stream_params_check(const voxtral_stream_params & params) {
    if (params.sample_rate != VOXTRAL_SAMPLE_RATE || params.channels != 1) {
        return voxtral_status::unsupported_audio_format;
    }
    if (params.max_total_samples == 0) {
        return voxtral_status::invalid_argument;
    }
    return voxtral_status::ok;
}

// ============================================================================
// Lifecycle
// ============================================================================
voxtral_stream * voxtral_stream_create_internal(
    voxtral_model                * model,
    const voxtral_context_params & ctx_params,
    const voxtral_stream_params  & params)
{
    if (voxtral_stream_params_check(params) != voxtral_status::ok) {
        return nullptr;
    }
    auto * s = new (std::nothrow) voxtral_stream();
    if (!s) return nullptr;
    s->params = params;

    if (model != nullptr) {
        // Preferred path: the stream creates and owns its own mutable execution
        // context from the shared, immutable model. One context per stream.
        voxtral_context * ctx = g_context_factory(model, ctx_params);
        if (!ctx) {
            // Surface the failure as a queryable status/error rather than a bare
            // nullptr. The caller still destroys the stream; it owns no context.
            s->state = voxtral_stream_state::failed;
            set_error(s, voxtral_status::backend_error,
                      "failed to create per-stream execution context from model");
            return s;
        }
        s->ctx          = ctx;
        s->owns_context = true;
    }
    // model == nullptr: lifecycle-only stream, no owned context.
    return s;
}

void voxtral_stream_destroy_internal(voxtral_stream * stream) {
    if (!stream) return;   // destroy(nullptr) is safe.
    // Owns only its own context (and mutable state); never frees the model. The
    // caller guarantees no operation is in flight (threading contract).
    if (stream->owns_context && stream->ctx) {
        g_context_free(stream->ctx);
    }
    delete stream;
}

voxtral_status voxtral_stream_feed_pcm16_internal(
    voxtral_stream * stream, const int16_t * samples, size_t sample_count)
{
    return feed_common(stream, samples, sample_count, [&](std::vector<float> & dst) {
        for (size_t i = 0; i < sample_count; ++i) {
            // Deterministic S16 -> float. -32768 maps to exactly -1.0f.
            dst.push_back(static_cast<float>(samples[i]) / 32768.0f);
        }
        return true;
    });
}

voxtral_status voxtral_stream_feed_f32_internal(
    voxtral_stream * stream, const float * samples, size_t sample_count)
{
    return feed_common(stream, samples, sample_count, [&](std::vector<float> & dst) {
        // Validate finiteness up front so a rejected payload never mutates the
        // buffer. The canonical range is [-1, 1]; values outside are passed
        // through unchanged (no silent clamp) to preserve batch-path parity.
        for (size_t i = 0; i < sample_count; ++i) {
            if (!std::isfinite(samples[i])) {
                set_error(stream, voxtral_status::invalid_argument,
                          "float32 feed contains a non-finite sample");
                return false;
            }
        }
        dst.insert(dst.end(), samples, samples + sample_count);
        return true;
    });
}

voxtral_status voxtral_stream_finish_internal(voxtral_stream * stream) {
    if (!stream) return voxtral_status::invalid_argument;

    op_guard guard(stream);
    if (!guard.engage()) {
        // Reentrant finish() (only reachable from within a running finish()).
        return voxtral_status::busy;
    }

    switch (stream->state) {
        case voxtral_stream_state::completed:
            // Idempotent: never run inference a second time.
            return voxtral_status::ok;
        case voxtral_stream_state::cancelled:
            // Cancelled before finish: no inference, no additional events.
            return voxtral_status::ok;
        case voxtral_stream_state::failed:
            set_error(stream, voxtral_status::invalid_state, "finish not allowed after failure");
            return voxtral_status::invalid_state;
        case voxtral_stream_state::finishing:
            // Unreachable while externally serialized (the guard would have
            // rejected a reentrant call with `busy` already); defensive.
            return voxtral_status::busy;
        case voxtral_stream_state::created:
        case voxtral_stream_state::running:
            break;
    }

    stream->state = voxtral_stream_state::finishing;

    // Test seam: observe the transient `finishing` state / probe reentrancy.
    // Any reentrant stream call from here returns `busy` (the guard is engaged).
    if (stream->finishing_hook) {
        stream->finishing_hook(stream, stream->finishing_hook_user);
    }

    // Empty stream: documented as COMPLETED with an empty final transcript and
    // no inference.
    if (stream->total_samples_received == 0) {
        stream->transcript.clear();
        stream->tokens.clear();
        stream->total_samples_consumed = 0;
        emit_final_and_completed(stream);
        stream->state = voxtral_stream_state::completed;
        clear_error(stream);
        return voxtral_status::ok;
    }

    if (stream->ctx == nullptr) {
        set_error(stream, voxtral_status::backend_error,
                  "no execution context: cannot transcribe buffered audio");
        emit_error(stream, voxtral_status::backend_error, stream->last_error);
        stream->state = voxtral_stream_state::failed;
        return voxtral_status::backend_error;
    }

    // Compatibility path: hand the fully buffered canonical PCM to the existing
    // batch inference exactly once. Chunk boundaries do not affect the result
    // because the accumulated buffer is byte-identical regardless of feeding.
    voxtral_result result;
    bool ok = false;
    try {
        ok = voxtral_transcribe_audio(*stream->ctx, stream->pcm, stream->params.max_tokens, result);
        stream->inference_runs++;   // one execution per finish, success or failure
    } catch (const std::exception & e) {
        set_error(stream, voxtral_status::backend_error,
                  std::string("inference threw: ") + e.what());
        emit_error(stream, voxtral_status::backend_error, stream->last_error);
        stream->state = voxtral_stream_state::failed;
        return voxtral_status::backend_error;
    }

    if (!ok) {
        set_error(stream, voxtral_status::backend_error, "batch inference reported failure");
        emit_error(stream, voxtral_status::backend_error, stream->last_error);
        stream->state = voxtral_stream_state::failed;
        return voxtral_status::backend_error;
    }

    stream->transcript             = std::move(result.text);
    stream->tokens                 = std::move(result.tokens);
    stream->total_samples_consumed = stream->total_samples_received;
    emit_final_and_completed(stream);
    stream->state = voxtral_stream_state::completed;
    clear_error(stream);
    return voxtral_status::ok;
}

voxtral_status voxtral_stream_reset_internal(voxtral_stream * stream) {
    if (!stream) return voxtral_status::invalid_argument;

    op_guard guard(stream);
    if (!guard.engage()) {
        // reset() during an in-flight finish() is rejected without mutating state.
        return voxtral_status::busy;
    }

    // Clears all per-stream mutable runtime state; the owned context, params,
    // and test knobs (max_events / finishing hook) are kept for cheap reuse.
    // Deliberately no shrink_to_fit(): keep the PCM capacity so reuse does not
    // re-allocate.
    stream->pcm.clear();
    stream->pcm_read_offset        = 0;
    stream->total_samples_received = 0;
    stream->total_samples_consumed = 0;
    stream->feed_calls             = 0;
    stream->inference_runs         = 0;
    stream->tokens.clear();
    stream->transcript.clear();
    stream->events.clear();
    stream->events_overflowed = false;
    stream->cancel_requested  = false;
    stream->cancelled_emitted = false;
    stream->state = voxtral_stream_state::created;
    clear_error(stream);
    return voxtral_status::ok;
}

voxtral_status voxtral_stream_cancel_internal(voxtral_stream * stream) {
    if (!stream) return voxtral_status::invalid_argument;

    op_guard guard(stream);
    if (!guard.engage()) {
        // cancel() cannot interrupt an in-flight finish(): it returns `busy` and
        // does NOT set `cancelled`. This is what prevents a cancelled -> completed
        // transition (the finish proceeds and completes normally).
        return voxtral_status::busy;
    }

    switch (stream->state) {
        case voxtral_stream_state::completed:
            set_error(stream, voxtral_status::invalid_state, "cannot cancel a completed stream");
            return voxtral_status::invalid_state;
        case voxtral_stream_state::cancelled:
        case voxtral_stream_state::failed:
            // Idempotent no-op; do not emit a second CANCELLED event.
            return voxtral_status::ok;
        case voxtral_stream_state::finishing:
            // Unreachable while externally serialized (guard rejects reentrancy
            // with `busy`); defensive — never convert an in-flight finish.
            set_error(stream, voxtral_status::invalid_state,
                      "cannot cancel during finish");
            return voxtral_status::invalid_state;
        case voxtral_stream_state::created:
        case voxtral_stream_state::running:
            break;
    }

    stream->cancel_requested = true;
    stream->state = voxtral_stream_state::cancelled;
    if (!stream->cancelled_emitted) {
        voxtral_stream_event ev;
        ev.type       = voxtral_stream_event_type::cancelled;
        ev.t_audio_ms = samples_to_ms(stream->total_samples_received, stream->params.sample_rate);
        push_event(stream, std::move(ev));
        stream->cancelled_emitted = true;
    }
    clear_error(stream);
    return voxtral_status::ok;
}

// ============================================================================
// Introspection
// ============================================================================
voxtral_stream_state voxtral_stream_get_state(const voxtral_stream * s) {
    return s ? s->state : voxtral_stream_state::failed;
}
voxtral_status voxtral_stream_last_status(const voxtral_stream * s) {
    return s ? s->last_status : voxtral_status::invalid_argument;
}
const std::string & voxtral_stream_last_error(const voxtral_stream * s) {
    static const std::string empty;
    return s ? s->last_error : empty;
}
uint64_t voxtral_stream_samples_received(const voxtral_stream * s) {
    return s ? s->total_samples_received : 0;
}
uint64_t voxtral_stream_samples_consumed(const voxtral_stream * s) {
    return s ? s->total_samples_consumed : 0;
}
uint64_t voxtral_stream_feed_calls(const voxtral_stream * s) {
    return s ? s->feed_calls : 0;
}
uint64_t voxtral_stream_inference_runs(const voxtral_stream * s) {
    return s ? s->inference_runs : 0;
}
double voxtral_stream_audio_ms(const voxtral_stream * s) {
    return s ? samples_to_ms(s->total_samples_received, s->params.sample_rate) : 0.0;
}
const std::vector<int32_t> & voxtral_stream_tokens(const voxtral_stream * s) {
    static const std::vector<int32_t> empty;
    return s ? s->tokens : empty;
}
const std::string & voxtral_stream_transcript(const voxtral_stream * s) {
    static const std::string empty;
    return s ? s->transcript : empty;
}
bool voxtral_stream_owns_context(const voxtral_stream * s) {
    return s ? s->owns_context : false;
}
const void * voxtral_stream_context_ptr(const voxtral_stream * s) {
    return s ? static_cast<const void *>(s->ctx) : nullptr;
}
const float * voxtral_stream_pcm_data(const voxtral_stream * s) {
    return (s && !s->pcm.empty()) ? s->pcm.data() : nullptr;
}
size_t voxtral_stream_pcm_size(const voxtral_stream * s) {
    return s ? s->pcm.size() : 0;
}

// ============================================================================
// Events
// ============================================================================
bool voxtral_stream_poll_event(voxtral_stream * s, voxtral_stream_event & out) {
    if (!s || s->events.empty()) return false;
    out = std::move(s->events.front());
    s->events.pop_front();
    return true;
}
size_t voxtral_stream_pending_events(const voxtral_stream * s) {
    return s ? s->events.size() : 0;
}

// ============================================================================
// Name helpers (diagnostics / machine-readable output)
// ============================================================================
const char * voxtral_stream_state_name(voxtral_stream_state state) {
    switch (state) {
        case voxtral_stream_state::created:   return "created";
        case voxtral_stream_state::running:   return "running";
        case voxtral_stream_state::finishing: return "finishing";
        case voxtral_stream_state::completed: return "completed";
        case voxtral_stream_state::cancelled: return "cancelled";
        case voxtral_stream_state::failed:    return "failed";
    }
    return "unknown";
}

const char * voxtral_stream_status_name(voxtral_status status) {
    switch (status) {
        case voxtral_status::ok:                       return "ok";
        case voxtral_status::invalid_argument:         return "invalid_argument";
        case voxtral_status::invalid_state:            return "invalid_state";
        case voxtral_status::unsupported_audio_format: return "unsupported_audio_format";
        case voxtral_status::cancelled:                return "cancelled";
        case voxtral_status::backend_error:            return "backend_error";
        case voxtral_status::out_of_memory:            return "out_of_memory";
        case voxtral_status::limit_exceeded:           return "limit_exceeded";
        case voxtral_status::busy:                     return "busy";
        case voxtral_status::internal_error:           return "internal_error";
    }
    return "unknown";
}

const char * voxtral_stream_event_name(voxtral_stream_event_type type) {
    switch (type) {
        case voxtral_stream_event_type::token:        return "token";
        case voxtral_stream_event_type::partial_text: return "partial_text";
        case voxtral_stream_event_type::final_text:   return "final_text";
        case voxtral_stream_event_type::error:        return "error";
        case voxtral_stream_event_type::completed:    return "completed";
        case voxtral_stream_event_type::cancelled:    return "cancelled";
    }
    return "unknown";
}

// ============================================================================
// Test seam (internal, test-only — see header). No effect unless a test calls
// these; production code always uses the defaults.
// ============================================================================
void voxtral_stream_test_set_context_factory(voxtral_stream_context_factory_fn factory) {
    g_context_factory = factory ? factory : &voxtral_init_from_model;
}
void voxtral_stream_test_set_context_free(voxtral_stream_context_free_fn free_fn) {
    g_context_free = free_fn ? free_fn : &voxtral_free;
}
void voxtral_stream_test_set_finishing_hook(
    voxtral_stream * stream, voxtral_stream_finishing_hook_fn hook, void * user)
{
    if (!stream) return;
    stream->finishing_hook      = hook;
    stream->finishing_hook_user = user;
}
void voxtral_stream_test_set_max_events(voxtral_stream * stream, size_t max_events) {
    if (!stream || max_events == 0) return;
    stream->max_events = max_events;
}
bool voxtral_stream_test_events_overflowed(const voxtral_stream * stream) {
    return stream ? stream->events_overflowed : false;
}

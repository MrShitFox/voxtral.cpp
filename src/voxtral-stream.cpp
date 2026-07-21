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

// Safety valve for the event queue. v1 emits only a handful of lifecycle events
// per session, so this is never reached in practice; lifecycle/error events are
// never dropped (see push_event).
constexpr size_t kMaxEvents = 4096;

} // namespace

// ----------------------------------------------------------------------------
// The stream object.
// ----------------------------------------------------------------------------
struct voxtral_stream {
    // Borrowed execution engine (may be null for lifecycle-only streams).
    voxtral_context     * ctx = nullptr;
    voxtral_stream_params params;

    voxtral_stream_state  state       = voxtral_stream_state::created;
    voxtral_status        last_status = voxtral_status::ok;
    std::string           last_error;

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

void push_event(voxtral_stream * s, voxtral_stream_event ev) {
    const bool is_lifecycle =
        ev.type == voxtral_stream_event_type::final_text ||
        ev.type == voxtral_stream_event_type::completed  ||
        ev.type == voxtral_stream_event_type::cancelled  ||
        ev.type == voxtral_stream_event_type::error;
    if (s->events.size() >= kMaxEvents && !is_lifecycle) {
        // Drop a stale, droppable (token/partial) event to make room. Lifecycle
        // events are always retained.
        s->events.pop_front();
    }
    s->events.push_back(std::move(ev));
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

// Common core for both feed variants: validate state/args, size guards, then
// append via the caller-provided converter. `convert(dst)` must append exactly
// `count` floats to `dst`.
template <typename Convert>
voxtral_status feed_common(voxtral_stream * s, const void * ptr, size_t count, Convert convert) {
    if (!s) return voxtral_status::invalid_argument;

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
    // Overflow-safe cumulative bound check.
    if (count > s->params.max_total_samples ||
        received > s->params.max_total_samples - static_cast<uint64_t>(count)) {
        set_error(s, voxtral_status::out_of_memory,
                  "accumulated PCM exceeds max_total_samples bound");
        return voxtral_status::out_of_memory;
    }

    try {
        s->pcm.reserve(s->pcm.size() + count);
    } catch (const std::exception & e) {
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
    voxtral_context             * ctx,
    const voxtral_stream_params & params)
{
    if (voxtral_stream_params_check(params) != voxtral_status::ok) {
        return nullptr;
    }
    auto * s = new (std::nothrow) voxtral_stream();
    if (!s) return nullptr;
    s->ctx    = ctx;
    s->params = params;
    return s;
}

void voxtral_stream_destroy_internal(voxtral_stream * stream) {
    // Owns only its own mutable state; never frees ctx or model. Safe in any state.
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

    switch (stream->state) {
        case voxtral_stream_state::completed:
        case voxtral_stream_state::finishing:
            // Idempotent: never run inference a second time.
            return voxtral_status::ok;
        case voxtral_stream_state::cancelled:
            // Cancelled before finish: no inference, no additional events.
            return voxtral_status::ok;
        case voxtral_stream_state::failed:
            set_error(stream, voxtral_status::invalid_state, "finish not allowed after failure");
            return voxtral_status::invalid_state;
        case voxtral_stream_state::created:
        case voxtral_stream_state::running:
            break;
    }

    stream->state = voxtral_stream_state::finishing;

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

    // Clears all per-stream mutable state; ctx/params (model, device) are kept.
    stream->pcm.clear();
    stream->pcm.shrink_to_fit();
    stream->pcm_read_offset       = 0;
    stream->total_samples_received = 0;
    stream->total_samples_consumed = 0;
    stream->feed_calls             = 0;
    stream->inference_runs         = 0;
    stream->tokens.clear();
    stream->transcript.clear();
    stream->events.clear();
    stream->cancel_requested  = false;
    stream->cancelled_emitted = false;
    stream->state = voxtral_stream_state::created;
    clear_error(stream);
    return voxtral_status::ok;
}

voxtral_status voxtral_stream_cancel_internal(voxtral_stream * stream) {
    if (!stream) return voxtral_status::invalid_argument;

    switch (stream->state) {
        case voxtral_stream_state::completed:
            set_error(stream, voxtral_status::invalid_state, "cannot cancel a completed stream");
            return voxtral_status::invalid_state;
        case voxtral_stream_state::cancelled:
        case voxtral_stream_state::failed:
            // Idempotent no-op; do not emit a second CANCELLED event.
            return voxtral_status::ok;
        case voxtral_stream_state::created:
        case voxtral_stream_state::running:
        case voxtral_stream_state::finishing:
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

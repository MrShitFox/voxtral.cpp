// ============================================================================
// Internal streaming session skeleton (v1) — implementation.
//
// Compatibility path: full buffered execution at finish.
// Will be replaced incrementally by Mel/encoder/decoder stages.
//
// See src/voxtral-stream.h and docs/architecture/streaming-runtime.md.
// ============================================================================

#include "voxtral-stream.h"
#include "voxtral-internal.h"   // voxtral_transcribe_mel_internal (Mel -> text path)

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <exception>
#include <new>
#include <stdexcept>
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

// ----------------------------------------------------------------------------
// Minimal incremental SHA-256 over the canonical PCM. Lets the stream report a
// stable, chunk-invariant digest of the whole audio without retaining it. The
// running state is copied before finalization so it can be queried repeatedly.
// ----------------------------------------------------------------------------
struct Sha256 {
    uint32_t h[8];
    uint64_t len = 0;
    uint8_t  buf[64];
    size_t   buf_len = 0;

    Sha256() { reset(); }

    void reset() {
        static const uint32_t init[8] = {
            0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
            0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
        std::memcpy(h, init, sizeof(init));
        len = 0;
        buf_len = 0;
    }

    static uint32_t rotr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }

    void block(const uint8_t * p) {
        static const uint32_t k[64] = {
            0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
            0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
            0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
            0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
            0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
            0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
            0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
            0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u};
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = (uint32_t(p[i * 4]) << 24) | (uint32_t(p[i * 4 + 1]) << 16) |
                   (uint32_t(p[i * 4 + 2]) << 8) | uint32_t(p[i * 4 + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            uint32_t ch = (e & f) ^ (~e & g);
            uint32_t t1 = hh + S1 + ch + k[i] + w[i];
            uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t t2 = S0 + maj;
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }

    void update(const void * data, size_t n) {
        const uint8_t * p = static_cast<const uint8_t *>(data);
        len += n;
        while (n > 0) {
            size_t take = 64 - buf_len;
            if (take > n) take = n;
            std::memcpy(buf + buf_len, p, take);
            buf_len += take; p += take; n -= take;
            if (buf_len == 64) { block(buf); buf_len = 0; }
        }
    }

    // Finalize a COPY, leaving `this` usable for further updates.
    std::string hex() const {
        Sha256 s = *this;
        uint64_t bits = s.len * 8;
        uint8_t pad = 0x80;
        s.update(&pad, 1);
        uint8_t zero = 0;
        while (s.buf_len != 56) s.update(&zero, 1);
        uint8_t lenbe[8];
        for (int i = 0; i < 8; ++i) lenbe[i] = uint8_t(bits >> (56 - i * 8));
        s.update(lenbe, 8);
        static const char * hexd = "0123456789abcdef";
        std::string out;
        out.reserve(64);
        for (int i = 0; i < 8; ++i) {
            for (int j = 3; j >= 0; --j) {
                uint8_t byte = uint8_t(s.h[i] >> (j * 8));
                out.push_back(hexd[byte >> 4]);
                out.push_back(hexd[byte & 0xf]);
            }
        }
        return out;
    }
};

} // namespace

// Streaming left/right zero padding (mirrors the batch pad_audio_streaming in
// voxtral_transcribe_from_audio): left = 32 tokens, right = align-to-1280 + 17
// tokens, one token = VOXTRAL_RAW_AUDIO_LENGTH_PER_TOK raw samples.
static constexpr int64_t kStreamLeftPad =
    (int64_t) VOXTRAL_N_LEFT_PAD_TOKENS * VOXTRAL_RAW_AUDIO_LENGTH_PER_TOK;

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

    // Canonical PCM buffer. Used ONLY by lifecycle-only streams (no context):
    // inference streams route samples straight into the incremental Mel frontend
    // and never retain the full PCM here.
    std::vector<float> pcm;
    // Scratch buffer reused each feed to convert PCM16/f32 to canonical float32.
    std::vector<float> feed_scratch;

    // Incremental Mel frontend (inference streams only; null for lifecycle-only).
    // Owns the bounded rolling PCM tail and the accumulated Mel frames.
    voxtral_mel_frontend * mel_fe = nullptr;
    bool     left_pad_injected    = false;  // streaming left zero-pad injected once
    bool     full_pcm_buffered_at_finish = false;

    // Incremental causal encoder (inference streams only; null otherwise). Consumes
    // stable Mel frames during feed and accumulates the encoder output, so finish()
    // never re-runs the encoder over the whole Mel.
    voxtral_encoder_stream * enc = nullptr;
    int64_t  enc_pushed_frames   = 0;  // stable Mel frames already handed to the encoder

    // Rolling SHA-256 of the canonical PCM (chunk-invariant; no retention needed).
    Sha256   pcm_sha;

    // Even-trimmed Mel matrix [n_mel, n_frames] assembled at finish() (batch path).
    std::vector<float> mel_even;
    int32_t            mel_even_frames = 0;

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

// Lazily create the incremental Mel frontend for an inference stream on first
// feed. Deferred (not done in create) so the model-free ownership tests, which
// substitute an opaque sentinel context and never feed, never dereference it.
// Returns false only on a genuine allocation failure. A lifecycle-only stream
// (no owned context) or a context without frontend tables leaves mel_fe null and
// keeps the full-PCM path.
bool ensure_frontend(voxtral_stream * s) {
    if (s->mel_fe) return true;
    if (!s->owns_context || !s->ctx) return true;
    const float * hann    = voxtral_ctx_hann_window(s->ctx);
    const float * filters = voxtral_ctx_mel_filters(s->ctx);
    if (!hann || !filters) return true;
    s->mel_fe = voxtral_mel_frontend_create(hann, filters);
    if (!s->mel_fe) {
        set_error(s, voxtral_status::out_of_memory, "failed to create incremental Mel frontend");
        return false;
    }
    return true;
}

// Inject the streaming left zero-padding into the Mel frontend exactly once,
// before the first real sample. Mirrors the batch left_pad prefix. Bounded: the
// injected silence is emitted to stable Mel frames and the PCM tail compacted
// before this returns, so it does not create a lasting 40960-sample buffer.
void ensure_left_pad(voxtral_stream * s) {
    if (s->mel_fe && !s->left_pad_injected) {
        s->left_pad_injected = true;
        voxtral_mel_frontend_feed_silence(s->mel_fe, (size_t) kStreamLeftPad);
    }
}

voxtral_mel_metrics stream_mel_metrics(const voxtral_stream * s) {
    return (s && s->mel_fe) ? voxtral_mel_frontend_metrics(s->mel_fe) : voxtral_mel_metrics{};
}

voxtral_encoder_metrics stream_encoder_metrics(const voxtral_stream * s) {
    return (s && s->enc) ? voxtral_encoder_stream_metrics(s->enc) : voxtral_encoder_metrics{};
}

// Lazily create the incremental causal encoder once the Mel frontend exists (which
// implies a real execution context with frontend tables). Model-free ownership
// tests use a sentinel context with no frontend, so mel_fe stays null and no
// encoder is created. Returns false only on a genuine allocation failure.
bool ensure_encoder(voxtral_stream * s) {
    if (s->enc) return true;
    if (!s->mel_fe || !s->owns_context || !s->ctx) return true;
    s->enc = voxtral_encoder_stream_create(s->ctx);
    if (!s->enc) {
        set_error(s, voxtral_status::out_of_memory, "failed to create incremental encoder");
        return false;
    }
    return true;
}

// Hand every newly-stable Mel frame from the frontend to the incremental encoder,
// which runs any encoder chunks that became available. Called after each feed and
// once more (with the finish-flushed frames) at finish().
bool drain_mel_to_encoder(voxtral_stream * s) {
    if (!s->enc || !s->mel_fe) return true;
    const int64_t total = voxtral_mel_frontend_frame_count(s->mel_fe);
    if (total <= s->enc_pushed_frames) return true;
    const float * data = voxtral_mel_frontend_frames_data(s->mel_fe);
    if (!data) return true;
    const int64_t base = s->enc_pushed_frames;
    const int32_t n    = (int32_t) (total - base);
    const float * frames = data + (size_t) base * VOXTRAL_MEL_N_MEL;
    if (!voxtral_encoder_stream_push_mel(s->enc, frames, base, n)) {
        set_error(s, voxtral_status::backend_error, "incremental encoder rejected Mel frames");
        return false;
    }
    s->enc_pushed_frames = total;
    return true;
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

    // Lazily bring up the incremental Mel frontend (inference streams). Done
    // before any state mutation so a failure leaves the stream untouched.
    if (!ensure_frontend(s)) {
        return voxtral_status::out_of_memory;
    }

    // Convert into a reusable scratch buffer first, so a rejected payload never
    // mutates any persistent state.
    s->feed_scratch.clear();
    try {
        s->feed_scratch.reserve(count);
    } catch (const std::exception & e) {
        set_error(s, voxtral_status::out_of_memory, std::string("pcm reserve failed: ") + e.what());
        return voxtral_status::out_of_memory;
    }
    if (!convert(s->feed_scratch)) {
        // Converter rejected the payload (e.g. non-finite float); nothing changed.
        s->feed_scratch.clear();
        return s->last_status;
    }

    // Rolling SHA-256 of the canonical PCM (chunk-invariant; no retention needed).
    s->pcm_sha.update(s->feed_scratch.data(), s->feed_scratch.size() * sizeof(float));

    if (s->mel_fe) {
        // Inference stream: stream samples through the incremental Mel frontend.
        // No full PCM is retained — only the frontend's bounded rolling tail.
        ensure_left_pad(s);
        if (!voxtral_mel_frontend_feed(s->mel_fe, s->feed_scratch.data(), s->feed_scratch.size())) {
            set_error(s, voxtral_status::backend_error, "incremental Mel frontend rejected feed");
            s->feed_scratch.clear();
            return voxtral_status::backend_error;
        }
        // Drive the incremental causal encoder over the new stable Mel frames, so
        // encoder output is produced during feed (not deferred to finish).
        if (!ensure_encoder(s)) {
            s->feed_scratch.clear();
            return s->last_status;
        }
        if (!drain_mel_to_encoder(s)) {
            s->feed_scratch.clear();
            return voxtral_status::backend_error;
        }
    } else {
        // Lifecycle-only stream (no context): retain the full canonical PCM as
        // before (used by model-free tests; never runs inference).
        try {
            s->pcm.insert(s->pcm.end(), s->feed_scratch.begin(), s->feed_scratch.end());
        } catch (const std::exception & e) {
            set_error(s, voxtral_status::out_of_memory, std::string("pcm append failed: ") + e.what());
            s->feed_scratch.clear();
            return voxtral_status::out_of_memory;
        }
    }
    s->feed_scratch.clear();

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
        // The incremental Mel frontend is created lazily on the first feed (see
        // ensure_frontend): the model-free ownership tests substitute an opaque
        // sentinel context via the test seam and never feed, so create() must not
        // dereference the returned context here.
    }
    // model == nullptr: lifecycle-only stream, no owned context / frontend.
    return s;
}

void voxtral_stream_destroy_internal(voxtral_stream * stream) {
    if (!stream) return;   // destroy(nullptr) is safe.
    // Owns only its own context (and mutable state); never frees the model. The
    // caller guarantees no operation is in flight (threading contract).
    if (stream->enc) {
        voxtral_encoder_stream_destroy(stream->enc);
    }
    if (stream->mel_fe) {
        voxtral_mel_frontend_destroy(stream->mel_fe);
    }
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

    if (stream->ctx == nullptr || stream->mel_fe == nullptr) {
        set_error(stream, voxtral_status::backend_error,
                  "no execution context / Mel frontend: cannot transcribe audio");
        emit_error(stream, voxtral_status::backend_error, stream->last_error);
        stream->state = voxtral_stream_state::failed;
        return voxtral_status::backend_error;
    }

    // Incremental frontend + encoder path: no full PCM was buffered, and the
    // causal encoder already produced most of its output DURING feed. Push the
    // streaming right zero-padding (equivalent to the batch pad_audio_streaming
    // tail), flush the final Mel frames, drain them into the encoder, finalize it
    // (which runs at most the last one/two encoder chunks — never the whole Mel),
    // then run the shared adapter/decoder over the accumulated encoder output.
    // Chunk boundaries do not affect the Mel or the encoder output (both are
    // bit-for-bit the batch result of the same audio), so tokens are invariant.
    // The even-trimmed Mel is still assembled for the session-5 introspection.
    voxtral_result result;
    bool ok = false;
    try {
        ensure_left_pad(stream);   // no-op if already injected during feed
        const int64_t n_raw     = (int64_t) stream->total_samples_received;
        const int64_t mult      = VOXTRAL_RAW_AUDIO_LENGTH_PER_TOK;
        const int64_t align_pad = (mult - (n_raw % mult)) % mult;
        const int64_t right_pad = align_pad + (int64_t) VOXTRAL_N_RIGHT_PAD_TOKENS * mult;
        voxtral_mel_frontend_feed_silence(stream->mel_fe, (size_t) right_pad);
        voxtral_mel_frontend_finish(stream->mel_fe);
        voxtral_mel_frontend_assemble_even(stream->mel_fe, stream->mel_even, stream->mel_even_frames);
        stream->full_pcm_buffered_at_finish = false;

        if (!ensure_encoder(stream)) {
            throw std::runtime_error(stream->last_error);
        }
        if (stream->enc) {
            // Feed the finish-flushed Mel frames, then finalize the encoder.
            if (!drain_mel_to_encoder(stream)) {
                throw std::runtime_error(stream->last_error);
            }
            if (!voxtral_encoder_stream_finish(stream->enc)) {
                throw std::runtime_error("incremental encoder finish failed");
            }
            const float * enc_out    = voxtral_encoder_stream_output(stream->enc);
            const int32_t enc_frames = voxtral_encoder_stream_output_frames(stream->enc);
            ok = voxtral_transcribe_encoder_output_internal(
                *stream->ctx, enc_out, enc_frames, stream->params.max_tokens, result);
        } else {
            // Defensive fallback (should not happen for inference streams): run the
            // shared Mel -> text path over the accumulated incremental Mel.
            ok = voxtral_transcribe_mel_internal(*stream->ctx, stream->mel_even.data(),
                                                 stream->mel_even_frames, stream->params.max_tokens, result);
        }
        stream->inference_runs++;   // one execution per finish, success or failure
    } catch (const std::exception & e) {
        set_error(stream, voxtral_status::backend_error,
                  std::string("inference threw: ") + e.what());
        emit_error(stream, voxtral_status::backend_error, stream->last_error);
        stream->state = voxtral_stream_state::failed;
        return voxtral_status::backend_error;
    }

    if (!ok) {
        set_error(stream, voxtral_status::backend_error, "incremental Mel inference reported failure");
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
    // Deliberately no shrink_to_fit(): keep buffer capacities so reuse does not
    // re-allocate.
    stream->pcm.clear();
    stream->feed_scratch.clear();
    if (stream->mel_fe) {
        voxtral_mel_frontend_reset(stream->mel_fe);   // clears rolling PCM, frames, counters
    }
    if (stream->enc) {
        voxtral_encoder_stream_reset(stream->enc);    // clears Mel window, accumulated output, counters
    }
    stream->enc_pushed_frames            = 0;
    stream->left_pad_injected            = false;
    stream->full_pcm_buffered_at_finish  = false;
    stream->pcm_sha.reset();
    stream->mel_even.clear();
    stream->mel_even_frames        = 0;
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
std::string voxtral_stream_pcm_sha256(const voxtral_stream * s) {
    return s ? s->pcm_sha.hex() : Sha256{}.hex();
}

// --- Incremental Mel frontend introspection --------------------------------
bool voxtral_stream_uses_incremental_mel(const voxtral_stream * s) {
    return s ? (s->mel_fe != nullptr) : false;
}
int64_t voxtral_stream_mel_frames(const voxtral_stream * s) {
    return stream_mel_metrics(s).frames_total;
}
int64_t voxtral_stream_mel_frames_before_finish(const voxtral_stream * s) {
    return stream_mel_metrics(s).frames_during_feed;
}
int64_t voxtral_stream_mel_frames_flushed_at_finish(const voxtral_stream * s) {
    return stream_mel_metrics(s).frames_at_finish;
}
int64_t voxtral_stream_dft_frames_computed(const voxtral_stream * s) {
    return stream_mel_metrics(s).dft_frames_computed;
}
int64_t voxtral_stream_pcm_retained_samples(const voxtral_stream * s) {
    return stream_mel_metrics(s).pcm_retained;
}
int64_t voxtral_stream_pcm_peak_retained_samples(const voxtral_stream * s) {
    return stream_mel_metrics(s).pcm_peak_retained;
}
int64_t voxtral_stream_pcm_base_sample(const voxtral_stream * s) {
    return stream_mel_metrics(s).pcm_base;
}
bool voxtral_stream_full_pcm_buffered_at_finish(const voxtral_stream * s) {
    return s ? s->full_pcm_buffered_at_finish : false;
}
const float * voxtral_stream_mel_data(const voxtral_stream * s) {
    return (s && !s->mel_even.empty()) ? s->mel_even.data() : nullptr;
}
int32_t voxtral_stream_mel_data_frames(const voxtral_stream * s) {
    return s ? s->mel_even_frames : 0;
}

// --- Incremental causal encoder introspection ------------------------------
bool voxtral_stream_uses_incremental_encoder(const voxtral_stream * s) {
    return s ? (s->enc != nullptr) : false;
}
int64_t voxtral_stream_encoder_frames(const voxtral_stream * s) {
    return stream_encoder_metrics(s).encoderOutputFrames;
}
int64_t voxtral_stream_encoder_frames_before_finish(const voxtral_stream * s) {
    return stream_encoder_metrics(s).encoderFramesBeforeFinish;
}
int64_t voxtral_stream_encoder_frames_flushed_at_finish(const voxtral_stream * s) {
    return stream_encoder_metrics(s).encoderFramesFlushedAtFinish;
}
int64_t voxtral_stream_encoder_executions(const voxtral_stream * s) {
    return stream_encoder_metrics(s).encoderExecutions;
}
int64_t voxtral_stream_encoder_input_frames_processed(const voxtral_stream * s) {
    return stream_encoder_metrics(s).encoderInputFramesProcessed;
}
int64_t voxtral_stream_encoder_frames_recomputed(const voxtral_stream * s) {
    return stream_encoder_metrics(s).encoderFramesRecomputed;
}
int64_t voxtral_stream_encoder_max_window_frames(const voxtral_stream * s) {
    return stream_encoder_metrics(s).encoderMaxWindowFrames;
}
int64_t voxtral_stream_encoder_peak_context_frames(const voxtral_stream * s) {
    return stream_encoder_metrics(s).encoderPeakContextFrames;
}
int64_t voxtral_stream_encoder_context_frames_retained(const voxtral_stream * s) {
    return stream_encoder_metrics(s).encoderContextFramesRetained;
}
int64_t voxtral_stream_encoder_state_bytes(const voxtral_stream * s) {
    return stream_encoder_metrics(s).encoderStateBytes;
}
int64_t voxtral_stream_encoder_output_accumulated_bytes(const voxtral_stream * s) {
    return stream_encoder_metrics(s).encoderOutputAccumulatedBytes;
}
voxtral_encoder_metrics voxtral_stream_encoder_metrics_full(const voxtral_stream * s) {
    return stream_encoder_metrics(s);
}
const float * voxtral_stream_encoder_output_data(const voxtral_stream * s) {
    return (s && s->enc) ? voxtral_encoder_stream_output(s->enc) : nullptr;
}
int32_t voxtral_stream_encoder_output_frames_count(const voxtral_stream * s) {
    return (s && s->enc) ? voxtral_encoder_stream_output_frames(s->enc) : 0;
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

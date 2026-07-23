// ============================================================================
// Internal streaming session runtime — implementation.
//
// PCM/STFT/log-Mel and the causal per-layer-KV encoder run incrementally during
// feed(). The adapter and decoder are the remaining finish-only stages.
//
// See src/voxtral-stream.h and docs/architecture/streaming-runtime.md.
// ============================================================================

#include "voxtral-stream.h"
#include "voxtral-internal.h"   // voxtral_transcribe_mel_internal (Mel -> text path)

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
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
constexpr size_t kBacklogSamples = 32768;

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
    voxtral_model       * model        = nullptr;  // shared, immutable; for detokenization
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
    bool warmup_complete = false;

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

    struct stage_backlog_series {
        std::array<double, kBacklogSamples> values{};
        std::array<double, kBacklogSamples> audio_seconds{};
        size_t stored = 0;
        uint64_t seen = 0;
        uint64_t deadline_misses = 0;
        double final_ms = 0.0;

        void reset() {
            stored = 0;
            seen = 0;
            deadline_misses = 0;
            final_ms = 0.0;
        }
        void add(double audio_s, double backlog_ms) {
            ++seen;
            final_ms = backlog_ms;
            if (backlog_ms > 0.0) ++deadline_misses;
            if (stored < values.size()) {
                values[stored] = backlog_ms;
                audio_seconds[stored] = audio_s;
                ++stored;
            } else {
                // Deterministic bounded reservoir for streams longer than the
                // required 30-minute matrix. The 30-minute run fits exactly and
                // therefore remains lossless.
                const uint64_t mixed = seen * 0x9e3779b97f4a7c15ULL;
                const uint64_t slot = (mixed ^ (mixed >> 29)) % seen;
                if (slot < values.size()) {
                    values[(size_t) slot] = backlog_ms;
                    audio_seconds[(size_t) slot] = audio_s;
                }
            }
        }
    } encoder_backlog, adapter_backlog, decoder_backlog;

    // ---- Session 7: device-resident incremental adapter + decoder ----------
    // Selected once at create. Incremental is the production default;
    // VOXTRAL_STREAM_DECODER=reference selects the finish-only oracle.
    bool     incremental          = false;
    std::vector<int32_t> prompt_ids;          // BOS + left-pad + delay STREAMING_PADs
    int64_t  adapter_groups_committed = 0;    // audio embeddings written == next group
    int64_t  adapter_commit_calls = 0;
    bool     decoder_prefill_done = false;
    int32_t  decoder_prev_token   = 0;
    int64_t  decoder_position     = 0;        // next decoder position (== audio_pos)
    int64_t  decoder_steps        = 0;
    int64_t  decoder_tokens_emitted = 0;
    int64_t  tokens_before_finish = 0;        // tokens emitted during feed()
    bool     decoder_eos          = false;
    uint64_t token_sequence       = 0;        // strictly monotonic TOKEN sequence
    uint64_t partial_revision     = 0;        // monotonic PARTIAL_TEXT revision
    std::string partial_text;                 // incremental detokenized snapshot
    size_t   partial_stable_bytes = 0;        // UTF-8-safe stable prefix length
    double   first_adapter_commit_ms = 0.0;
    double   first_decoder_step_ms   = 0.0;
    double   first_token_ms          = 0.0;   // first non-special token
    double   first_visible_text_ms   = 0.0;   // first non-empty partial text
    // Eligibility-based latency markers.  Absolute stream time is still
    // reported, but gates use the overhead after the exact audio group needed
    // by an operation became available to the stream.  The ring mirrors the
    // fixed audio-embedding ring, so this telemetry cannot grow with duration.
    std::vector<int64_t> eligibility_absolute_group;
    std::vector<double>  eligibility_arrival_ms;
    double   current_audio_availability_ms       = 0.0;
    double   first_decoder_step_eligibility_ms   = -1.0;
    double   first_decoder_step_overhead_ms      = -1.0;
    double   first_token_eligibility_ms          = -1.0;
    double   first_token_overhead_ms             = -1.0;
    double   first_partial_eligibility_ms        = -1.0;
    double   first_partial_overhead_ms           = -1.0;
    int64_t  token_id_d2h_bytes      = 0;      // 4 bytes per decoder step (argmax readback)

    // ---- Session 7.1: explicit backpressure --------------------------------
    // A decoder step that already ran (KV advanced) but whose mandatory TOKEN
    // event did not fit the bounded event queue is held here so the token is
    // never dropped and the step is never re-run. pump flushes it before doing
    // any new work; feed refuses new audio while it is set (atomic backpressure).
    struct pending_token {
        bool    valid    = false;
        int32_t token    = 0;
        int64_t position = 0;
        bool    special  = false;
    } pending;
    bool     decoder_backpressured   = false;  // event queue full; caller must drain
    bool     finalizing_flush        = false;  // finish(): terminal events bypass the bound
    bool     aggressive_partial_coalescing = false; // one large accepted feed

    // Rolling SHA-256 of the canonical PCM (chunk-invariant; no retention needed).
    Sha256   pcm_sha;
    // Opt-in test diagnostics. Newly produced device-ring rows are copied into
    // one bounded scratch vector and hashed immediately; no utterance-sized
    // encoder/adapter tensor is retained.
    bool     capture_output_sha = false;
    Sha256   encoder_output_sha;
    Sha256   adapter_output_sha;
    std::vector<float> output_sha_scratch;
    int64_t  encoder_sha_rows = 0;
    int64_t  adapter_sha_rows = 0;
    int64_t  output_sha_d2h_bytes = 0;

    // Even-trimmed Mel matrix [n_mel, n_frames] assembled at finish() (batch path).
    std::vector<float> mel_even;
    int32_t            mel_even_frames = 0;

    uint64_t total_samples_received = 0;
    uint64_t total_samples_consumed = 0;
    uint64_t feed_calls             = 0;
    uint64_t inference_runs         = 0;

    // Monotonic timeline for realtime residence/backlog telemetry. The stream
    // stores only aggregate finish breakdowns; encoder frame histograms live in
    // the opt-in per-layer collector.
    int64_t timeline_start_ns       = 0;
    double  finish_frontend_ms     = 0.0;
    double  finish_encoder_ms      = 0.0;
    double  finish_decoder_ms      = 0.0;
    double  first_mel_absolute_ms  = 0.0;

    std::vector<int32_t> tokens;
    std::string          transcript;

    std::deque<voxtral_stream_event> events;
    size_t max_events        = kMaxEvents;   // hard bound; test-overridable
    bool   events_overflowed = false;

    // ---- Session 7.1: event telemetry (all hard-gate: events_dropped == 0) --
    uint64_t events_emitted             = 0;  // total events successfully enqueued
    uint64_t token_events               = 0;
    uint64_t partial_events             = 0;  // partials that were enqueued as new entries
    uint64_t partial_events_coalesced   = 0;  // partials that replaced/were dropped when full
    uint64_t event_queue_high_watermark = 0;  // peak queue depth observed
    uint64_t event_queue_overflow_attempts = 0;  // mandatory pushes that hit the bound (→ backpressure)
    uint64_t events_dropped             = 0;  // mandatory events actually lost — MUST stay 0

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

int64_t stream_now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct context_profile_scope {
    voxtral_context * ctx = nullptr;
    voxtral_profile_stage stage = voxtral_profile_stage::pipeline_feed;
    int64_t start_ns = 0;

    context_profile_scope(voxtral_context * c, voxtral_profile_stage s)
        : ctx(c), stage(s), start_ns(stream_now_ns()) {}
    ~context_profile_scope() {
        if (ctx) {
            voxtral_context_profile_record_internal(
                ctx, stage, (double) (stream_now_ns() - start_ns) / 1e6);
        }
    }
};

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

// Bookkeeping after a successful enqueue: aggregate + per-type counters and the
// running queue-depth high-watermark.
void note_enqueued(voxtral_stream * s, voxtral_stream_event_type type) {
    s->events_emitted++;
    if (type == voxtral_stream_event_type::token) s->token_events++;
    else if (type == voxtral_stream_event_type::partial_text) s->partial_events++;
    if (s->events.size() > s->event_queue_high_watermark)
        s->event_queue_high_watermark = s->events.size();
}

// Bounded push for mandatory events. Returns false WITHOUT enqueuing when the
// queue is at its bound — the caller turns that into explicit backpressure
// (feed → queue_full) and never drops the event. During finish()'s terminal
// flush (finalizing_flush) the small bounded finish tail is always delivered, so
// FINAL_TEXT / COMPLETED / ERROR can never be lost.
bool push_event(voxtral_stream * s, voxtral_stream_event ev) {
    if (!s->finalizing_flush && s->events.size() >= s->max_events) {
        s->events_overflowed = true;
        s->event_queue_overflow_attempts++;
        set_error(s, voxtral_status::limit_exceeded,
                  std::string("event queue full (bound ") + std::to_string(s->max_events) +
                  "): backpressure on " + voxtral_stream_event_name(ev.type) + " event");
        return false;
    }
    const auto type = ev.type;
    s->events.push_back(std::move(ev));
    note_enqueued(s, type);
    return true;
}

void emit_final_and_completed(voxtral_stream * s) {
    context_profile_scope profile(s ? s->ctx : nullptr, voxtral_profile_stage::event_processing);
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
    voxtral_mel_frontend_set_retain_history(s->mel_fe, s->params.retain_mel_history);
    return true;
}

// Inject the streaming left zero-padding into the Mel frontend exactly once,
// before the first real sample. Mirrors the batch left_pad prefix. Bounded: the
// injected silence is emitted to stable Mel frames and the PCM tail compacted
// before this returns, so it does not create a lasting 40960-sample buffer.
void ensure_left_pad(voxtral_stream * s) {
    if (s->mel_fe && !s->left_pad_injected) {
        s->left_pad_injected = true;
        context_profile_scope profile(s->ctx, voxtral_profile_stage::mel_compute);
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
    // Incremental path: mirror encoder output into the device ring from the first
    // batch on, so the adapter can read complete groups on-device. The finish-only
    // reference leaves the ring copy off (no extra per-batch work).
    //
    // The incremental adapter/decoder reads encoder output from that device ring,
    // which ONLY the per-layer KV encoder fills. If the reference (bounded-window
    // recompute) encoder is selected (VOXTRAL_ENCODER_STRATEGY=reference), couple
    // the decoder to the coherent reference finish-only path rather than feed the
    // adapter an empty ring. This is a deterministic configuration coupling, not a
    // runtime error fallback.
    if (s->incremental && !voxtral_encoder_stream_uses_kv(s->enc)) {
        s->incremental = false;
    }
    if (s->incremental) voxtral_ctx_set_enc_out_ring_active(s->ctx, true);
    return true;
}

bool capture_new_encoder_output_sha(voxtral_stream * s) {
    if (!s || !s->capture_output_sha || !s->incremental || !s->enc || !s->ctx) {
        return true;
    }
    const int64_t available = voxtral_encoder_stream_output_frames(s->enc);
    const int64_t count64 = available - s->encoder_sha_rows;
    if (count64 <= 0) return true;
    const int32_t capacity = voxtral_ctx_enc_out_ring_frames(s->ctx);
    if (count64 > capacity || count64 > INT32_MAX) {
        set_error(s, voxtral_status::backend_error,
                  "encoder SHA capture fell behind bounded output ring");
        return false;
    }
    const int32_t count = (int32_t) count64;
    s->output_sha_scratch.resize((size_t) count * VOXTRAL_ENC_DIM);
    if (!voxtral_ctx_read_enc_out_ring_internal(
            s->ctx, s->encoder_sha_rows, count,
            s->output_sha_scratch.data())) {
        set_error(s, voxtral_status::backend_error,
                  "encoder SHA diagnostic readback failed");
        return false;
    }
    const size_t bytes = s->output_sha_scratch.size() * sizeof(float);
    s->encoder_output_sha.update(s->output_sha_scratch.data(), bytes);
    s->output_sha_d2h_bytes += (int64_t) bytes;
    s->encoder_sha_rows = available;
    return true;
}

bool capture_new_adapter_output_sha(voxtral_stream * s,
                                    int64_t start, int32_t count) {
    if (!s || !s->capture_output_sha || !s->incremental || !s->ctx ||
        count <= 0) return true;
    const int32_t capacity = voxtral_ctx_aemb_ring_frames(s->ctx);
    if (count > capacity || start != s->adapter_sha_rows) {
        set_error(s, voxtral_status::backend_error,
                  "adapter SHA capture lost monotonic ring position");
        return false;
    }
    s->output_sha_scratch.resize((size_t) count * VOXTRAL_DEC_DIM);
    if (!voxtral_ctx_read_aemb_ring_internal(
            s->ctx, start, count, s->output_sha_scratch.data())) {
        set_error(s, voxtral_status::backend_error,
                  "adapter SHA diagnostic readback failed");
        return false;
    }
    const size_t bytes = s->output_sha_scratch.size() * sizeof(float);
    s->adapter_output_sha.update(s->output_sha_scratch.data(), bytes);
    s->output_sha_d2h_bytes += (int64_t) bytes;
    s->adapter_sha_rows += count;
    return true;
}

// Hand every newly-stable Mel frame from the frontend to the incremental encoder,
// which runs any encoder chunks that became available. Called after each feed and
// once more (with the finish-flushed frames) at finish().
bool drain_mel_to_encoder(voxtral_stream * s, bool final = false) {
    if (!s->enc || !s->mel_fe) return true;
    const int64_t total = voxtral_mel_frontend_frame_count(s->mel_fe);
    if (total <= s->enc_pushed_frames) return true;
    int64_t base = s->enc_pushed_frames;
    if (base < voxtral_mel_frontend_frames_base(s->mel_fe) || total < base) {
        set_error(s, voxtral_status::backend_error, "incremental Mel history base advanced past encoder cursor");
        return false;
    }
    // The terminal frontend flush is already a bounded suffix. Preserve it as
    // one push so the encoder can batch right-padding work; ordinary feed keeps
    // the fixed absolute cadence below for feed-plan invariance.
    if (final) {
        const int64_t remaining = total - base;
        const float * frames = voxtral_mel_frontend_frame_data(s->mel_fe, base);
        if (!frames || remaining > INT32_MAX ||
            !voxtral_encoder_stream_push_mel_final(
                s->enc, frames, base, (int32_t) remaining)) {
            set_error(s, voxtral_status::backend_error,
                      "incremental encoder rejected terminal Mel frames");
            return false;
        }
        s->enc_pushed_frames = total;
        voxtral_mel_frontend_discard_before(s->mel_fe, total);
        return capture_new_encoder_output_sha(s);
    }

    // Feed the encoder on a fixed absolute Mel cadence: eight Mel frames are
    // exactly four encoder frames, i.e. one future adapter group. A one-shot
    // feed and an 80/160 ms feed therefore expose identical graph-ready
    // boundaries to the KV scheduler instead of letting caller chunk size alter
    // how much future Mel happens to be resident when a graph starts.
    constexpr int32_t kMelDrainSlice = 8;
    while (base < total) {
        const int32_t n = (int32_t) std::min<int64_t>(kMelDrainSlice, total - base);
        const float * frames = voxtral_mel_frontend_frame_data(s->mel_fe, base);
        if (!frames) {
            set_error(s, voxtral_status::backend_error, "incremental Mel frame is outside retained window");
            return false;
        }
        if (!voxtral_encoder_stream_push_mel(s->enc, frames, base, n)) {
            set_error(s, voxtral_status::backend_error, "incremental encoder rejected Mel frames");
            return false;
        }
        if (!capture_new_encoder_output_sha(s)) return false;
        base += n;
        s->enc_pushed_frames = base;
        voxtral_mel_frontend_discard_before(s->mel_fe, base);
    }
    return true;
}

// ============================================================================
// Session 7: device-resident incremental adapter + decoder scheduling (feed()).
// ============================================================================

// Prompt for the realtime decoder: [BOS] + STREAMING_PAD * (left-pad + delay).
// Exactly the tokens run_adapter_and_decode_realtime prefills, so the incremental
// decoder produces a byte-identical token stream.
void build_prompt_ids(voxtral_stream * s) {
    if (!s->prompt_ids.empty()) return;
    s->prompt_ids.push_back(VOXTRAL_TOKEN_BOS);
    for (int32_t i = 0; i < VOXTRAL_N_LEFT_PAD_TOKENS + VOXTRAL_N_DELAY_TOKENS; ++i) {
        s->prompt_ids.push_back(VOXTRAL_TOKEN_STREAMING_PAD);
    }
}

// Length of the longest prefix of `text` that ends on a UTF-8 code-point boundary
// (i.e. excludes a trailing incomplete multi-byte sequence). Detokenization is
// append-only, so everything up to this point is permanently stable.
size_t utf8_stable_prefix(const std::string & text) {
    size_t i = text.size();
    while (i > 0 && (static_cast<unsigned char>(text[i - 1]) & 0xC0) == 0x80) --i; // skip continuations
    if (i == 0) return text.size();
    const unsigned char lead = static_cast<unsigned char>(text[i - 1]);
    size_t need;
    if      ((lead & 0x80) == 0x00) need = 1;
    else if ((lead & 0xE0) == 0xC0) need = 2;
    else if ((lead & 0xF0) == 0xE0) need = 3;
    else if ((lead & 0xF8) == 0xF0) need = 4;
    else                            need = 1;   // stray continuation / invalid lead
    // The final code point starts at i-1; it is complete iff all its bytes are present.
    return (i - 1) + need <= text.size() ? text.size() : (i - 1);
}

// Real-audio end sample for an audio position, derived from the 80 ms cadence and
// the injected left pad. Clamped to 0 for positions inside the left-pad region.
int64_t audio_end_sample_for(int64_t position) {
    const int64_t s = (position + 1) * VOXTRAL_RAW_AUDIO_LENGTH_PER_TOK - kStreamLeftPad;
    return s > 0 ? s : 0;
}

// Push an event, coalescing PARTIAL_TEXT: a full queue keeps only the newest
// revision (partials are a replaceable snapshot, so coalescing/dropping one is
// allowed and is NOT a mandatory-event drop). Mandatory events (token / final /
// error / completed) never coalesce: a full queue returns false so the caller
// can raise explicit backpressure instead of losing the event. The finish()
// terminal flush (finalizing_flush) bypasses the bound entirely.
bool push_event_coalesced(voxtral_stream * s, voxtral_stream_event ev) {
    if (ev.type == voxtral_stream_event_type::partial_text &&
        s->aggressive_partial_coalescing) {
        // A multi-minute one-shot feed cannot let replaceable partial snapshots
        // consume half of the mandatory TOKEN queue. Keep exactly the newest
        // snapshot, moving it behind the current token so observable ordering
        // remains coherent. The queue stays fixed-size; token events are never
        // coalesced or dropped.
        for (auto it = s->events.begin(); it != s->events.end(); ++it) {
            if (it->type == voxtral_stream_event_type::partial_text) {
                s->events.erase(it);
                s->events.push_back(std::move(ev));
                s->partial_events_coalesced++;
                return true;
            }
        }
    }
    if (s->finalizing_flush || s->events.size() < s->max_events) {
        const auto type = ev.type;
        s->events.push_back(std::move(ev));
        note_enqueued(s, type);
        return true;
    }
    if (ev.type == voxtral_stream_event_type::partial_text) {
        for (auto it = s->events.rbegin(); it != s->events.rend(); ++it) {
            if (it->type == voxtral_stream_event_type::partial_text) {
                *it = std::move(ev);                 // keep only the newest revision
                s->partial_events_coalesced++;
                return true;
            }
        }
        s->partial_events_coalesced++;
        return false;   // no prior partial to replace; drop newest (partials are lossy)
    }
    return push_event(s, std::move(ev));   // mandatory: caller raises backpressure
}

// Returns false when the mandatory TOKEN event did not fit the bounded queue; the
// caller must NOT advance the decoder past it (the token is stashed and retried).
// The sequence id is committed only on a successful enqueue, so no gaps appear.
bool emit_token_event(voxtral_stream * s, int32_t token, int64_t position, bool special) {
    voxtral_stream_event ev;
    ev.type                    = voxtral_stream_event_type::token;
    ev.token                   = token;
    ev.text                    = voxtral_token_piece_internal(s->model, token);
    ev.special                 = special;
    ev.sequence                = s->token_sequence + 1;   // tentative
    ev.decoder_position        = position;
    ev.audio_end_sample        = audio_end_sample_for(position);
    ev.emitted_at_monotonic_ns = stream_now_ns();
    ev.t_audio_ms              = (double) ev.audio_end_sample * 1000.0 / (double) s->params.sample_rate;
    if (!push_event_coalesced(s, std::move(ev))) return false;   // queue full → backpressure
    ++s->token_sequence;                                          // commit only on success
    return true;
}

void emit_partial_text_event(voxtral_stream * s, int64_t position) {
    voxtral_stream_event ev;
    ev.type               = voxtral_stream_event_type::partial_text;
    ev.text               = s->partial_text;   // full snapshot
    ev.revision           = ++s->partial_revision;
    ev.stable_prefix_bytes= s->partial_stable_bytes;
    ev.audio_end_sample   = audio_end_sample_for(position);
    ev.t_audio_ms         = (double) ev.audio_end_sample * 1000.0 / (double) s->params.sample_rate;
    push_event_coalesced(s, std::move(ev));
}

// Append one non-special token's bytes to the incremental transcript and refresh
// the UTF-8-safe stable prefix. Byte-identical to decode_tokens() by construction.
void append_token_to_partial(voxtral_stream * s, int32_t token) {
    const std::string & piece = voxtral_token_piece_internal(s->model, token);
    if (!piece.empty()) s->partial_text.append(piece);
    s->partial_stable_bytes = utf8_stable_prefix(s->partial_text);
}

bool token_piece_has_lexical_content(const voxtral_stream * s, int32_t token) {
    if (!s || !s->model) return false;
    const std::string & piece = voxtral_token_piece_internal(s->model, token);
    return std::any_of(piece.begin(), piece.end(), [](unsigned char c) {
        return c >= 0x80 || std::isalnum(c) != 0;
    });
}

double stream_elapsed_ms(const voxtral_stream * s) {
    return s->timeline_start_ns > 0
        ? (double) (stream_now_ns() - s->timeline_start_ns) / 1e6 : 0.0;
}

// Record when each committed audio group became available.  Adapter and decoder
// use the same absolute group index; modulo storage is safe because the decoder
// cannot consume an entry after the bounded audio-embedding ring overwrote it.
void note_group_eligibility(voxtral_stream * s, int64_t start, int32_t count,
                            int32_t capacity) {
    if (!s || capacity <= 0 || count <= 0) return;
    if ((int32_t) s->eligibility_absolute_group.size() != capacity) {
        s->eligibility_absolute_group.assign((size_t) capacity, -1);
        s->eligibility_arrival_ms.assign((size_t) capacity, 0.0);
    }
    for (int32_t i = 0; i < count; ++i) {
        const int64_t absolute = start + i;
        const size_t slot = (size_t) (absolute % capacity);
        s->eligibility_absolute_group[slot] = absolute;
        s->eligibility_arrival_ms[slot] = s->current_audio_availability_ms;
    }
}

double group_eligibility_ms(const voxtral_stream * s, int64_t absolute) {
    if (!s || absolute < 0 || s->eligibility_absolute_group.empty()) return -1.0;
    const size_t slot = (size_t) (absolute %
        (int64_t) s->eligibility_absolute_group.size());
    return s->eligibility_absolute_group[slot] == absolute
        ? s->eligibility_arrival_ms[slot] : -1.0;
}

void sample_stage_backlogs(voxtral_stream * s) {
    if (!s || !s->incremental || !s->enc || !s->mel_fe ||
        s->total_samples_received == 0) return;
    constexpr double group_ms =
        (double) VOXTRAL_RAW_AUDIO_LENGTH_PER_TOK * 1000.0 / VOXTRAL_SAMPLE_RATE;
    const double audio_s =
        (double) s->total_samples_received / (double) VOXTRAL_SAMPLE_RATE;

    // A group is backlog only after all Mel needed for its four encoder frames
    // exists. Fixed causal/model latency is residence, not queued work.
    const int64_t stable_mel = voxtral_mel_frontend_frame_count(s->mel_fe);
    const int64_t realizable_frames = stable_mel < 2 ? 0 : (stable_mel - 2) / 2 + 1;
    const int64_t encoder_ready_groups = realizable_frames / VOXTRAL_DOWNSAMPLE_FACTOR;
    const int64_t encoder_done_groups =
        voxtral_encoder_stream_output_frames(s->enc) / VOXTRAL_DOWNSAMPLE_FACTOR;
    const int64_t encoder_queued =
        std::max<int64_t>(0, encoder_ready_groups - encoder_done_groups);
    const int64_t adapter_queued =
        std::max<int64_t>(0, encoder_done_groups - s->adapter_groups_committed);

    const int64_t prompt_prefix = (int64_t) s->prompt_ids.size() - 1;
    const int64_t decoder_ready = s->adapter_groups_committed > prompt_prefix
        ? s->adapter_groups_committed - prompt_prefix : 0;
    const int64_t decoder_done = s->decoder_prefill_done
        ? std::max<int64_t>(0, s->decoder_position - prompt_prefix) : 0;
    const int64_t decoder_queued = std::max<int64_t>(0, decoder_ready - decoder_done);

    s->encoder_backlog.add(audio_s, (double) encoder_queued * group_ms);
    s->adapter_backlog.add(audio_s, (double) adapter_queued * group_ms);
    s->decoder_backlog.add(audio_s, (double) decoder_queued * group_ms);
}

// One scheduler pass: commit every ready adapter group, then run every ready
// decoder step, emitting TOKEN + PARTIAL_TEXT. Called after each feed slice and at
// finish. All three stages stay bounded because they are drained here every slice.
voxtral_status pump_incremental(voxtral_stream * s) {
    if (!s->incremental || !s->enc || !s->ctx) return voxtral_status::ok;
    build_prompt_ids(s);

    const int32_t aemb_cap    = voxtral_ctx_aemb_ring_frames(s->ctx);
    const int64_t enc_frames  = voxtral_encoder_stream_output_frames(s->enc);
    const int64_t avail_groups = enc_frames / VOXTRAL_DOWNSAMPLE_FACTOR;

    // 1. Adapter: commit newly-complete groups, throttled so the audio-embedding
    //    ring never overwrites an embedding the decoder has not consumed yet.
    const int64_t consumed_floor = s->decoder_prefill_done ? s->decoder_position : 0;
    const int64_t committable    = std::min<int64_t>(avail_groups, consumed_floor + aemb_cap);
    if (committable > s->adapter_groups_committed) {
        const int32_t n = (int32_t) (committable - s->adapter_groups_committed);
        if (voxtral_ctx_adapter_commit(s->ctx, s->adapter_groups_committed, n) < 0) {
            set_error(s, voxtral_status::backend_error, "incremental adapter commit failed");
            return voxtral_status::backend_error;
        }
        if (!capture_new_adapter_output_sha(
                s, s->adapter_groups_committed, n)) {
            return voxtral_status::backend_error;
        }
        note_group_eligibility(s, s->adapter_groups_committed, n, aemb_cap);
        if (s->first_adapter_commit_ms == 0.0) s->first_adapter_commit_ms = stream_elapsed_ms(s);
        s->adapter_groups_committed = committable;
        s->adapter_commit_calls++;
    }

    // 2. Decoder: prefill once over the prompt (needs positions [0, L-1)), then step
    //    one token per available audio position (audio_pos == position).
    const int32_t L = (int32_t) s->prompt_ids.size();   // 39
    if (!s->decoder_prefill_done && s->adapter_groups_committed >= (L - 1)) {
        voxtral_ctx_decoder_begin_incremental(s->ctx);
        if (!voxtral_ctx_decoder_prefill_incremental(s->ctx, s->prompt_ids.data(), L - 1)) {
            set_error(s, voxtral_status::backend_error, "incremental decoder prefill failed");
            return voxtral_status::backend_error;
        }
        s->decoder_prefill_done = true;
        s->decoder_position     = L - 1;                // first step is at position L-1
        s->decoder_prev_token   = s->prompt_ids[L - 1];
    }

    // Deliver a token that a decoder step already produced (its KV is committed)
    // and advance the bookkeeping. Returns false when the mandatory TOKEN event
    // does not fit the bounded queue: the token stays stashed, the position does
    // not advance, and the step is never re-run — that is the backpressure point.
    auto commit_pending = [&]() -> bool {
        if (!s->pending.valid) return true;
        context_profile_scope profile(s->ctx, voxtral_profile_stage::event_processing);
        if (!emit_token_event(s, s->pending.token, s->pending.position, s->pending.special)) {
            s->decoder_backpressured = true;
            return false;
        }
        const int32_t tok      = s->pending.token;
        const int64_t position = s->pending.position;
        const bool is_special  = s->pending.special;
        const bool is_lexical  =
            !is_special && token_piece_has_lexical_content(s, tok);
        s->pending.valid = false;
        if (tok == VOXTRAL_TOKEN_EOS) {          // terminal: matches finish-path trailing-EOS drop
            s->decoder_eos = true;
            return true;
        }
        // Token history (drives FINAL_TEXT) + incremental transcript.
        s->tokens.push_back(tok);
        append_token_to_partial(s, tok);
        emit_partial_text_event(s, position);    // lossy snapshot; never backpressures
        s->decoder_tokens_emitted++;
        if (is_lexical && s->first_token_ms == 0.0) {
            s->first_token_ms = stream_elapsed_ms(s);
            s->first_token_eligibility_ms = group_eligibility_ms(s, position);
            if (s->first_token_eligibility_ms >= 0.0) {
                s->first_token_overhead_ms =
                    s->first_token_ms - s->first_token_eligibility_ms;
            }
        }
        if (is_lexical && !s->partial_text.empty() &&
            s->first_visible_text_ms == 0.0) {
            s->first_visible_text_ms = stream_elapsed_ms(s);
            s->first_partial_eligibility_ms = group_eligibility_ms(s, position);
            if (s->first_partial_eligibility_ms >= 0.0) {
                s->first_partial_overhead_ms =
                    s->first_visible_text_ms - s->first_partial_eligibility_ms;
            }
        }
        s->decoder_prev_token = tok;
        s->decoder_position++;
        return true;
    };

    // Resume: flush a token stashed by a previous backpressured pass before doing
    // any new work. Still full → remain backpressured (caller drains and retries).
    if (s->decoder_prefill_done) {
        if (!commit_pending()) return voxtral_status::limit_exceeded;
        s->decoder_backpressured = false;
    }

    if (s->decoder_prefill_done && !s->decoder_eos) {
        const int64_t last_pos = s->adapter_groups_committed - 1;   // last committed audio pos
        const bool unlimited   = (s->params.max_tokens <= 0);
        while (s->decoder_position <= last_pos && !s->decoder_eos &&
               (unlimited || s->decoder_tokens_emitted < (int64_t) s->params.max_tokens)) {
            int32_t tok = 0;
            if (!voxtral_ctx_decoder_step_incremental(
                    s->ctx, s->decoder_prev_token, (int32_t) s->decoder_position, &tok)) {
                set_error(s, voxtral_status::backend_error, "incremental decoder step failed");
                return voxtral_status::backend_error;
            }
            s->decoder_steps++;
            s->token_id_d2h_bytes += (int64_t) sizeof(int32_t);   // 4-byte argmax readback
            if (s->first_decoder_step_ms == 0.0) {
                s->first_decoder_step_ms = stream_elapsed_ms(s);
                s->first_decoder_step_eligibility_ms =
                    group_eligibility_ms(s, s->decoder_position);
                if (s->first_decoder_step_eligibility_ms >= 0.0) {
                    s->first_decoder_step_overhead_ms =
                        s->first_decoder_step_ms -
                        s->first_decoder_step_eligibility_ms;
                }
            }

            s->pending.valid    = true;
            s->pending.token    = tok;
            s->pending.position = s->decoder_position;
            s->pending.special  = (tok == VOXTRAL_TOKEN_EOS || tok == VOXTRAL_TOKEN_BOS ||
                                   tok == VOXTRAL_TOKEN_STREAMING_PAD);
            if (!commit_pending()) return voxtral_status::limit_exceeded;
        }
    }
    return voxtral_status::ok;
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

    // Session 7.1 backpressure resume: a prior feed stalled the decoder because the
    // event queue filled. Flush the stashed token + drain pending steps before
    // accepting anything new. If the queue is still full, reject this feed WITHOUT
    // consuming its audio (atomic; samples_received unchanged) so the caller drains
    // and retries the same buffer. A zero-length feed thus doubles as a drain pump.
    if (s->incremental && s->decoder_backpressured) {
        const voxtral_status rp = pump_incremental(s);
        if (rp == voxtral_status::limit_exceeded) {
            s->feed_calls++;
            return voxtral_status::limit_exceeded;   // still backpressured; drain + retry
        }
        if (rp != voxtral_status::ok) {
            return rp;
        }
        // Resumed with room; fall through (the zero-length early return or the
        // normal audio path below counts this feed call exactly once).
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
    // Overflow-safe cumulative admission bound, not an allocation failure.
    // Production inference retains only bounded frontend/encoder state, and
    // decoder rollover is handled by the fixed circular KV.
    if (count > s->params.max_total_samples ||
        received > s->params.max_total_samples - static_cast<uint64_t>(count)) {
        set_error(s, voxtral_status::limit_exceeded,
                  "stream duration exceeds max_total_samples admission limit");
        return voxtral_status::limit_exceeded;
    }

    // The payload became available when this accepted feed entered the
    // pipeline. Include frontend creation and PCM conversion in residence;
    // timestamping after conversion would make the metric systematically low.
    const int64_t arrival_ns = stream_now_ns();
    if (s->timeline_start_ns == 0) s->timeline_start_ns = arrival_ns;
    s->current_audio_availability_ms =
        (double) (arrival_ns - s->timeline_start_ns) / 1e6;
    const bool previous_partial_mode = s->aggressive_partial_coalescing;
    const uint64_t estimated_token_events =
        ((uint64_t) count + VOXTRAL_RAW_AUDIO_LENGTH_PER_TOK - 1) /
        VOXTRAL_RAW_AUDIO_LENGTH_PER_TOK +
        VOXTRAL_N_RIGHT_PAD_TOKENS + 2;
    s->aggressive_partial_coalescing =
        s->events.size() + 2 * estimated_token_events > s->max_events &&
        s->events.size() + estimated_token_events + 1 <= s->max_events;
    struct partial_mode_guard {
        voxtral_stream * stream;
        bool previous;
        ~partial_mode_guard() {
            stream->aggressive_partial_coalescing = previous;
        }
    } restore_partial_mode{s, previous_partial_mode};
    context_profile_scope pipeline_profile(s->ctx, voxtral_profile_stage::pipeline_feed);

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

    bool backpressured = false;   // event queue filled during this feed's decoder pump
    if (s->mel_fe) {
        // Inference stream: stream samples through the incremental Mel frontend.
        // No full PCM is retained — only the frontend's bounded rolling tail.
        ensure_left_pad(s);
        if (s->first_mel_absolute_ms == 0.0 && voxtral_mel_frontend_frame_count(s->mel_fe) > 0) {
            s->first_mel_absolute_ms = (double) (stream_now_ns() - s->timeline_start_ns) / 1e6;
        }
        if (!ensure_encoder(s)) {
            s->feed_scratch.clear();
            return s->last_status;
        }
        // Bound frontend/encoder transient state for a large compute-only feed;
        // realtime callers normally enter this loop once (80/160 ms chunk).
        constexpr size_t kAudioDrainSlice = 16'000;
        for (size_t audio_off = 0; audio_off < s->feed_scratch.size(); audio_off += kAudioDrainSlice) {
            const size_t audio_n = std::min(kAudioDrainSlice, s->feed_scratch.size() - audio_off);
            const int64_t mel_start_ns = stream_now_ns();
            const bool mel_ok = voxtral_mel_frontend_feed(
                s->mel_fe, s->feed_scratch.data() + audio_off, audio_n);
            voxtral_context_profile_record_internal(
                s->ctx, voxtral_profile_stage::mel_compute,
                (double) (stream_now_ns() - mel_start_ns) / 1e6);
            if (!mel_ok) {
                set_error(s, voxtral_status::backend_error, "incremental Mel frontend rejected feed");
                s->feed_scratch.clear();
                return voxtral_status::backend_error;
            }
            // Drive the incremental causal encoder over the newly-stable Mel
            // frames, so output is produced during feed rather than finish().
            voxtral_encoder_stream_note_audio(
                s->enc,
                (int64_t) (received + (uint64_t) count),
                arrival_ns,
                voxtral_mel_frontend_frame_count(s->mel_fe),
                stream_now_ns(),
                s->timeline_start_ns,
                kStreamLeftPad);
            if (!drain_mel_to_encoder(s)) {
                s->feed_scratch.clear();
                return voxtral_status::backend_error;
            }
            // Session 7: advance the device-resident adapter + decoder during feed,
            // draining every slice so the encoder-output / audio-embedding rings stay
            // bounded. TOKEN and PARTIAL_TEXT are emitted here, not at finish.
            if (s->incremental && !backpressured) {
                const voxtral_status ps = pump_incremental(s);
                if (ps == voxtral_status::limit_exceeded) {
                    // Event queue full: the decoder is stalled with a stashed token.
                    // Keep consuming this feed's audio into the (bounded) Mel/encoder
                    // so nothing is lost, but stop advancing the decoder until the
                    // caller drains. feed returns queue_full below. For realtime
                    // (single-slice) feeds this is the whole feed → fully atomic.
                    backpressured = true;
                } else if (ps != voxtral_status::ok) {
                    s->feed_scratch.clear();
                    return ps;
                }
            }
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
    sample_stage_backlogs(s);
    if (s->state == voxtral_stream_state::created) {
        s->state = voxtral_stream_state::running;
    }
    if (backpressured) {
        // Audio was accepted (Mel/encoder consumed it); the decoder has output
        // pending because the event queue is full. Surface explicit backpressure —
        // the caller drains the event queue and feeds again (a zero-length feed
        // suffices) to resume. last_error carries the reason; do not clear it.
        return voxtral_status::limit_exceeded;
    }
    clear_error(s);
    return voxtral_status::ok;
}

} // namespace

void voxtral_stream_set_timeline_start_internal(voxtral_stream * stream,
                                                int64_t timeline_start_ns) {
    if (!stream || timeline_start_ns <= 0) return;
    // This is an internal/test seam. It is meaningful only before the first
    // feed; a running stream keeps the original anchor for all later chunks.
    if (stream->timeline_start_ns == 0) stream->timeline_start_ns = timeline_start_ns;
}

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
    s->model  = model;

    // Session 7.1: the device-resident incremental adapter + decoder is the
    // production default. Only the explicit reference oracle disables it; any
    // other value (including the legacy explicit "incremental") keeps the
    // incremental path. No silent incremental→reference fallback: a reference run
    // must be asked for. (ensure_encoder additionally couples the decoder to the
    // encoder strategy — the incremental decoder needs the KV encoder ring.)
    s->incremental = true;
    if (const char * mode = std::getenv("VOXTRAL_STREAM_DECODER")) {
        const std::string m = mode;
        if (m == "reference" || m == "finish-only" || m == "finish_only" || m == "oracle") {
            s->incremental = false;
        }
    }
    if (const char * capture = std::getenv("VOXTRAL_CAPTURE_OUTPUT_SHA")) {
        const std::string value = capture;
        s->capture_output_sha =
            value == "1" || value == "true" || value == "yes";
    }

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

voxtral_status voxtral_stream_warmup_internal(voxtral_stream * stream) {
    if (!stream) return voxtral_status::invalid_argument;
    op_guard guard(stream);
    if (!guard.engage()) return voxtral_status::busy;
    if (stream->warmup_complete) return voxtral_status::ok;
    if (stream->state != voxtral_stream_state::created ||
        stream->total_samples_received != 0 || !stream->ctx || !stream->incremental) {
        set_error(stream, voxtral_status::invalid_state,
                  "warmup requires a fresh production incremental stream");
        return voxtral_status::invalid_state;
    }
    if (!ensure_frontend(stream) || !ensure_encoder(stream) || !stream->enc) {
        if (stream->last_status == voxtral_status::ok) {
            set_error(stream, voxtral_status::backend_error,
                      "failed to initialize production warmup state");
        }
        return stream->last_status;
    }
    if (!voxtral_encoder_stream_warmup(stream->enc)) {
        set_error(stream, voxtral_status::backend_error,
                  "production graph warmup failed");
        return voxtral_status::backend_error;
    }
    stream->enc_pushed_frames = 0;
    stream->warmup_complete = true;
    clear_error(stream);
    return voxtral_status::ok;
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
    // finish() delivers a bounded tail (remaining audio positions + EOS) plus the
    // terminal FINAL_TEXT/COMPLETED (or ERROR). These mandatory events must never
    // be dropped, so the terminal flush bypasses the streaming event-queue bound;
    // the queue can exceed it only by this small, bounded finish tail.
    stream->finalizing_flush = true;

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
        const int64_t finish_front_start_ns = stream_now_ns();
        ensure_left_pad(stream);   // no-op if already injected during feed
        const int64_t n_raw     = (int64_t) stream->total_samples_received;
        const int64_t mult      = VOXTRAL_RAW_AUDIO_LENGTH_PER_TOK;
        const int64_t align_pad = (mult - (n_raw % mult)) % mult;
        const int64_t right_pad = align_pad + (int64_t) VOXTRAL_N_RIGHT_PAD_TOKENS * mult;
        voxtral_mel_frontend_feed_silence(stream->mel_fe, (size_t) right_pad);
        voxtral_mel_frontend_finish(stream->mel_fe);
        voxtral_mel_frontend_assemble_even(stream->mel_fe, stream->mel_even, stream->mel_even_frames);
        stream->full_pcm_buffered_at_finish = false;
        stream->finish_frontend_ms = (double) (stream_now_ns() - finish_front_start_ns) / 1e6;
        voxtral_context_profile_record_internal(stream->ctx, voxtral_profile_stage::mel_compute,
                                                stream->finish_frontend_ms);

        if (!ensure_encoder(stream)) {
            throw std::runtime_error(stream->last_error);
        }
        if (stream->enc) {
            // Feed the finish-flushed Mel frames, then finalize the encoder.
            voxtral_encoder_stream_note_audio(stream->enc,
                                              (int64_t) stream->total_samples_received,
                                              stream_now_ns(),
                                              voxtral_mel_frontend_frame_count(stream->mel_fe),
                                              stream_now_ns(),
                                              stream->timeline_start_ns,
                                              kStreamLeftPad);
            const int64_t finish_encoder_start_ns = stream_now_ns();
            if (!drain_mel_to_encoder(stream, /*final=*/true)) {
                throw std::runtime_error(stream->last_error);
            }
            if (!voxtral_encoder_stream_finish(stream->enc)) {
                throw std::runtime_error("incremental encoder finish failed");
            }
            stream->finish_encoder_ms = (double) (stream_now_ns() - finish_encoder_start_ns) / 1e6;
            const int64_t finish_decoder_start_ns = stream_now_ns();
            if (stream->incremental) {
                // Device-resident path: the adapter and decoder already ran during
                // feed(). Only the flushed tail remains — commit its groups and run
                // its decoder steps. No whole-utterance adapter or decoder replay.
                stream->tokens_before_finish = stream->decoder_tokens_emitted;
                const voxtral_status ps = pump_incremental(stream);
                ok = (ps == voxtral_status::ok);
                if (!ok) throw std::runtime_error(stream->last_error);
            } else {
                const float * enc_out    = voxtral_encoder_stream_output(stream->enc);
                const int32_t enc_frames = voxtral_encoder_stream_output_frames(stream->enc);
                ok = voxtral_transcribe_encoder_output_internal(
                    *stream->ctx, enc_out, enc_frames, stream->params.max_tokens, result);
            }
            stream->finish_decoder_ms = (double) (stream_now_ns() - finish_decoder_start_ns) / 1e6;
        } else {
            // Defensive fallback (should not happen for inference streams): run the
            // shared Mel -> text path over the accumulated incremental Mel.
            const int64_t finish_decoder_start_ns = stream_now_ns();
            ok = voxtral_transcribe_mel_internal(*stream->ctx, stream->mel_even.data(),
                                                 stream->mel_even_frames, stream->params.max_tokens, result);
            stream->finish_decoder_ms = (double) (stream_now_ns() - finish_decoder_start_ns) / 1e6;
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

    if (stream->incremental) {
        // Built incrementally during feed/finish; byte-identical to decode_tokens
        // over stream->tokens (which already excludes the terminal EOS).
        stream->transcript = stream->partial_text;
    } else {
        stream->transcript = std::move(result.text);
        stream->tokens     = std::move(result.tokens);
    }
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
    stream->encoder_backlog.reset();
    stream->adapter_backlog.reset();
    stream->decoder_backlog.reset();
    stream->left_pad_injected            = false;
    stream->full_pcm_buffered_at_finish  = false;
    // Session 7 incremental adapter/decoder state. The device rings are append-only
    // and only ever read at written positions, so resetting the counters is enough;
    // detach the shared decoder from this stream's audio ring and clear its KV.
    if (stream->ctx) {
        voxtral_ctx_decoder_reset_incremental(stream->ctx);
        voxtral_context_profile_reset_internal(stream->ctx);
    }
    stream->adapter_groups_committed = 0;
    stream->adapter_commit_calls     = 0;
    stream->decoder_prefill_done     = false;
    stream->decoder_prev_token       = 0;
    stream->decoder_position         = 0;
    stream->decoder_steps            = 0;
    stream->decoder_tokens_emitted   = 0;
    stream->tokens_before_finish     = 0;
    stream->decoder_eos              = false;
    stream->token_sequence           = 0;
    stream->partial_revision         = 0;
    stream->partial_text.clear();
    stream->partial_stable_bytes     = 0;
    stream->first_adapter_commit_ms  = 0.0;
    stream->first_decoder_step_ms    = 0.0;
    stream->first_token_ms           = 0.0;
    stream->first_visible_text_ms    = 0.0;
    std::fill(stream->eligibility_absolute_group.begin(),
              stream->eligibility_absolute_group.end(), -1);
    std::fill(stream->eligibility_arrival_ms.begin(),
              stream->eligibility_arrival_ms.end(), 0.0);
    stream->current_audio_availability_ms       = 0.0;
    stream->first_decoder_step_eligibility_ms   = -1.0;
    stream->first_decoder_step_overhead_ms      = -1.0;
    stream->first_token_eligibility_ms          = -1.0;
    stream->first_token_overhead_ms             = -1.0;
    stream->first_partial_eligibility_ms        = -1.0;
    stream->first_partial_overhead_ms           = -1.0;
    stream->token_id_d2h_bytes       = 0;
    stream->pending                  = {};      // no token in flight
    stream->decoder_backpressured    = false;
    stream->finalizing_flush         = false;
    stream->aggressive_partial_coalescing = false;
    stream->pcm_sha.reset();
    stream->encoder_output_sha.reset();
    stream->adapter_output_sha.reset();
    stream->output_sha_scratch.clear();
    stream->encoder_sha_rows = 0;
    stream->adapter_sha_rows = 0;
    stream->output_sha_d2h_bytes = 0;
    stream->mel_even.clear();
    stream->mel_even_frames        = 0;
    stream->total_samples_received = 0;
    stream->total_samples_consumed = 0;
    stream->feed_calls             = 0;
    stream->inference_runs         = 0;
    stream->timeline_start_ns      = 0;
    stream->finish_frontend_ms    = 0.0;
    stream->finish_encoder_ms     = 0.0;
    stream->finish_decoder_ms     = 0.0;
    stream->first_mel_absolute_ms = 0.0;
    stream->tokens.clear();
    stream->transcript.clear();
    stream->events.clear();
    stream->events_overflowed = false;
    stream->events_emitted             = 0;
    stream->token_events               = 0;
    stream->partial_events             = 0;
    stream->partial_events_coalesced   = 0;
    stream->event_queue_high_watermark = 0;
    stream->event_queue_overflow_attempts = 0;
    stream->events_dropped             = 0;
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
std::string voxtral_stream_encoder_output_sha256(const voxtral_stream * s) {
    return s ? s->encoder_output_sha.hex() : Sha256{}.hex();
}
std::string voxtral_stream_adapter_output_sha256(const voxtral_stream * s) {
    return s ? s->adapter_output_sha.hex() : Sha256{}.hex();
}
int64_t voxtral_stream_encoder_output_sha_rows(const voxtral_stream * s) {
    return s ? s->encoder_sha_rows : 0;
}
int64_t voxtral_stream_adapter_output_sha_rows(const voxtral_stream * s) {
    return s ? s->adapter_sha_rows : 0;
}
int64_t voxtral_stream_output_sha_d2h_bytes(const voxtral_stream * s) {
    return s ? s->output_sha_d2h_bytes : 0;
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
bool voxtral_stream_mel_history_retained(const voxtral_stream * s) {
    return s && s->mel_fe ? (s->params.retain_mel_history && voxtral_mel_frontend_frames_base(s->mel_fe) == 0) : false;
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
int64_t voxtral_stream_decoder_kv_allocated_bytes(const voxtral_stream * s) {
    return s ? voxtral_context_decoder_kv_bytes_internal(s->ctx) : 0;
}
double voxtral_stream_finish_frontend_ms(const voxtral_stream * s) {
    return s ? s->finish_frontend_ms : 0.0;
}
double voxtral_stream_finish_encoder_ms(const voxtral_stream * s) {
    return s ? s->finish_encoder_ms : 0.0;
}
double voxtral_stream_finish_decoder_ms(const voxtral_stream * s) {
    return s ? s->finish_decoder_ms : 0.0;
}
double voxtral_stream_first_mel_absolute_ms(const voxtral_stream * s) {
    return s ? s->first_mel_absolute_ms : 0.0;
}
const float * voxtral_stream_encoder_output_data(const voxtral_stream * s) {
    return (s && s->enc) ? voxtral_encoder_stream_output(s->enc) : nullptr;
}
int32_t voxtral_stream_encoder_output_frames_count(const voxtral_stream * s) {
    return (s && s->enc) ? voxtral_encoder_stream_output_frames(s->enc) : 0;
}

// --- Session 7: incremental adapter + decoder introspection ----------------
bool voxtral_stream_uses_incremental_decode(const voxtral_stream * s) {
    return s ? s->incremental : false;
}
int64_t voxtral_stream_adapter_groups_committed(const voxtral_stream * s) {
    return s ? s->adapter_groups_committed : 0;
}
int64_t voxtral_stream_adapter_commit_calls(const voxtral_stream * s) {
    return s ? s->adapter_commit_calls : 0;
}
int64_t voxtral_stream_decoder_steps(const voxtral_stream * s) {
    return s ? s->decoder_steps : 0;
}
int64_t voxtral_stream_decoder_tokens_emitted(const voxtral_stream * s) {
    return s ? s->decoder_tokens_emitted : 0;
}
int64_t voxtral_stream_decoder_position(const voxtral_stream * s) {
    return s ? s->decoder_position : 0;
}
bool voxtral_stream_decoder_prefill_complete(const voxtral_stream * s) {
    return s ? s->decoder_prefill_done : false;
}
int64_t voxtral_stream_tokens_before_finish(const voxtral_stream * s) {
    return s ? s->tokens_before_finish : 0;
}
int64_t voxtral_stream_tokens_flushed_at_finish(const voxtral_stream * s) {
    return s ? (s->decoder_tokens_emitted - s->tokens_before_finish) : 0;
}
double voxtral_stream_first_adapter_commit_ms(const voxtral_stream * s) {
    return s ? s->first_adapter_commit_ms : 0.0;
}
double voxtral_stream_first_decoder_step_ms(const voxtral_stream * s) {
    return s ? s->first_decoder_step_ms : 0.0;
}
double voxtral_stream_first_token_ms(const voxtral_stream * s) {
    return s ? s->first_token_ms : 0.0;
}
double voxtral_stream_first_visible_text_ms(const voxtral_stream * s) {
    return s ? s->first_visible_text_ms : 0.0;
}
double voxtral_stream_first_decoder_step_eligibility_ms(const voxtral_stream * s) {
    return s ? s->first_decoder_step_eligibility_ms : -1.0;
}
double voxtral_stream_first_decoder_step_overhead_ms(const voxtral_stream * s) {
    return s ? s->first_decoder_step_overhead_ms : -1.0;
}
double voxtral_stream_first_token_eligibility_ms(const voxtral_stream * s) {
    return s ? s->first_token_eligibility_ms : -1.0;
}
double voxtral_stream_first_token_overhead_ms(const voxtral_stream * s) {
    return s ? s->first_token_overhead_ms : -1.0;
}
double voxtral_stream_first_partial_eligibility_ms(const voxtral_stream * s) {
    return s ? s->first_partial_eligibility_ms : -1.0;
}
double voxtral_stream_first_partial_overhead_ms(const voxtral_stream * s) {
    return s ? s->first_partial_overhead_ms : -1.0;
}
int64_t voxtral_stream_adapter_input_d2h_bytes(const voxtral_stream * s) {
    (void) s; return 0;   // adapter reads the encoder-output ring on-device
}
int64_t voxtral_stream_adapter_output_d2h_bytes(const voxtral_stream * s) {
    (void) s; return 0;   // adapter writes the audio-embedding ring on-device
}
int64_t voxtral_stream_logits_d2h_bytes(const voxtral_stream * s) {
    (void) s; return 0;   // steps read back only the argmax token; prefill logits skipped
}
int64_t voxtral_stream_token_id_d2h_bytes(const voxtral_stream * s) {
    return s ? s->token_id_d2h_bytes : 0;
}
uint64_t voxtral_stream_partial_text_revision(const voxtral_stream * s) {
    return s ? s->partial_revision : 0;
}

// Session 7.1: active decoder path. "incremental" is the production default;
// "reference" is the finish-only oracle (env override, or the coupled fallback
// when the reference encoder is selected). Meaningful once the encoder is created
// (first feed); reflects the env choice before then.
const char * voxtral_stream_decoder_mode(const voxtral_stream * s) {
    return (s && s->incremental) ? "incremental" : "reference";
}
// Actual encoder-output device->host bytes performed by this stream. Hard gate:
// 0 in the incremental production path (the adapter reads the device ring).
int64_t voxtral_stream_encoder_output_d2h_bytes(const voxtral_stream * s) {
    return stream_encoder_metrics(s).encoderOutputD2hBytes;
}

// ---- Event-queue telemetry (events_dropped is a hard gate == 0) ------------
uint64_t voxtral_stream_events_emitted(const voxtral_stream * s) {
    return s ? s->events_emitted : 0;
}
uint64_t voxtral_stream_token_events(const voxtral_stream * s) {
    return s ? s->token_events : 0;
}
uint64_t voxtral_stream_partial_events(const voxtral_stream * s) {
    return s ? s->partial_events : 0;
}
uint64_t voxtral_stream_partial_events_coalesced(const voxtral_stream * s) {
    return s ? s->partial_events_coalesced : 0;
}
uint64_t voxtral_stream_event_queue_high_watermark(const voxtral_stream * s) {
    return s ? s->event_queue_high_watermark : 0;
}
uint64_t voxtral_stream_event_queue_overflow_attempts(const voxtral_stream * s) {
    return s ? s->event_queue_overflow_attempts : 0;
}
uint64_t voxtral_stream_events_dropped(const voxtral_stream * s) {
    return s ? s->events_dropped : 0;
}

static voxtral_backlog_metrics backlog_metrics_from(
    const voxtral_stream::stage_backlog_series & series) {
    voxtral_backlog_metrics out;
    out.count = series.seen;
    out.finalMs = series.final_ms;
    out.deadlineMisses = series.deadline_misses;
    out.deadlineMissRate = series.seen
        ? (double) series.deadline_misses / (double) series.seen : 0.0;
    if (series.stored == 0) return out;

    std::vector<double> sorted(
        series.values.begin(), series.values.begin() + (std::ptrdiff_t) series.stored);
    std::sort(sorted.begin(), sorted.end());
    auto pct = [&](double p) {
        const double x = p * (double) (sorted.size() - 1);
        const size_t lo = (size_t) x;
        const size_t hi = std::min(lo + 1, sorted.size() - 1);
        return sorted[lo] + (sorted[hi] - sorted[lo]) * (x - (double) lo);
    };
    out.p50Ms = pct(0.50);
    out.p95Ms = pct(0.95);
    out.p99Ms = pct(0.99);
    out.maxMs = sorted.back();

    if (series.stored > 1) {
        double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
        for (size_t i = 0; i < series.stored; ++i) {
            const double x = series.audio_seconds[i];
            const double y = series.values[i];
            sx += x; sy += y; sxx += x * x; sxy += x * y;
        }
        const double n = (double) series.stored;
        const double denom = n * sxx - sx * sx;
        if (denom > 0.0) out.slopeMsPerSec = (n * sxy - sx * sy) / denom;
    }
    return out;
}

voxtral_backlog_metrics voxtral_stream_encoder_backlog(const voxtral_stream * s) {
    return s ? backlog_metrics_from(s->encoder_backlog) : voxtral_backlog_metrics{};
}
voxtral_backlog_metrics voxtral_stream_adapter_backlog(const voxtral_stream * s) {
    return s ? backlog_metrics_from(s->adapter_backlog) : voxtral_backlog_metrics{};
}
voxtral_backlog_metrics voxtral_stream_decoder_backlog(const voxtral_stream * s) {
    return s ? backlog_metrics_from(s->decoder_backlog) : voxtral_backlog_metrics{};
}

// Explicit backpressure state (maps the most recent operation's status onto the
// documented feed contract: queue_full = drain events and retry).
voxtral_stream_feed_status voxtral_stream_last_feed_status(const voxtral_stream * s) {
    if (!s) return voxtral_stream_feed_status::failed;
    switch (s->last_status) {
        case voxtral_status::ok:             return voxtral_stream_feed_status::ok;
        case voxtral_status::limit_exceeded: return voxtral_stream_feed_status::queue_full;
        case voxtral_status::busy:           return voxtral_stream_feed_status::would_block;
        case voxtral_status::cancelled:      return voxtral_stream_feed_status::cancelled;
        default:                             return voxtral_stream_feed_status::failed;
    }
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

#ifndef VOXTRAL_STREAM_INTERNAL_H
#define VOXTRAL_STREAM_INTERNAL_H

// ============================================================================
// Private streaming-runtime definitions shared across the streaming modules
// (voxtral-stream.cpp and voxtral-stream-{frontend,decoder,events,telemetry,
// diagnostics}.cpp). This header is INTERNAL to those translation units: it is
// not part of include/voxtral.h nor of the src/voxtral-stream.h contract, and
// none of its declarations are a public or cross-project ABI.
//
// It holds:
//   * the internal event-queue / backlog bounds;
//   * the `voxtral_stream` object and its per-subsystem state;
//   * a handful of small shared inline helpers (error state, time, the
//     reentrancy guard and the profiler scope);
//   * declarations of the internal entry points each streaming module exposes
//     to the others (see the module .cpp files for the implementations).
// ============================================================================

#include "voxtral-stream.h"           // voxtral_stream_* API, event/params/state types
#include "voxtral-internal.h"         // encoder-stream, profiler, ctx incremental entry points
#include "voxtral-stream-diagnostics.h"  // Sha256

#include <array>
#include <chrono>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

// ----------------------------------------------------------------------------
// Bounds
// ----------------------------------------------------------------------------

// Never grow a single feed's copy past what a float vector can address; also a
// sanity ceiling that keeps overflow arithmetic well-defined.
inline constexpr uint64_t kMaxFeedSamples = 64ull * 1024 * 1024;   // 64M samples / feed

// Default ordinary-output bound for the event queue. Incremental TOKEN/PARTIAL
// output can reach it when the caller stops polling; mandatory output then
// backpressures the producer instead of being dropped.
inline constexpr size_t kMaxEvents = 4096;
// Public API v1 reserves this fixed number of slots for the at-most-18-position
// right-pad decoder tail plus TOKEN/PARTIAL and FINAL/COMPLETED markers.
inline constexpr size_t kTerminalEventHeadroom = 64;
static_assert(
    kTerminalEventHeadroom >=
        2u * (static_cast<size_t>(VOXTRAL_N_RIGHT_PAD_TOKENS) + 1u) + 2u,
    "terminal event reserve must hold TOKEN/PARTIAL tail plus FINAL/COMPLETED");
inline constexpr size_t kBacklogSamples = 32768;

// Streaming left/right zero padding (mirrors the batch pad_audio_streaming in
// voxtral_transcribe_from_audio): left = 32 tokens, right = align-to-1280 + 17
// tokens, one token = VOXTRAL_RAW_AUDIO_LENGTH_PER_TOK raw samples.
inline constexpr int64_t kStreamLeftPad =
    (int64_t) VOXTRAL_N_LEFT_PAD_TOKENS * VOXTRAL_RAW_AUDIO_LENGTH_PER_TOK;

// ----------------------------------------------------------------------------
// Struct-independent time helpers.
// ----------------------------------------------------------------------------
inline double samples_to_ms(uint64_t samples, int32_t sample_rate) {
    if (sample_rate <= 0) return 0.0;
    // Derived on demand from the 64-bit count — no floating-point accumulation.
    return static_cast<double>(samples) * 1000.0 / static_cast<double>(sample_rate);
}

inline int64_t stream_now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Bounded per-stage backlog reservoir (fixed memory; deterministic sampling).
// Backlog is work still queued after the next capture deadline; the reservoir
// stores at most kBacklogSamples samples (more than a 30-minute run) losslessly.
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
};

// ----------------------------------------------------------------------------
// The stream object. Per-subsystem mutable state is grouped into typed
// sub-structs; ctx / model / owns_context / params stay at the top level. Field
// names are unchanged, so each module accesses e.g. s->decoder.decoder_position
// or s->events.queue. Ownership: the stream owns its context (iff owns_context),
// the Mel frontend and the encoder stream; it never owns the shared model.
// ----------------------------------------------------------------------------
struct voxtral_stream {
    // Execution engine. Owned by this stream iff `owns_context` (created from a
    // model). ctx may be null for lifecycle-only streams (model == nullptr).
    voxtral_context     * ctx          = nullptr;
    voxtral_model       * model        = nullptr;  // shared, immutable; for detokenization
    bool                  owns_context = false;
    voxtral_stream_params_internal params;

    // ---- Lifecycle / state machine ----------------------------------------
    struct lifecycle_state {
        voxtral_stream_state_internal state       = voxtral_stream_state_internal::created;
        voxtral_status_internal       last_status = voxtral_status_internal::ok;
        std::string          last_error;
        // Non-blocking reentrancy/concurrency guard (see the threading
        // contract). Set while a mutating entry point runs on this stream.
        bool in_operation      = false;
        bool cancel_requested  = false;
        bool cancelled_emitted = false;
        bool warmup_complete   = false;
        uint64_t total_samples_received = 0;
        uint64_t total_samples_consumed = 0;
        uint64_t feed_calls             = 0;
        uint64_t inference_runs         = 0;
        // Test seam: invoked once inside finish() while state == finishing.
        voxtral_stream_finishing_hook_fn finishing_hook = nullptr;
        void *                           finishing_hook_user = nullptr;
    } lifecycle;

    // ---- Audio frontend (incremental Mel + causal encoder) ----------------
    struct frontend_state {
        // Canonical PCM buffer. Used ONLY by lifecycle-only streams (no
        // context): inference streams route samples straight into the
        // incremental Mel frontend and never retain the full PCM here.
        std::vector<float> pcm;
        // Scratch reused each feed to convert PCM16/f32 to canonical float32.
        std::vector<float> feed_scratch;
        // Incremental Mel frontend (inference streams only; null otherwise).
        voxtral_mel_frontend * mel_fe = nullptr;
        bool left_pad_injected            = false;  // left zero-pad injected once
        bool full_pcm_buffered_at_finish  = false;
        // Incremental causal encoder (inference streams only; null otherwise).
        voxtral_encoder_stream * enc = nullptr;
        int64_t enc_pushed_frames    = 0;   // stable Mel frames handed to the encoder
        // Even-trimmed Mel matrix [n_mel, n_frames] assembled at finish().
        std::vector<float> mel_even;
        int32_t            mel_even_frames = 0;
    } frontend;

    // ---- Device-resident incremental adapter + decoder + detokenizer ------
    struct decoder_state {
        // Selected once at create. Incremental is the production default;
        // VOXTRAL_STREAM_DECODER=reference selects the finish-only oracle.
        bool incremental = false;
        std::vector<int32_t> prompt_ids;          // BOS + left-pad + delay STREAMING_PADs
        int64_t adapter_groups_committed = 0;     // audio embeddings written == next group
        int64_t adapter_commit_calls     = 0;
        bool    decoder_prefill_done     = false;
        int32_t decoder_prev_token       = 0;
        int64_t decoder_position         = 0;     // next decoder position (== audio_pos)
        int64_t decoder_steps            = 0;
        int64_t decoder_tokens_emitted   = 0;
        int64_t tokens_before_finish     = 0;     // tokens emitted during feed()
        bool    decoder_eos              = false;
        uint64_t token_sequence          = 0;     // strictly monotonic TOKEN sequence
        uint64_t partial_revision        = 0;     // monotonic PARTIAL_TEXT revision
        std::string partial_text;                 // incremental detokenized snapshot
        size_t   partial_stable_bytes    = 0;     // UTF-8-safe stable prefix length
        // A decoder step that already ran (KV advanced) but whose mandatory
        // TOKEN event did not fit the bounded queue is held here so the token is
        // never dropped and the step is never re-run (atomic backpressure).
        struct pending_token {
            bool    valid    = false;
            int32_t token    = 0;
            int64_t position = 0;
            bool    special  = false;
        } pending;
        std::vector<int32_t> tokens;              // token history (drives FINAL_TEXT)
        std::string          transcript;
    } decoder;

    // ---- Event queue + ordering / backpressure ----------------------------
    struct event_state {
        std::deque<voxtral_stream_event_internal> queue;
        size_t max_events        = kMaxEvents;    // hard bound; test-overridable
        bool   events_overflowed = false;
        // Event telemetry (all hard-gate: events_dropped == 0).
        uint64_t events_emitted             = 0;  // total events successfully enqueued
        uint64_t token_events               = 0;
        uint64_t partial_events             = 0;  // partials enqueued as new entries
        uint64_t partial_events_coalesced   = 0;  // partials replaced/dropped when full
        uint64_t event_queue_high_watermark = 0;  // peak queue depth observed
        uint64_t event_queue_overflow_attempts = 0;  // mandatory pushes that hit the bound
        uint64_t events_dropped             = 0;  // mandatory events actually lost — MUST stay 0
        uint64_t public_poll_sequence        = 0;  // all public events, reset-scoped
        bool decoder_backpressured         = false;  // event queue full; caller must drain
        bool finalizing_flush              = false;  // finish/cancel: allow fixed terminal reserve
        bool aggressive_partial_coalescing = false;  // one large accepted feed
    } events;

    // ---- Latency / backlog telemetry --------------------------------------
    struct telemetry_state {
        stage_backlog_series encoder_backlog, adapter_backlog, decoder_backlog;
        // Monotonic timeline for realtime residence/backlog telemetry.
        int64_t timeline_start_ns    = 0;
        double  first_adapter_commit_ms = 0.0;
        double  first_decoder_step_ms   = 0.0;
        double  first_token_ms          = 0.0;   // first non-special token
        double  first_visible_text_ms   = 0.0;   // first non-empty partial text
        // Eligibility-based latency markers. The ring mirrors the fixed
        // audio-embedding ring, so this telemetry cannot grow with duration.
        std::vector<int64_t> eligibility_absolute_group;
        std::vector<double>  eligibility_arrival_ms;
        double  current_audio_availability_ms     = 0.0;
        double  first_decoder_step_eligibility_ms = -1.0;
        double  first_decoder_step_overhead_ms    = -1.0;
        double  first_token_eligibility_ms        = -1.0;
        double  first_token_overhead_ms           = -1.0;
        double  first_partial_eligibility_ms      = -1.0;
        double  first_partial_overhead_ms         = -1.0;
        int64_t token_id_d2h_bytes    = 0;        // 4 bytes per decoder step (argmax readback)
        double  finish_frontend_ms    = 0.0;
        double  finish_encoder_ms     = 0.0;
        double  finish_decoder_ms     = 0.0;
        double  first_mel_absolute_ms = 0.0;
    } telemetry;

    // ---- Diagnostic output hashing (opt-in) -------------------------------
    struct diagnostics_state {
        // Rolling SHA-256 of the canonical PCM (always maintained; no retention).
        Sha256 pcm_sha;
        // Opt-in test diagnostics. Newly produced device-ring rows are copied
        // into one bounded scratch vector and hashed immediately; no
        // utterance-sized encoder/adapter tensor is retained.
        bool   capture_output_sha = false;
        Sha256 encoder_output_sha;
        Sha256 adapter_output_sha;
        std::vector<float> output_sha_scratch;
        int64_t encoder_sha_rows     = 0;
        int64_t adapter_sha_rows     = 0;
        int64_t output_sha_d2h_bytes = 0;
    } diagnostics;
};

// ----------------------------------------------------------------------------
// Shared inline helpers (error state, reentrancy guard, profiler scope).
// ----------------------------------------------------------------------------
inline void set_error(voxtral_stream * s, voxtral_status_internal status, const std::string & msg) {
    s->lifecycle.last_status = status;
    s->lifecycle.last_error  = msg;
}

inline void clear_error(voxtral_stream * s) {
    s->lifecycle.last_status = voxtral_status_internal::ok;
    s->lifecycle.last_error.clear();
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
        if (s->lifecycle.in_operation) return false;
        s->lifecycle.in_operation = true;
        engaged = true;
        return true;
    }
    ~op_guard() { if (engaged) s->lifecycle.in_operation = false; }
};

// ============================================================================
// Internal module entry points. Each streaming module (.cpp) implements the
// functions in its section and may call those declared in the others; the
// dependency graph is acyclic (lifecycle -> frontend/decoder/events/telemetry/
// diagnostics; frontend -> diagnostics; decoder -> events/telemetry/diagnostics).
// Declarations are added here as each module is extracted into its own TU.
// ============================================================================

// ---- diagnostics (voxtral-stream-diagnostics.cpp) --------------------------
// Hash newly produced encoder / adapter device-ring rows (opt-in; bounded
// readback). Return false on a readback / ring-position error (error set).
bool capture_new_encoder_output_sha(voxtral_stream * s);
bool capture_new_adapter_output_sha(voxtral_stream * s, int64_t start, int32_t count);

// ---- events (voxtral-stream-events.cpp) ------------------------------------
// Bounded push for mandatory events; false without enqueuing when the queue is
// at its bound (caller raises backpressure). Terminal/error/token emitters used
// by the lifecycle and decoder modules.
bool push_event(voxtral_stream * s, voxtral_stream_event_internal ev);
void emit_final_and_completed(voxtral_stream * s);
void emit_error(voxtral_stream * s, voxtral_status_internal status, const std::string & msg);
bool emit_token_event(voxtral_stream * s, int32_t token, int64_t position, bool special);
void emit_partial_text_event(voxtral_stream * s, int64_t position);

// ---- telemetry (voxtral-stream-telemetry.cpp) ------------------------------
// Monotonic timeline, per-group eligibility ring and bounded per-stage backlog
// sampling. Driven by feed (lifecycle) and pump (decoder).
double stream_elapsed_ms(const voxtral_stream * s);
void note_group_eligibility(voxtral_stream * s, int64_t start, int32_t count, int32_t capacity);
double group_eligibility_ms(const voxtral_stream * s, int64_t absolute);
void sample_stage_backlogs(voxtral_stream * s);

// ---- frontend (voxtral-stream-frontend.cpp) --------------------------------
// Incremental PCM -> Mel -> encoder drain and the frontend/encoder metric
// adapters. Driven by feed/finish (lifecycle).
bool ensure_frontend(voxtral_stream * s);
void ensure_left_pad(voxtral_stream * s);
bool ensure_encoder(voxtral_stream * s);
bool drain_mel_to_encoder(voxtral_stream * s, bool final = false);
voxtral_mel_metrics stream_mel_metrics(const voxtral_stream * s);
voxtral_encoder_metrics stream_encoder_metrics(const voxtral_stream * s);

// ---- decoder (voxtral-stream-decoder.cpp) ----------------------------------
// One incremental adapter+decoder scheduler pass. Driven by feed/finish
// (lifecycle); emits TOKEN/PARTIAL_TEXT via the events module.
voxtral_status_internal pump_incremental(voxtral_stream * s);

#endif // VOXTRAL_STREAM_INTERNAL_H

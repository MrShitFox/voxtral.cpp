#ifndef VOXTRAL_STREAM_H
#define VOXTRAL_STREAM_H

// ============================================================================
// Internal streaming session runtime (v2)
// ----------------------------------------------------------------------------
// This header is INTERNAL and UNSTABLE. It is intentionally not part of the
// public C++ surface in include/voxtral.h. The public streaming C ABI and the
// WebSocket server remain out of scope for this session (see
// docs/architecture/streaming-runtime.md, "Implementation status").
//
// What this layer provides today:
//   * an opaque-ish `voxtral_stream` object with a strict lifecycle;
//   * a shared, immutable `voxtral_model` and a per-stream, mutable
//     `voxtral_context` OWNED by the stream (created via the existing
//     model/context factory, freed on destroy);
//   * incremental PCM16 / float32 feeding with 64-bit sample accounting;
//   * a strictly bounded internal event queue (lifecycle + error events);
//   * finish/reset/cancel/destroy;
//   * an incremental Mel frontend and per-layer encoder KV scheduler driven by
//     feed() (production default logical=4 / physical=4);
//   * device-resident incremental adapter/decoder execution that emits TOKEN and
//     replaceable PARTIAL_TEXT events while audio is fed.
//
// The end-to-end path is low-latency realtime-capable and does not retain full
// PCM, encoder output, or an utterance-sized event history.
//
// ----------------------------------------------------------------------------
// Ownership model (v1)
// ----------------------------------------------------------------------------
//     voxtral_model        immutable weights + tokenizer + device selection.
//                          SHARED. Outlives every stream created from it. Never
//                          freed by a stream.
//     voxtral_context      the mutable per-inference execution engine: decoder
//                          KV-cache, kv_used, decoder/encoder memory, schedulers,
//                          scratch buffers and the current transcription sizes.
//                          This is genuinely mutable per-inference state, so it
//                          is OWNED exclusively by exactly one stream and freed
//                          when that stream is destroyed.
//     voxtral_stream       owns its context plus its own PCM / counters /
//                          transcript / event queue / lifecycle state.
//
//     model  outlives  stream  ⊃  its own context
//
// One `voxtral_context` is NEVER shared by two streams: each stream creates its
// own via voxtral_init_from_model(). Several streams may be created from the
// same model, sequentially; concurrent execution of streams is NOT supported in
// this session (a separate concurrency session owns that).
//
// ----------------------------------------------------------------------------
// Threading contract (v1): EXTERNALLY SERIALIZED
// ----------------------------------------------------------------------------
// All methods of a single voxtral_stream must be called serially by the caller.
// Concurrent or reentrant calls on the same stream are NOT supported. As a cheap
// diagnostic (not a real synchronization primitive), every mutating entry point
// (feed/finish/reset/cancel) takes a non-blocking guard and returns
// `voxtral_status::busy` if it detects another operation already in progress on
// the same stream. In particular:
//   * finish() is SYNCHRONOUS;
//   * cancel() does NOT interrupt an in-flight finish() — it returns `busy`.
//     In-flight (GGML graph) cancellation is deliberately not implemented;
//   * feed()/reset() called during finish() return `busy`, never mutate state;
//   * destroy() must NOT be called while any other method is running: the caller
//     is responsible for waiting for all operations to complete first. The guard
//     cannot make concurrent destroy/use safe and does not claim to.
// The guard is a best-effort misuse detector, not protection against a genuine
// data race across threads.
// ============================================================================

#include "voxtral.h"   // public API: voxtral_model, voxtral_context, voxtral_context_params,
                        // voxtral_init_from_model, voxtral_free, voxtral_transcribe_audio, constants
#include "voxtral-mel.h"  // incremental Mel frontend + voxtral_mel_metrics
#include "voxtral-internal.h"  // voxtral_encoder_metrics (full encoder KV / reference metrics)

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// ----------------------------------------------------------------------------
// Status codes. Errors never cross a boundary as exceptions; every fallible
// entry point returns one of these and records a human-readable last error on
// the stream (owned by the stream, see voxtral_stream_last_error).
// ----------------------------------------------------------------------------
enum class voxtral_status {
    ok,
    invalid_argument,
    invalid_state,
    unsupported_audio_format,
    cancelled,
    backend_error,
    out_of_memory,     // a genuine allocation failure
    limit_exceeded,    // a documented compatibility limit was hit (NOT an OOM):
                       // the full-buffer max_total_samples cap, or the event
                       // queue hard bound
    busy,              // a concurrent/reentrant call was detected on the stream
    internal_error,
};

// ----------------------------------------------------------------------------
// Lifecycle states.
//
// Transition table (rows = current state, cells = resulting state / effect;
// "err" = call rejected with a status, no state change, never crashes):
//
//   from \ call | feed         | finish            | cancel      | reset    | destroy
//   ------------+--------------+-------------------+-------------+----------+--------
//   created     | running (*)  | completed (empty) | cancelled   | created  | frees
//   running     | running      | completed/failed  | cancelled   | created  | frees
//   finishing   | busy (**)    | busy (**)         | busy (**)   | busy (**)| UB (***)
//   completed   | err          | ok (idempotent)   | err         | created  | frees
//   cancelled   | err          | ok (no inference) | ok (no-op)  | created  | frees
//   failed      | err          | err               | ok (no-op)  | created  | frees
//
// (*)   A feed carrying >=1 sample moves created -> running. A zero-length feed
//       is a successful no-op and does NOT change state or audio position.
// (**)  `finishing` is a transient state entered only for the duration of the
//       synchronous final adapter/decoder inference. Because the API is externally
//       serialized, the ONLY way another call can observe `finishing` is a
//       reentrant call from within finish() (e.g. a callback). The reentrancy
//       guard rejects such calls with `busy` before the state machine runs, so
//       finish() is never converted to `cancelled` mid-flight. There are no
//       hidden transitions: finishing -> completed on success, finishing ->
//       failed on error.
// (***) destroy() during an in-flight operation is caller error (see the
//       threading contract above); it is not made safe here.
// ----------------------------------------------------------------------------
enum class voxtral_stream_state {
    created,
    running,
    finishing,
    completed,
    cancelled,
    failed,
};

// ----------------------------------------------------------------------------
// Internal event queue. token/partial_text are reserved for the future
// incremental decoder and are never emitted by the current finish-only path.
// Every event owns its text (no dangling const char *).
// ----------------------------------------------------------------------------
enum class voxtral_stream_event_type {
    token,
    partial_text,
    final_text,
    error,
    completed,
    cancelled,
};

struct voxtral_stream_event {
    voxtral_stream_event_type type   = voxtral_stream_event_type::final_text;
    int32_t                   token  = 0;      // token id (token events)
    std::string               text;            // owned copy (partial/final text)
    double                    t_audio_ms = 0.0;// audio position, derived from 64-bit count
    int32_t                   error_code = 0;  // voxtral_status value for error events

    // Session 7 incremental streaming payload.
    uint64_t sequence               = 0;   // token events: strictly monotonic id
    int64_t  decoder_position       = 0;   // token events: absolute decoder position
    int64_t  audio_end_sample       = 0;   // token/partial: real-audio end sample (>=0)
    int64_t  emitted_at_monotonic_ns= 0;   // token events: steady_clock at emission
    bool     special                = false;// token events: special/filtered token
    uint64_t revision               = 0;   // partial_text events: monotonic revision
    size_t   stable_prefix_bytes    = 0;   // partial_text events: never splits a UTF-8 codepoint
};

// ----------------------------------------------------------------------------
// Creation parameters. The runtime does not resample or downmix, so the only
// currently supported format is mono 16 kHz. Extra fields are reserved so the
// incremental frontend session can extend this without breaking callers.
// ----------------------------------------------------------------------------
struct voxtral_stream_params {
    int32_t  sample_rate = VOXTRAL_SAMPLE_RATE;   // must equal 16000
    int32_t  channels    = 1;                      // must equal 1
    int32_t  max_tokens  = 0;                       // 0 = decode whole buffer
    // Bounded utterance limit retained for the finish-only decoder. This is an
    // internal compatibility guard, not a public streaming contract: it keeps
    // Admission limit only; inference streams do not retain full PCM. The
    // decoder now has a fixed circular KV, so the default safely admits the
    // required 30-minute acceptance run (57.6M = 60 minutes @ 16 kHz).
    // Exceeding it returns limit_exceeded (NOT out_of_memory).
    uint64_t max_total_samples = 57'600'000ull;
    // Keep complete Mel history only for tensor-parity/debug inspection.
    // Production realtime streams consume and discard stable frames promptly.
    bool retain_mel_history = false;
};

struct voxtral_stream;  // defined in voxtral-stream.cpp

// ----------------------------------------------------------------------------
// Params validation (exposed so tests can assert the specific rejection code).
// Returns ok, unsupported_audio_format or invalid_argument.
// ----------------------------------------------------------------------------
voxtral_status voxtral_stream_params_check(const voxtral_stream_params & params);

// ----------------------------------------------------------------------------
// Lifecycle.
//
// The stream creates and OWNS its own execution context from the shared model
// via voxtral_init_from_model(model, ctx_params). It frees that context on
// destroy and never frees the model. See the ownership model at the top.
//
// Return contract:
//   * nullptr                    -> params rejected (see voxtral_stream_params_check)
//                                   or an allocation failure with no object to
//                                   carry an error.
//   * non-null, state==failed    -> context creation failed; the reason is in
//                                   voxtral_stream_last_error / last_status
//                                   (backend_error). The stream must still be
//                                   destroyed; it owns no context.
//   * non-null, state==created   -> ready to feed.
//
// `model == nullptr` is permitted for pure lifecycle/PCM tests that never run
// inference: the stream then owns no context and finish() only succeeds for
// empty audio (there is nothing to execute otherwise). `ctx_params` is ignored
// in that case.
// ----------------------------------------------------------------------------
voxtral_stream * voxtral_stream_create_internal(
    voxtral_model                * model,
    const voxtral_context_params & ctx_params,
    const voxtral_stream_params  & params);

// Explicit production-graph warmup. Valid only before the first feed (and
// idempotent). Compiles shaders and retains reusable buffers without changing
// audio/token/transcript/event state.
voxtral_status voxtral_stream_warmup_internal(voxtral_stream * stream);

// Incremental feeds. The input buffer is copied into the stream; the caller may
// free or reuse it immediately after return. `samples == nullptr` is only valid
// when `sample_count == 0`.
voxtral_status voxtral_stream_feed_pcm16_internal(
    voxtral_stream * stream, const int16_t * samples, size_t sample_count);

// Canonical float32 samples. Any FINITE value is accepted and stored unchanged:
// the nominal range is [-1, 1] but values outside it are NOT rejected and are
// NEVER silently clamped (this preserves parity with the batch audio path).
// NaN and ±Inf are rejected with invalid_argument and never mutate the buffer.
voxtral_status voxtral_stream_feed_f32_internal(
    voxtral_stream * stream, const float * samples, size_t sample_count);

// Flush the remaining Mel/encoder frames, then run the finish-only adapter and
// decoder once and emit FINAL_TEXT + COMPLETED. Idempotent after completion
// (never re-runs the encoder prefix).
// Synchronous; see the threading contract.
voxtral_status voxtral_stream_finish_internal(voxtral_stream * stream);

// Return the stream to a state equivalent to a freshly created stream with the
// same params and (owned) context: clears PCM, counters, transcript, tokens,
// events, error and cancellation, and RETAINS the PCM buffer's capacity for
// cheap reuse (no shrink_to_fit). Never touches model weights, tokenizer, device
// or the owned context. Intended for cheap stream reuse.
voxtral_status voxtral_stream_reset_internal(voxtral_stream * stream);

// Pre-execution cancellation. A later finish() will not run inference. v1 only
// supports cancelling before finish() runs; a cancel() attempted while finish()
// is in flight returns `busy` and does not change state (in-flight GGML graph
// cancellation is not implemented).
voxtral_status voxtral_stream_cancel_internal(voxtral_stream * stream);

// Frees the stream and its owned context (if any). Never frees the model. The
// caller must ensure no other method is running on this stream (see the
// threading contract). Safe to call on nullptr.
void voxtral_stream_destroy_internal(voxtral_stream * stream);

// ----------------------------------------------------------------------------
// Introspection (read-only; for tests and the internal realtime harness).
// ----------------------------------------------------------------------------
voxtral_stream_state voxtral_stream_get_state       (const voxtral_stream * stream);
voxtral_status       voxtral_stream_last_status      (const voxtral_stream * stream);
const std::string &  voxtral_stream_last_error       (const voxtral_stream * stream);
uint64_t             voxtral_stream_samples_received (const voxtral_stream * stream);
uint64_t             voxtral_stream_samples_consumed (const voxtral_stream * stream);
uint64_t             voxtral_stream_feed_calls       (const voxtral_stream * stream);
uint64_t             voxtral_stream_inference_runs   (const voxtral_stream * stream);
double               voxtral_stream_audio_ms         (const voxtral_stream * stream);
const std::vector<int32_t> & voxtral_stream_tokens   (const voxtral_stream * stream);
const std::string &  voxtral_stream_transcript       (const voxtral_stream * stream);

// True iff the stream created and exclusively owns a voxtral_context.
bool                 voxtral_stream_owns_context     (const voxtral_stream * stream);
// Opaque identity of the owned execution context (nullptr for lifecycle-only
// streams). Only meaningful for identity comparison while the stream is alive.
const void *         voxtral_stream_context_ptr      (const voxtral_stream * stream);

// Canonical PCM view: float32 mono in nominal [-1, 1]. For lifecycle-only streams
// (no context) this is the full accumulated buffer; for inference streams the
// full PCM is NOT retained (only a bounded rolling tail lives inside the Mel
// frontend), so this returns nullptr / 0. Valid until the next feed/reset/destroy.
const float *        voxtral_stream_pcm_data         (const voxtral_stream * stream);
size_t               voxtral_stream_pcm_size         (const voxtral_stream * stream);

// SHA-256 (lowercase hex) of the full canonical PCM, maintained incrementally as
// samples are fed. Byte-for-byte over the float32 samples in feed order, so it is
// invariant to chunk boundaries and does not require retaining the PCM. Empty
// digest string ("e3b0c4...") for an empty stream.
std::string          voxtral_stream_pcm_sha256       (const voxtral_stream * stream);
// Opt-in VOXTRAL_CAPTURE_OUTPUT_SHA diagnostics. These hash newly produced
// device-ring rows through bounded readback; the byte counter is separate from
// the production-path transfer counters.
std::string          voxtral_stream_encoder_output_sha256(const voxtral_stream * stream);
std::string          voxtral_stream_adapter_output_sha256(const voxtral_stream * stream);
int64_t              voxtral_stream_encoder_output_sha_rows(const voxtral_stream * stream);
int64_t              voxtral_stream_adapter_output_sha_rows(const voxtral_stream * stream);
int64_t              voxtral_stream_output_sha_d2h_bytes(const voxtral_stream * stream);

// ----------------------------------------------------------------------------
// Incremental Mel frontend introspection. For inference streams these track the
// true incremental STFT/log-Mel state; for lifecycle-only streams they are all
// zero / false (no frontend runs). See docs/architecture/streaming-runtime.md.
// ----------------------------------------------------------------------------
bool        voxtral_stream_uses_incremental_mel        (const voxtral_stream * stream);
int64_t     voxtral_stream_mel_frames                  (const voxtral_stream * stream);
int64_t     voxtral_stream_mel_frames_before_finish    (const voxtral_stream * stream);
int64_t     voxtral_stream_mel_frames_flushed_at_finish(const voxtral_stream * stream);
int64_t     voxtral_stream_dft_frames_computed         (const voxtral_stream * stream);
int64_t     voxtral_stream_pcm_retained_samples        (const voxtral_stream * stream);
int64_t     voxtral_stream_pcm_peak_retained_samples   (const voxtral_stream * stream);
int64_t     voxtral_stream_pcm_base_sample             (const voxtral_stream * stream);
// True iff the whole PCM was still buffered when finish() ran (always false for
// inference streams; the incremental frontend only keeps a bounded tail).
bool        voxtral_stream_full_pcm_buffered_at_finish (const voxtral_stream * stream);
bool        voxtral_stream_mel_history_retained        (const voxtral_stream * stream);

// Assembled even-trimmed Mel matrix [n_mel, n_frames], channel-major, produced by
// finish() (empty / nullptr before finish or for lifecycle-only streams). Borrowed;
// valid until reset/destroy. Used by the acceptance harness for Mel SHA / parity.
const float * voxtral_stream_mel_data       (const voxtral_stream * stream);
int32_t       voxtral_stream_mel_data_frames(const voxtral_stream * stream);

// ----------------------------------------------------------------------------
// Incremental causal encoder introspection. For inference streams these track the
// production per-layer KV scheduler (or the opt-in bounded-recompute reference);
// zero / false for lifecycle-only streams. See
// docs/architecture/streaming-runtime.md.
// ----------------------------------------------------------------------------
bool    voxtral_stream_uses_incremental_encoder        (const voxtral_stream * stream);
int64_t voxtral_stream_encoder_frames                  (const voxtral_stream * stream);
int64_t voxtral_stream_encoder_frames_before_finish    (const voxtral_stream * stream);
int64_t voxtral_stream_encoder_frames_flushed_at_finish(const voxtral_stream * stream);
int64_t voxtral_stream_encoder_executions              (const voxtral_stream * stream);
int64_t voxtral_stream_encoder_input_frames_processed  (const voxtral_stream * stream);
int64_t voxtral_stream_encoder_frames_recomputed       (const voxtral_stream * stream);
int64_t voxtral_stream_encoder_max_window_frames       (const voxtral_stream * stream);
int64_t voxtral_stream_encoder_peak_context_frames     (const voxtral_stream * stream);
int64_t voxtral_stream_encoder_context_frames_retained (const voxtral_stream * stream);
int64_t voxtral_stream_encoder_state_bytes             (const voxtral_stream * stream);
int64_t voxtral_stream_encoder_output_accumulated_bytes(const voxtral_stream * stream);

// Full encoder metrics (strategy identity + per-layer KV work/memory
// instrumentation, or the reference bounded-window fields). For lifecycle-only
// streams every field is zero / default. See voxtral_encoder_metrics.
voxtral_encoder_metrics voxtral_stream_encoder_metrics_full(const voxtral_stream * stream);
int64_t voxtral_stream_decoder_kv_allocated_bytes(const voxtral_stream * stream);
double voxtral_stream_finish_frontend_ms(const voxtral_stream * stream);
double voxtral_stream_finish_encoder_ms (const voxtral_stream * stream);
double voxtral_stream_finish_decoder_ms (const voxtral_stream * stream);
double voxtral_stream_first_mel_absolute_ms(const voxtral_stream * stream);

// Accumulated encoder output [enc_dim, frames], channel-major (borrowed; valid
// until the next feed/finish/reset/destroy). The incremental counterpart of the
// batch ctx.encoder_output; used by the acceptance harness for encoder tensor
// parity against voxtral_encode_mel_batch_internal.
const float * voxtral_stream_encoder_output_data        (const voxtral_stream * stream);
int32_t       voxtral_stream_encoder_output_frames_count (const voxtral_stream * stream);

// ----------------------------------------------------------------------------
// Session 7: device-resident incremental adapter + decoder introspection. For a
// finish-only stream (VOXTRAL_STREAM_DECODER != "incremental") these are all zero
// / false. `uses_incremental_decode` reports the active path. Adapter groups and
// decoder positions each advance exactly once; the counters below prove no replay.
// ----------------------------------------------------------------------------
bool    voxtral_stream_uses_incremental_decode      (const voxtral_stream * stream);
int64_t voxtral_stream_adapter_groups_committed     (const voxtral_stream * stream);
int64_t voxtral_stream_adapter_commit_calls         (const voxtral_stream * stream);
int64_t voxtral_stream_decoder_steps                (const voxtral_stream * stream);
int64_t voxtral_stream_decoder_tokens_emitted       (const voxtral_stream * stream);
int64_t voxtral_stream_decoder_position             (const voxtral_stream * stream);
bool    voxtral_stream_decoder_prefill_complete      (const voxtral_stream * stream);
int64_t voxtral_stream_tokens_before_finish         (const voxtral_stream * stream);
int64_t voxtral_stream_tokens_flushed_at_finish     (const voxtral_stream * stream);
// Latency markers relative to the stream timeline (ms; 0 until reached).
double  voxtral_stream_first_adapter_commit_ms      (const voxtral_stream * stream);
double  voxtral_stream_first_decoder_step_ms        (const voxtral_stream * stream);
double  voxtral_stream_first_token_ms               (const voxtral_stream * stream);
double  voxtral_stream_first_visible_text_ms        (const voxtral_stream * stream);
double  voxtral_stream_first_decoder_step_eligibility_ms(const voxtral_stream * stream);
double  voxtral_stream_first_decoder_step_overhead_ms(const voxtral_stream * stream);
double  voxtral_stream_first_token_eligibility_ms   (const voxtral_stream * stream);
double  voxtral_stream_first_token_overhead_ms      (const voxtral_stream * stream);
double  voxtral_stream_first_partial_eligibility_ms (const voxtral_stream * stream);
double  voxtral_stream_first_partial_overhead_ms    (const voxtral_stream * stream);
// Device-traffic accounting for the adapter/decoder path (steady-state gates).
int64_t voxtral_stream_adapter_input_d2h_bytes      (const voxtral_stream * stream);
int64_t voxtral_stream_adapter_output_d2h_bytes     (const voxtral_stream * stream);
int64_t voxtral_stream_logits_d2h_bytes             (const voxtral_stream * stream);
int64_t voxtral_stream_token_id_d2h_bytes           (const voxtral_stream * stream);
// Actual encoder-output device->host bytes. Hard gate: 0 in the incremental
// production path (the adapter reads the on-device encoder-output ring); non-zero
// only for the reference finish-only path.
int64_t voxtral_stream_encoder_output_d2h_bytes     (const voxtral_stream * stream);
// The accumulated final transcript's stable partial-text revision count.
uint64_t voxtral_stream_partial_text_revision       (const voxtral_stream * stream);
// Active decoder path: "incremental" (production default) or "reference"
// (finish-only oracle). Reflects the coupled reality after the first feed.
const char * voxtral_stream_decoder_mode            (const voxtral_stream * stream);

// ----------------------------------------------------------------------------
// Session 7.1: event-queue telemetry. events_dropped is a hard gate (== 0):
// mandatory events (token / final_text / completed / error) are never dropped —
// a full queue is surfaced as explicit feed backpressure (queue_full) instead,
// and partials coalesce to the newest revision.
// ----------------------------------------------------------------------------
uint64_t voxtral_stream_events_emitted              (const voxtral_stream * stream);
uint64_t voxtral_stream_token_events                (const voxtral_stream * stream);
uint64_t voxtral_stream_partial_events              (const voxtral_stream * stream);
uint64_t voxtral_stream_partial_events_coalesced    (const voxtral_stream * stream);
uint64_t voxtral_stream_event_queue_high_watermark  (const voxtral_stream * stream);
uint64_t voxtral_stream_event_queue_overflow_attempts(const voxtral_stream * stream);
uint64_t voxtral_stream_events_dropped              (const voxtral_stream * stream);

// Fixed-memory stage backlog telemetry. Backlog is work that remains after the
// next 80 ms capture deadline; slope is least-squares ms/s over the paced
// timeline. The runtime stores at most 32,768 samples per stage (more than a
// 30-minute run) and never allocates per event.
struct voxtral_backlog_metrics {
    uint64_t count = 0;
    double p50Ms = 0.0;
    double p95Ms = 0.0;
    double p99Ms = 0.0;
    double maxMs = 0.0;
    double finalMs = 0.0;
    double slopeMsPerSec = 0.0;
    uint64_t deadlineMisses = 0;
    double deadlineMissRate = 0.0;
};

voxtral_backlog_metrics voxtral_stream_encoder_backlog(const voxtral_stream * stream);
voxtral_backlog_metrics voxtral_stream_adapter_backlog(const voxtral_stream * stream);
voxtral_backlog_metrics voxtral_stream_decoder_backlog(const voxtral_stream * stream);

// ----------------------------------------------------------------------------
// Explicit backpressure state. feed()/finish() still return voxtral_status; this
// projects the most recent operation's status onto the documented feed contract:
//   ok          -> proceed
//   queue_full  -> event queue full; drain events (poll) and feed again (a
//                  zero-length feed suffices) to resume. Audio was NOT lost.
//   would_block -> a concurrent/reentrant call; retry once serialized
//   cancelled / failed -> terminal
// ----------------------------------------------------------------------------
enum class voxtral_stream_feed_status { ok, would_block, queue_full, cancelled, failed };
voxtral_stream_feed_status voxtral_stream_last_feed_status(const voxtral_stream * stream);

// ----------------------------------------------------------------------------
// Events. poll copies and removes the front event, returning false when empty.
// Repeated polling after drain is safe. reset() clears the queue. The queue is
// strictly bounded (see voxtral_stream_test_set_max_events for the default). A
// mandatory event that does not fit is NOT dropped: the stream raises explicit
// backpressure (feed -> queue_full) so the caller drains and retries; partials
// coalesce to the newest revision. The bounded finish() tail is always delivered.
//
// poll_event is part of the externally-serialized contract: do not call it
// concurrently with (or reentrantly from) another stream method.
// ----------------------------------------------------------------------------
bool   voxtral_stream_poll_event   (voxtral_stream * stream, voxtral_stream_event & out);
size_t voxtral_stream_pending_events(const voxtral_stream * stream);

const char * voxtral_stream_state_name (voxtral_stream_state state);
const char * voxtral_stream_status_name(voxtral_status status);
const char * voxtral_stream_event_name (voxtral_stream_event_type type);

// ============================================================================
// Test seam (INTERNAL, test-only). None of the following is part of any
// production contract; production code never calls these. They let the model-free
// C++ unit tests exercise paths that otherwise require a real backend:
//   * force / observe context creation without a GGUF file;
//   * observe the transient `finishing` state and the reentrancy guard;
//   * drive the event-queue hard bound with a small limit.
// ============================================================================
using voxtral_stream_context_factory_fn =
    voxtral_context * (*)(voxtral_model *, const voxtral_context_params &);
using voxtral_stream_context_free_fn = void (*)(voxtral_context *);
using voxtral_stream_finishing_hook_fn = void (*)(voxtral_stream *, void * user);

// Override the context factory / free used by create/destroy. nullptr restores
// the default (voxtral_init_from_model / voxtral_free). Global; tests are serial.
void voxtral_stream_test_set_context_factory(voxtral_stream_context_factory_fn factory);
void voxtral_stream_test_set_context_free   (voxtral_stream_context_free_fn free_fn);

// Called once inside finish() right after the stream enters `finishing`, before
// any inference. Lets a test observe the transient state and probe reentrancy.
void voxtral_stream_test_set_finishing_hook(
    voxtral_stream * stream, voxtral_stream_finishing_hook_fn hook, void * user);

// Shrink the event-queue hard bound for overflow testing.
void voxtral_stream_test_set_max_events(voxtral_stream * stream, size_t max_events);
// True iff a push into the (full) event queue has been dropped since creation/reset.
bool voxtral_stream_test_events_overflowed(const voxtral_stream * stream);

#endif // VOXTRAL_STREAM_H

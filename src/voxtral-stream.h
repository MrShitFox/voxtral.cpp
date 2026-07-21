#ifndef VOXTRAL_STREAM_H
#define VOXTRAL_STREAM_H

// ============================================================================
// Internal streaming session skeleton (v1)
// ----------------------------------------------------------------------------
// This header is INTERNAL and UNSTABLE. It is intentionally not part of the
// public C++ surface in include/voxtral.h. The public streaming C ABI, the
// WebSocket server and the incremental Mel/encoder/decoder frontends are all
// deliberately out of scope for this session (see
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
//   * a TEMPORARY compatibility execution path: at finish() the accumulated
//     audio is transcribed once through the existing batch inference
//     (`voxtral_transcribe_audio`). No token is produced before finish().
//
// This is NOT real-time streaming inference. Audio is fully buffered and the
// model runs exactly once, at finish().
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
//       synchronous compatibility inference. Because the API is externally
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
// incremental decoder and are never emitted by the v1 compatibility path.
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
    int32_t                   token  = 0;      // valid for token events (unused in v1)
    std::string               text;            // owned copy
    double                    t_audio_ms = 0.0;// audio position, derived from 64-bit count
    int32_t                   error_code = 0;  // voxtral_status value for error events
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
    // Bounded PCM accumulation. This is a TEMPORARY v1 compatibility limit, not
    // a final public contract: it keeps the full-buffer finish safely below the
    // decoder KV window so it never reaches the (Vulkan-unsafe) kv_cache_shift_left
    // path. 9.6M samples = 10 minutes @ 16 kHz. Exceeding it returns
    // voxtral_status::limit_exceeded (NOT out_of_memory).
    uint64_t max_total_samples = 9'600'000ull;
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

// Run the compatibility batch inference once over the accumulated audio and emit
// FINAL_TEXT + COMPLETED. Idempotent after completion (never re-runs inference).
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
// Introspection (read-only; for tests and the temporary compatibility harness).
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

// Canonical accumulated PCM: float32 mono in nominal [-1, 1]. Valid until the
// next feed/reset/destroy. Used by tests to verify chunk invariance and the
// PCM16 conversion.
const float *        voxtral_stream_pcm_data         (const voxtral_stream * stream);
size_t               voxtral_stream_pcm_size         (const voxtral_stream * stream);

// ----------------------------------------------------------------------------
// Events. poll copies and removes the front event, returning false when empty.
// Repeated polling after drain is safe. reset() clears the queue. The queue is
// strictly bounded (see voxtral_stream_test_set_max_events for the default); a
// push into a full queue is dropped and recorded (last_error + overflow flag),
// never grown past the bound and never silently lost.
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

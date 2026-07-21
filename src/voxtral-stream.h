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
//   * incremental PCM16 / float32 feeding with 64-bit sample accounting;
//   * a bounded internal event queue (lifecycle + error events);
//   * finish/reset/cancel/destroy;
//   * a TEMPORARY compatibility execution path: at finish() the accumulated
//     audio is transcribed once through the existing batch inference
//     (`voxtral_transcribe_audio`). No token is produced before finish().
//
// This is NOT real-time streaming inference. Audio is fully buffered and the
// model runs exactly once, at finish().
// ============================================================================

#include "voxtral.h"   // public API: voxtral_context, voxtral_transcribe_audio, constants

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
    out_of_memory,
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
//   finishing   | err          | ok (idempotent)   | cancelled   | created  | frees
//   completed   | err          | ok (idempotent)   | err         | created  | frees
//   cancelled   | err          | ok (no inference) | ok (no-op)  | created  | frees
//   failed      | err          | err               | ok (no-op)  | created  | frees
//
// (*) A feed carrying >=1 sample moves created -> running. A zero-length feed
//     is a successful no-op and does NOT change state or audio position.
//
// `finishing` is a transient state that is only observable if finish() is
// re-entered; the synchronous compatibility path passes through it while the
// single batch inference runs. There are no hidden transitions.
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
    // Bounded PCM accumulation. Kept safely below the decoder KV window so the
    // compatibility finish never reaches the (Vulkan-unsafe) kv_cache_shift_left
    // path. 9.6M samples = 10 minutes @ 16 kHz. Temporary v1 limit.
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
// `ctx` is BORROWED, not owned. It carries the shared model + device and is the
// execution engine used by the compatibility finish. Ownership contract for v1:
//
//     voxtral_model  outlives  voxtral_context  outlives  voxtral_stream
//
// The stream owns only its own mutable state (PCM, counters, events, transcript)
// and never frees the context or the model. `ctx` may be nullptr for pure
// lifecycle/PCM tests that never run inference; finish() on such a stream only
// succeeds for empty audio (there is nothing to execute otherwise).
//
// Returns nullptr if the params are not supported (see voxtral_stream_params_check).
// ----------------------------------------------------------------------------
voxtral_stream * voxtral_stream_create_internal(
    voxtral_context             * ctx,
    const voxtral_stream_params & params);

// Incremental feeds. The input buffer is copied into the stream; the caller may
// free or reuse it immediately after return. `samples == nullptr` is only valid
// when `sample_count == 0`.
voxtral_status voxtral_stream_feed_pcm16_internal(
    voxtral_stream * stream, const int16_t * samples, size_t sample_count);

voxtral_status voxtral_stream_feed_f32_internal(
    voxtral_stream * stream, const float * samples, size_t sample_count);

// Run the compatibility batch inference once over the accumulated audio and emit
// FINAL_TEXT + COMPLETED. Idempotent after completion (never re-runs inference).
voxtral_status voxtral_stream_finish_internal(voxtral_stream * stream);

// Return the stream to a state equivalent to a freshly created stream with the
// same params and context (clears PCM, counters, transcript, tokens, events,
// error and cancellation). Never touches model weights, tokenizer or device.
voxtral_status voxtral_stream_reset_internal(voxtral_stream * stream);

// Pre-execution cancellation. A later finish() will not run inference. v1 only
// supports cancelling before finish() completes; in-flight GGML graph
// cancellation is not implemented.
voxtral_status voxtral_stream_cancel_internal(voxtral_stream * stream);

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

// Canonical accumulated PCM: float32 mono in nominal [-1, 1]. Valid until the
// next feed/reset/destroy. Used by tests to verify chunk invariance and the
// PCM16 conversion.
const float *        voxtral_stream_pcm_data         (const voxtral_stream * stream);
size_t               voxtral_stream_pcm_size         (const voxtral_stream * stream);

// ----------------------------------------------------------------------------
// Events. poll copies and removes the front event, returning false when empty.
// Repeated polling after drain is safe. reset() clears the queue.
// ----------------------------------------------------------------------------
bool   voxtral_stream_poll_event   (voxtral_stream * stream, voxtral_stream_event & out);
size_t voxtral_stream_pending_events(const voxtral_stream * stream);

const char * voxtral_stream_state_name (voxtral_stream_state state);
const char * voxtral_stream_status_name(voxtral_status status);
const char * voxtral_stream_event_name (voxtral_stream_event_type type);

#endif // VOXTRAL_STREAM_H

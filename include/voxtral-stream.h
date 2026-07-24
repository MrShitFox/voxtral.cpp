#ifndef VOXTRAL_PUBLIC_STREAM_H
#define VOXTRAL_PUBLIC_STREAM_H

/**
 * @file voxtral-stream.h
 * Stable polling-based C ABI for in-process realtime transcription.
 */

#include "voxtral.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VOXTRAL_EVENT_TEXT_CAPACITY 4096u
#define VOXTRAL_EVENT_QUEUE_CAPACITY_DEFAULT 4096u
#define VOXTRAL_EVENT_QUEUE_CAPACITY_MAX 4096u
#define VOXTRAL_EVENT_TERMINAL_HEADROOM 64u
/** Representational ceiling of the public cumulative sample counters. */
#define VOXTRAL_STREAM_MAX_AUDIO_SAMPLES UINT64_MAX

/** Session 11 accepts exactly signed little-endian PCM16 samples. */
typedef enum voxtral_audio_format {
    VOXTRAL_AUDIO_PCM_S16LE = 1
} voxtral_audio_format;

/** Versioned audio description. Unknown trailing fields are ignored. */
typedef struct voxtral_audio_config {
    uint32_t struct_size;
    uint32_t api_version;
    uint32_t sample_rate;
    uint32_t channels;
    voxtral_audio_format format;
    uint32_t reserved;
} voxtral_audio_config;

/**
 * Versioned stream parameters.
 *
 * event_queue_capacity bounds ordinary streaming output. A fixed
 * VOXTRAL_EVENT_TERMINAL_HEADROOM is reserved for the model's bounded finish
 * tail and terminal markers, so the absolute queue bound is capacity plus that
 * constant. When ordinary mandatory output cannot fit, feed reports
 * VOXTRAL_STATUS_QUEUE_FULL and the caller must poll.
 * Valid capacities are 1 through VOXTRAL_EVENT_QUEUE_CAPACITY_MAX.
 * flags and reserved must be zero in API v1.
 */
typedef struct voxtral_stream_params {
    uint32_t struct_size;
    uint32_t api_version;
    voxtral_audio_config audio;
    uint32_t event_queue_capacity;
    uint32_t flags;
    uint64_t reserved;
} voxtral_stream_params;

/** Public stream state. Values are append-only within API major version 1. */
typedef enum voxtral_stream_state {
    VOXTRAL_STREAM_CREATED = 0,
    VOXTRAL_STREAM_ACTIVE = 1,
    VOXTRAL_STREAM_FINISHING = 2,
    VOXTRAL_STREAM_COMPLETED = 3,
    VOXTRAL_STREAM_CANCELLED = 4,
    VOXTRAL_STREAM_ERROR = 5
} voxtral_stream_state;

/** Pollable event kind. Values are append-only within API major version 1. */
typedef enum voxtral_event_type {
    VOXTRAL_EVENT_NONE = 0,
    VOXTRAL_EVENT_TOKEN = 1,
    VOXTRAL_EVENT_PARTIAL_TEXT = 2,
    VOXTRAL_EVENT_FINAL_TEXT = 3,
    VOXTRAL_EVENT_ERROR = 4,
    VOXTRAL_EVENT_COMPLETED = 5
} voxtral_event_type;

/** Bit flags carried by voxtral_event.flags. */
typedef enum voxtral_event_flags {
    VOXTRAL_EVENT_FLAG_NONE = 0,
    VOXTRAL_EVENT_FLAG_SPECIAL_TOKEN = 1u << 0,
    VOXTRAL_EVENT_FLAG_TEXT_STABLE = 1u << 1,
    /**
     * The stable cumulative prefix exceeded the fixed event buffer. text holds
     * its largest UTF-8-safe leading prefix; get_final_text returns the complete
     * final transcript after completion.
     */
    VOXTRAL_EVENT_FLAG_TEXT_TRUNCATED = 1u << 2
} voxtral_event_flags;

/**
 * Caller-owned value event. It contains no borrowed pointer and crosses no
 * allocator boundary.
 *
 * text is UTF-8 and null-terminated. text_length excludes the terminator.
 * TOKEN text is the committed token piece and may be empty for special tokens.
 * PARTIAL_TEXT is a cumulative stable transcript prefix. FINAL_TEXT is the
 * complete transcript when it fits; use get_final_text for arbitrary length.
 */
typedef struct voxtral_event {
    uint32_t struct_size;
    uint32_t api_version;
    voxtral_event_type type;
    uint32_t reserved0;
    uint64_t sequence;
    int32_t token_id;
    uint32_t flags;
    uint64_t audio_start_ms;
    uint64_t audio_end_ms;
    voxtral_status status;
    uint32_t reserved1;
    size_t text_length;
    char text[VOXTRAL_EVENT_TEXT_CAPACITY];
} voxtral_event;

/** Cheap, caller-owned snapshot of stream-specific production telemetry. */
typedef struct voxtral_stream_metrics {
    uint32_t struct_size;
    uint32_t api_version;
    uint64_t audio_samples_accepted;
    uint64_t audio_duration_ms;
    uint64_t encoder_frames;
    uint64_t adapter_groups;
    uint64_t decoder_steps;
    uint64_t token_events;
    uint64_t partial_events;
    uint64_t decoder_kv_wraps;
    uint64_t decoder_kv_evictions;
    uint64_t decoder_kv_bytes_moved;
    double pipeline_rtf;
    double backlog_final_ms;
    double backlog_slope_ms_per_s;
} voxtral_stream_metrics;

/** Stable capabilities of a context in API v1. */
typedef struct voxtral_capabilities {
    uint32_t struct_size;
    uint32_t api_version;
    uint32_t sample_rate;
    uint32_t channels;
    voxtral_audio_format audio_format;
    uint32_t supports_incremental;
    uint32_t supports_reset;
    uint32_t max_active_streams_per_context;
} voxtral_capabilities;

/** Returns the only audio configuration supported by API v1. */
VOXTRAL_API voxtral_audio_config voxtral_audio_default_config(void);

/** Returns fully initialized v1 stream parameters. */
VOXTRAL_API voxtral_stream_params voxtral_stream_default_params(void);

/**
 * Creates one transcription stream that borrows a reusable context.
 *
 * @param context     Valid caller-owned context.
 * @param params      Valid versioned parameters; must not be NULL.
 * @param out_stream  Receives a new handle; set to NULL before validation.
 *
 * @return VOXTRAL_STATUS_OK on success.
 * @return VOXTRAL_STATUS_INVALID_STATE when the context already has a live
 *         stream handle.
 * @return VOXTRAL_STATUS_UNSUPPORTED_AUDIO_FORMAT unless audio is mono signed
 *         PCM16 at 16000 Hz.
 * @return VOXTRAL_STATUS_MODEL_ERROR when context contains an offline model
 *         without realtime incremental support.
 *
 * Ownership: the stream borrows context and never destroys it. Context and its
 * model must outlive the stream. One live stream per context is enforced.
 * Thread safety: externally serialize context and stream operations.
 */
VOXTRAL_API voxtral_status voxtral_stream_create(
    voxtral_context * context,
    const voxtral_stream_params * params,
    voxtral_stream ** out_stream);

/**
 * Destroys a stream from any non-concurrently-used state. Passing NULL is safe.
 *
 * Frees stream-owned host resources and releases the context for a later stream.
 * Pending events are discarded. The borrowed context/model are not destroyed.
 * The function does not wait for or interrupt an in-flight backend graph.
 * Thread safety: no other operation may be running on the stream.
 */
VOXTRAL_API void voxtral_stream_destroy(voxtral_stream * stream);

/**
 * Feeds mono 16 kHz signed PCM16 samples.
 *
 * @param stream            Valid stream in CREATED or ACTIVE.
 * @param samples           Samples, or NULL only when sample_count is zero.
 * @param sample_count      Number of mono samples, not bytes.
 * @param samples_consumed  Optional accepted count; set to zero before
 *                          validation.
 *
 * @return VOXTRAL_STATUS_OK when all samples were accepted.
 * @return VOXTRAL_STATUS_QUEUE_FULL when output backpressure requires polling.
 *         samples_consumed precisely states whether this call accepted audio;
 *         no sample is silently dropped.
 * @return VOXTRAL_STATUS_INVALID_ARGUMENT when the request exceeds the fixed
 *         per-call guard or cannot be represented by cumulative runtime
 *         counters.
 * @return VOXTRAL_STATUS_INVALID_STATE after finish/cancel/completion.
 *
 * A successful zero-length feed is a no-op and may also resume a decoder after
 * the caller has drained events. Input is copied before return. The core
 * library does not impose a product-level session-duration policy.
 * Thread safety: all calls on one stream must be externally serialized.
 */
VOXTRAL_API voxtral_status voxtral_stream_feed_pcm16(
    voxtral_stream * stream,
    const int16_t * samples,
    size_t sample_count,
    size_t * samples_consumed);

/**
 * Polls one event without sleeping or blocking.
 *
 * Initialize out_event.struct_size and out_event.api_version before the call.
 * OK consumes exactly one queued event. NOT_READY consumes nothing.
 * Event sequence values strictly increase across successful polls.
 * API v1 requires storage through the complete fixed text array; future minor
 * versions may append trailing fields without rejecting this v1 size.
 *
 * Thread safety: all calls on one stream must be externally serialized.
 */
VOXTRAL_API voxtral_status voxtral_stream_poll_event(
    voxtral_stream * stream,
    voxtral_event * out_event);

/**
 * Signals end of input and synchronously executes the bounded finish tail.
 *
 * The first successful call emits exactly one FINAL_TEXT followed by exactly
 * one COMPLETED. Repeated calls after completion return OK and emit nothing.
 * No event is emitted after COMPLETED. Backend/GPU work may occur before this
 * function returns; queued events are then drained with poll_event().
 * If prior accepted output remains backpressured, the call returns QUEUE_FULL
 * before FINISHING; poll, issue a zero-length feed, and retry finish.
 *
 * Thread safety: all calls on one stream must be externally serialized.
 */
VOXTRAL_API voxtral_status voxtral_stream_finish(voxtral_stream * stream);

/**
 * Cancels before finish. The operation is idempotent.
 *
 * Audio is no longer accepted, no FINAL_TEXT is emitted, and exactly one ERROR
 * event with status CANCELLED is queued using terminal headroom. This API does
 * not interrupt an in-flight backend graph. Thread safety: externally
 * serialized.
 */
VOXTRAL_API voxtral_status voxtral_stream_cancel(voxtral_stream * stream);

/**
 * Restores the same stream handle to a pristine CREATED state.
 *
 * Valid from CREATED, COMPLETED, CANCELLED, and recoverable ERROR. It clears
 * PCM/Mel/encoder/adapter/decoder state, KV, token history, transcript, queued
 * events, sequence IDs, errors, latency samples and stream metrics. It does not
 * reload the model or recreate permanent backend pipelines unnecessarily.
 * Thread safety: externally serialized.
 */
VOXTRAL_API voxtral_status voxtral_stream_reset(voxtral_stream * stream);

/**
 * Returns a snapshot of the state, or VOXTRAL_STREAM_ERROR for NULL.
 *
 * Thread safety: externally serialize with mutating operations.
 */
VOXTRAL_API voxtral_stream_state voxtral_stream_get_state(
    const voxtral_stream * stream);

/**
 * Copies the stream's most recent structured error.
 *
 * The returned function status describes the copy operation; out_error.status
 * contains the stored operation status. No returned pointer requires free().
 * Thread safety: externally serialize with stream operations.
 */
VOXTRAL_API voxtral_status voxtral_stream_get_last_error(
    const voxtral_stream * stream,
    voxtral_error_info * out_error);

/**
 * Copies the complete final UTF-8 transcript after successful completion.
 *
 * required includes the terminating null byte. buffer==NULL with capacity==0
 * is a valid size query. BUFFER_TOO_SMALL writes no bytes beyond capacity.
 * Returns NOT_READY before successful completion and INVALID_STATE after
 * cancellation/error. Thread safety: externally serialized.
 */
VOXTRAL_API voxtral_status voxtral_stream_get_final_text(
    const voxtral_stream * stream,
    char * buffer,
    size_t capacity,
    size_t * required);

/**
 * Copies a cheap stream-metrics snapshot.
 *
 * No GPU synchronization, D2H transfer, allocation ownership or caller free is
 * introduced. Metrics remain valid after completion and are cleared by reset.
 * pipeline_rtf is zero when detailed context profiling was not enabled.
 * Thread safety: externally serialize with stream operations.
 */
VOXTRAL_API voxtral_status voxtral_stream_get_metrics(
    const voxtral_stream * stream,
    voxtral_stream_metrics * out_metrics);

/**
 * Copies stable v1 capabilities for a context.
 *
 * No inference or GPU synchronization occurs. API v1 reports exactly one active
 * stream per context. Thread safety: externally serialize with context use.
 */
VOXTRAL_API voxtral_status voxtral_context_get_capabilities(
    const voxtral_context * context,
    voxtral_capabilities * out_capabilities);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* VOXTRAL_PUBLIC_STREAM_H */

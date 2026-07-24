# Public streaming C API

## Overview

The public API is an in-process, polling-based foundation for native
applications and a future `voxtral-server`. It loads immutable model weights,
creates a reusable inference context, and drives one PCM transcription
lifecycle through an opaque `voxtral_stream`.

Applications include only:

```c
#include <voxtral.h>
#include <voxtral-stream.h>
```

No private layout, GGML type, C++ standard-library type, callback, network
dependency, or library-allocated text buffer crosses the ABI. The headers are
compiled by the test matrix as C11, C17, C++17, and C++20.
Filesystem paths and all diagnostic/transcription text at the boundary are
UTF-8.

API v1 deliberately does not provide file decoding, resampling, microphone
capture, VAD, networking, language forcing, callbacks, or concurrent streams
on one context.

The streaming factory requires a realtime Voxtral model. Context capabilities
report `supports_incremental == 0` for an offline model, and stream creation
then returns `MODEL_ERROR` rather than pretending incremental operation is
available.

## Ownership

The ownership chain is explicit:

```text
caller-owned model
    └── caller-owned context (borrows model)
            └── caller-owned stream handle (borrows context)
```

- Create and destroy every handle with the matching library function.
- A model must outlive all contexts created from it.
- A context must outlive its stream.
- A stream never destroys its borrowed context or model.
- `destroy(NULL)` is safe for all three handle types.
- Do not call `free()` on a handle, event text, status string, or error text.
- Destroying a stream discards pending events and releases the context lease.
- Destroying a context while a stream is alive violates the documented
  precondition. The implementation refuses to free that context.

API v1 enforces one live stream handle per context. Reuse one stream with
`voxtral_stream_reset()`, or destroy it before creating the next stream.
Different contexts may be operated on different threads.

## Lifecycle

`voxtral_stream_finish()` is synchronous: it may submit and wait for bounded
backend/GPU work before returning. Event polling itself never sleeps.

```text
CREATED ──feed──> ACTIVE ──finish──> FINISHING ──return──> COMPLETED
   │                │
   │                └──cancel────────────────────────────> CANCELLED
   ├──finish─────────────────────────────────────────────> COMPLETED
   └──cancel─────────────────────────────────────────────> CANCELLED

CREATED / COMPLETED / CANCELLED / recoverable ERROR
   └──reset──────────────────────────────────────────────> CREATED
```

The externally observable transitions are:

| Current state | Operation | Result |
|---|---|---|
| `CREATED` | non-empty feed | `ACTIVE` |
| `CREATED` | zero feed | `CREATED`, `OK` |
| `CREATED` or `ACTIVE` | finish | synchronous terminal work, then `COMPLETED` |
| `CREATED` or `ACTIVE` | cancel | `CANCELLED` |
| `COMPLETED` | finish again | `OK`, no duplicate events |
| `CANCELLED` | cancel again | `OK`, no duplicate events |
| terminal state | feed | `INVALID_STATE`, zero samples consumed |
| reusable state | reset | pristine `CREATED` |
| any state with no concurrent call | destroy | resources released |

A successful finish produces exactly one `FINAL_TEXT`, then exactly one
`COMPLETED`. No event follows `COMPLETED`. Cancellation produces no
`FINAL_TEXT`; API v1 exposes one `ERROR` event whose status is `CANCELLED`.

## Audio format

The only accepted format in API v1 is:

```text
format       VOXTRAL_AUDIO_PCM_S16LE
sample rate  16000 Hz
channels     1
sample type  signed 16-bit little-endian
```

`sample_count` is a count of mono samples, not bytes. The library performs no
implicit resampling, byte swapping, or channel mixing. Any other declared
format returns `VOXTRAL_STATUS_UNSUPPORTED_AUDIO_FORMAT`.

The caller retains ownership of its input array. Samples accepted by
`voxtral_stream_feed_pcm16()` are copied before the call returns.

The core streaming library does not impose a product-level session-duration
limit. It rejects only requests that cannot be represented by its runtime
counters or violate the bounded per-call input guard. Applications and network
servers must enforce their own duration, memory, and idle-time policies.

`VOXTRAL_STREAM_MAX_AUDIO_SAMPLES` remains source-compatible and denotes the
`UINT64_MAX` representational ceiling of the cumulative sample counters, not a
recommended session length. Transcript and token history can grow with stream
duration, so the absence of a library duration limit does not guarantee bounded
host memory indefinitely. A future server must enforce its own configurable
session limit. Reset releases stream-specific accumulated state as described
by the existing lifecycle contract.

## Backpressure

`event_queue_capacity` is bounded to
`1..VOXTRAL_EVENT_QUEUE_CAPACITY_MAX` (4096 in API v1). The default is
`VOXTRAL_EVENT_QUEUE_CAPACITY_DEFAULT`. It bounds ordinary streaming events.
API v1 reserves exactly `VOXTRAL_EVENT_TERMINAL_HEADROOM` (64) additional
slots for the fixed right-padding decoder tail plus `FINAL_TEXT`, `COMPLETED`,
or cancellation/error termination. Therefore the absolute host queue bound is
`event_queue_capacity + 64`; it never grows with utterance duration.

Every feed reports the exact accepted prefix through `samples_consumed`.
The function sets that output to zero before validation. If mandatory output
cannot fit, it returns `VOXTRAL_STATUS_QUEUE_FULL`. The caller must:

1. advance by `samples_consumed`;
2. poll events until `NOT_READY`;
3. retry only the unconsumed suffix.

A queue-full return may still report that the entire input slice was accepted;
poll and issue a zero-length feed to resume pending decoder work. The API never
returns `OK` for an unaccepted non-empty suffix, and mandatory events are not
silently dropped.

`TOKEN`, `FINAL_TEXT`, `ERROR`, and `COMPLETED` are mandatory. Cumulative
`PARTIAL_TEXT` snapshots are replaceable: under pressure an older partial may
be coalesced into a newer cumulative prefix without losing committed tokens or
the final transcript.

If `finish()` is called while previously accepted decoder output is still
backpressured, it returns `QUEUE_FULL` without entering `FINISHING`. Drain and
resume with a zero-length feed, then call `finish()` again. Only the fixed
terminal tail uses the reserved headroom.

## Events

`voxtral_event` is a caller-owned value of about 4 KiB. The internal queue does
not store this public representation; the adapter copies into it only during a
successful poll. There is no borrowed text pointer and no required free.

Before every poll initialize:

```c
voxtral_event event = {0};
event.struct_size = sizeof(event);
event.api_version = VOXTRAL_API_VERSION;
```

Event semantics are unambiguous:

| Event | Text meaning |
|---|---|
| `TOKEN` | Raw committed token piece; may be empty for a special token |
| `PARTIAL_TEXT` | Full currently stable cumulative transcript prefix |
| `FINAL_TEXT` | Complete final transcript when it fits in the event |
| `ERROR` | Diagnostic terminal/cancellation text and structured status |
| `COMPLETED` | Successful terminal marker |

`token_id` is `-1` outside `TOKEN`. `audio_end_ms` is the greatest real-audio
position eligible for that token/partial, or total accepted duration for a
terminal event. Token/partial `audio_start_ms` is the preceding 80 ms cadence
boundary clamped to zero; terminal events use zero as their start. These are
audio-timeline positions, not wall-clock timestamps.

All copied text is valid UTF-8, null-terminated, and `text_length` excludes the
terminator. A boundary never splits a UTF-8 code point. If cumulative or final
text exceeds 4095 bytes, the event contains a UTF-8-safe prefix and sets
`VOXTRAL_EVENT_FLAG_TEXT_TRUNCATED`. After completion,
`voxtral_stream_get_final_text()` provides the full transcript with the usual
size-query/copy pattern; `required` includes the null terminator.

Within one reset generation:

- `sequence` strictly increases for every successful poll;
- a `TOKEN` precedes the `PARTIAL_TEXT` it caused;
- partial text is monotonic;
- `FINAL_TEXT` occurs at most once;
- `COMPLETED` occurs exactly once after successful finish;
- no event occurs after `COMPLETED`.

Reset clears the queue and restarts event sequence numbering.

## Error handling

Zero is always success. Public status values are append-only within major
version 1:

| Status | Meaning |
|---|---|
| `OK` | Operation succeeded |
| `INVALID_ARGUMENT` | Invalid pointer, structure metadata, option, or count |
| `INVALID_STATE` | Operation is not valid in the current lifecycle |
| `OUT_OF_MEMORY` | Host allocation failed |
| `QUEUE_FULL` | Poll events, then retry any unconsumed audio |
| `NOT_READY` | No event or final text is currently available |
| `CANCELLED` | Cancellation status |
| `UNSUPPORTED_AUDIO_FORMAT` | Format is not PCM16LE mono 16 kHz |
| `MODEL_ERROR` | Model load or validation failed |
| `BACKEND_ERROR` | Inference/backend operation failed |
| `INTERNAL_ERROR` | Contained implementation failure |
| `BUFFER_TOO_SMALL` | Caller text buffer cannot hold the result |

The API does not use `errno` as its error contract and never lets a C++
exception cross the C boundary. `voxtral_status_string()` returns immutable
static diagnostic text.

Use `voxtral_stream_get_last_error()` for an existing stream. If stream
creation fails before a handle exists, use
`voxtral_context_get_last_error()`. Both copy status, backend code, and UTF-8
message into caller-owned `voxtral_error_info`; no internal mutable pointer is
returned. `backend_code == 0` means that no backend-specific numeric code was
available; the `status` field remains authoritative.

## Reset and reuse

Reset clears all stream-specific mutable state:

- rolling PCM and Mel frontend state;
- incremental encoder and adapter state;
- decoder KV, token history, and transcript;
- pending events and sequence IDs;
- error/cancellation state;
- latency/backlog samples and public metrics.

It retains the loaded model, context backend allocations, and reusable
pipelines. It does not reload weights or change inference mathematics. One
stream handle can therefore serve sequential utterances cheaply.

## Metrics and capabilities

`voxtral_stream_get_metrics()` returns a cheap caller-owned snapshot during or
after a stream. Reset clears it. The query performs no hidden GPU
synchronization or diagnostic D2H readback. `pipeline_rtf` is zero unless
detailed runtime profiling was enabled before context creation.

The snapshot contains accepted audio, encoder/adapter/decoder work counts,
event counts, decoder-ring wrap/eviction/move counters, pipeline RTF, and final
backlog metrics. It intentionally omits internal reservoirs and profiling
objects.

`voxtral_context_get_capabilities()` reports the fixed audio format,
incremental/reset support, and, for a realtime context,
`max_active_streams_per_context == 1`. An unsupported offline context reports
zero for all three streaming capability fields.

## Thread safety

All operations on one stream must be externally serialized. A context is not
safe for concurrent inference, and API v1 does not claim producer/poller
concurrency on the same stream. The library does not add a mutex around every
call.

Different streams on different contexts may run on different threads. Model
weights are immutable after loading, but callers must still serialize model
destruction with all related context operations.

## Minimal C usage

The complete buildable example is
[`examples/c/streaming.c`](../examples/c/streaming.c). Its core loop feeds
80 ms (1280-sample) slices and handles partial consumption:

```c
size_t consumed = 0;
voxtral_status status = voxtral_stream_feed_pcm16(
    stream, samples + offset, remaining, &consumed);
offset += consumed;

if (status == VOXTRAL_STATUS_QUEUE_FULL) {
    /*
     * Poll until NOT_READY, then zero-feed until it returns OK before
     * retrying only the remaining suffix.
     */
}
```

The example loads a PCM16 mono 16 kHz WAV only to demonstrate feeding. WAV
parsing is not part of the streaming API.

## P/Invoke notes

P/Invoke uses `CallingConvention.Cdecl`, `IntPtr` for opaque handles, `UIntPtr`
for C `size_t`, and 32-bit values for the supported C compiler enum ABI. The
managed process and native library must have matching bitness.

Representative declarations:

```csharp
internal enum VoxtralStatus : int
{
    Ok = 0,
    InvalidArgument = 1,
    InvalidState = 2,
    OutOfMemory = 3,
    QueueFull = 4,
    NotReady = 5,
    Cancelled = 6,
    UnsupportedAudioFormat = 7,
    ModelError = 8,
    BackendError = 9,
    InternalError = 10,
    BufferTooSmall = 11,
}

[StructLayout(LayoutKind.Sequential)]
internal struct VoxtralAudioConfig
{
    internal uint StructSize;
    internal uint ApiVersion;
    internal uint SampleRate;
    internal uint Channels;
    internal int Format;
    internal uint Reserved;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VoxtralStreamParams
{
    internal uint StructSize;
    internal uint ApiVersion;
    internal VoxtralAudioConfig Audio;
    internal uint EventQueueCapacity;
    internal uint Flags;
    internal ulong Reserved;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VoxtralEvent
{
    internal uint StructSize;
    internal uint ApiVersion;
    internal int Type;
    internal uint Reserved0;
    internal ulong Sequence;
    internal int TokenId;
    internal uint Flags;
    internal ulong AudioStartMs;
    internal ulong AudioEndMs;
    internal VoxtralStatus Status;
    internal uint Reserved1;
    internal UIntPtr TextLength;
    internal fixed byte Text[4096];
}

internal static class Native
{
    private const string Library = "voxtral";

    [DllImport(Library, CallingConvention = CallingConvention.Cdecl,
        ExactSpelling = true)]
    internal static extern VoxtralStatus voxtral_stream_create(
        IntPtr context,
        in VoxtralStreamParams parameters,
        out IntPtr stream);

    [DllImport(Library, CallingConvention = CallingConvention.Cdecl,
        ExactSpelling = true)]
    internal static extern VoxtralStatus voxtral_stream_feed_pcm16(
        IntPtr stream,
        short[] samples,
        UIntPtr sampleCount,
        out UIntPtr samplesConsumed);

    [DllImport(Library, CallingConvention = CallingConvention.Cdecl,
        ExactSpelling = true)]
    internal static extern VoxtralStatus voxtral_stream_poll_event(
        IntPtr stream,
        ref VoxtralEvent output);

    [DllImport(Library, CallingConvention = CallingConvention.Cdecl,
        ExactSpelling = true)]
    internal static extern VoxtralStatus voxtral_stream_finish(IntPtr stream);

    [DllImport(Library, CallingConvention = CallingConvention.Cdecl,
        ExactSpelling = true)]
    internal static extern VoxtralStatus voxtral_stream_reset(IntPtr stream);

    [DllImport(Library, CallingConvention = CallingConvention.Cdecl,
        ExactSpelling = true)]
    internal static extern void voxtral_stream_destroy(IntPtr stream);
}
```

Initialize every versioned structure using the native default initializer or
set both header fields exactly. On a 64-bit process the current v1 layouts are:
audio config 24 bytes, stream params 48 bytes, error info 272 bytes, event
4160 bytes, stream metrics 112 bytes, and capabilities 32 bytes. Bindings
should assert these sizes during their own smoke tests rather than hard-code
native allocation.

The fixed event value, copied error value, caller-owned final buffer, and
library-owned handle factories avoid allocator crossing and dangling managed
pointers.

## ABI and versioning policy

`VOXTRAL_API_VERSION` is packed as `major << 24 | minor << 16 | patch`.
`voxtral_api_version()` and `voxtral_version_string()` expose the runtime
version.

- Patch releases fix bugs without breaking ABI.
- Minor releases add fields, enum values, or functions while preserving
  existing ABI.
- Major releases may make breaking API/ABI changes.

All extensible structures begin with `struct_size` and `api_version`.
The library reads only fields present in a known prefix, supplies defaults for
missing current fields, and ignores unknown trailing data. An impossible size
or incompatible major version is rejected. Callers must zero reserved fields.
The embedded `voxtral_audio_config` v1 prefix is frozen at 24 bytes for major
version 1; future stream options append to `voxtral_stream_params` rather than
shifting the following fields.

Public functions use C linkage and unmangled names. Shared builds use hidden
visibility plus an explicit export allowlist; internal runtime helpers are not
part of the dynamic symbol table. Linux publishes SONAME `libvoxtral.so.1`.

ABI review for v1:

| Question | Result |
|---|---|
| Are all public handles opaque? | Yes |
| Are all exported functions C linkage? | Yes |
| Can public headers compile as C? | Yes, C11 and C17 are tested |
| Do extensible structs carry size/version? | Yes |
| Are status/event constants append-only? | Yes within major v1 |
| Are ownership and valid lifetimes explicit? | Yes |
| Can a caller incorrectly free returned library text? | No text allocation crosses the ABI |
| Can an event/error pointer dangle? | No; both are caller-owned values |
| Are exceptions contained? | Yes, every throwing adapter path is caught |
| Are internal helpers hidden? | Yes, enforced by the symbol allowlist test |
| Can C# marshal structures deterministically? | Yes, with matching bitness and verified sizes |
| Is one stream per context enforced? | Yes, creation returns `INVALID_STATE` |

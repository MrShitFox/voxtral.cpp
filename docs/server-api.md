# Standalone server API

`voxtral-server` is a small HTTP/WebSocket adapter around the public
`libvoxtral` C API. One process owns one model, one Vulkan context, and at most
one active inference stream. It does not create jobs or queue requests.

## Build

The server network layer uses Boost.Asio and Boost.Beast. JSON parsing and
serialization use Boost.JSON. Boost 1.75 or newer with the `system` and `json`
components is required. CMake enables `VOXTRAL_BUILD_SERVER` by default when
those components are found; an explicit `-DVOXTRAL_BUILD_SERVER=ON` reports a
configure error when they are missing.

Build the server and the versioned shared public library:

```bash
cmake -S . -B build-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DGGML_VULKAN=ON \
  -DVOXTRAL_BUILD_SERVER=ON
cmake --build build-release -j --target voxtral-server
cmake --install build-release --prefix /usr/local
```

On Nix, a temporary development environment can be created without vendoring
dependencies:

```bash
nix-shell -p cmake ninja boost
```

The installed executable is `bin/voxtral-server`. It includes only
`<voxtral.h>` and `<voxtral-stream.h>` for inference and links to
`libvoxtral.so.1`; it does not use runtime-private headers or helpers.

## Startup and configuration

Minimal authenticated startup:

```bash
printf '%s\n' 'replace-with-a-long-random-token' > /run/voxtral-api-key
chmod 600 /run/voxtral-api-key

voxtral-server \
  --model /models/Voxtral-Mini-4B-Realtime-2602-Q4_K_M.gguf \
  --listen 127.0.0.1 \
  --port 8080 \
  --api-key-file /run/voxtral-api-key
```

CLI settings override environment settings, which override defaults.

| CLI | Environment | Default |
|---|---|---|
| `--model PATH` | `VOXTRAL_SERVER_MODEL` | required |
| `--listen ADDRESS` | `VOXTRAL_SERVER_LISTEN` | `127.0.0.1` |
| `--port PORT` | `VOXTRAL_SERVER_PORT` | `8080` |
| `--api-key-file PATH` | `VOXTRAL_SERVER_API_KEY_FILE` | none |
| — | `VOXTRAL_SERVER_API_KEY` | none |
| `--max-upload-mib N` | `VOXTRAL_SERVER_MAX_UPLOAD_MIB` | `512` |
| `--realtime-soft-lag-ms N` | `VOXTRAL_SERVER_REALTIME_SOFT_LAG_MS` | `1000` |
| `--realtime-hard-lag-ms N` | `VOXTRAL_SERVER_REALTIME_HARD_LAG_MS` | `5000` |
| `--realtime-buffer-ms N` | `VOXTRAL_SERVER_REALTIME_BUFFER_MS` | `1000` |
| `--idle-timeout-sec N` | `VOXTRAL_SERVER_IDLE_TIMEOUT_SEC` | `60` |

`--help` prints the complete CLI and `--version` prints the server, library,
and C API versions.

Authentication must be configured unless `--no-auth` is explicit.
Unauthenticated operation is accepted automatically only for a loopback bind.
Using `--no-auth` on any other bind is a startup error unless
`--allow-insecure-no-auth` is also present; that override emits a warning.

At startup the process loads the model with the Vulkan backend and creates the
context before accepting connections. A model/context failure is a structured
startup error and a non-zero exit. `SIGINT` and `SIGTERM` stop the acceptor,
cancel the active stream at a public-API boundary, close its connection, join
the worker, and destroy stream, context, then model. Shutdown has a 10-second
network drain deadline; an in-flight Vulkan graph is never interrupted
unsafely.

## Authentication

All transcription endpoints require:

```http
Authorization: Bearer <token>
```

The scheme is case-sensitive, the token must be non-empty, and comparison with
the configured key covers both length and bytes without an early byte mismatch
exit. Missing and invalid credentials return `401 unauthorized`.

`GET /health` is intentionally unauthenticated. Health never exposes the model
filesystem path. Neither health responses nor normal server logs expose the API
key, Authorization header, request body, PCM, or complete transcript.

## Health

```http
GET /health
```

Example response:

```json
{
  "status": "ok",
  "ready": true,
  "busy": false,
  "busy_mode": null,
  "server_version": "1.0.0",
  "voxtral_version": "1.0.0",
  "voxtral_api_version": 16777216,
  "model": "voxtral-mini",
  "capabilities": {
    "sample_rate": 16000,
    "channels": 1,
    "audio_format": "pcm_s16le",
    "max_active_streams": 1
  }
}
```

While inference is active, `busy` is true and `busy_mode` is `batch` or
`realtime`. Health performs no inference, GPU synchronization, or diagnostic
readback.

## Synchronous batch transcription

```http
POST /v1/audio/transcriptions
```

The request remains open until inference completes. The server creates no job
ID, persistence record, or background result.

PCM WAV input:

```bash
curl -sS \
  -H "Authorization: Bearer $VOXTRAL_API_KEY" \
  -H "Content-Type: audio/wav" \
  --data-binary @speech.wav \
  "http://127.0.0.1:8080/v1/audio/transcriptions?response_format=json"
```

`audio/wav` and `audio/x-wav` accept RIFF/WAVE with unknown chunks and normal
odd-chunk padding, but the actual format must be uncompressed integer PCM,
signed 16-bit little-endian, mono, 16000 Hz.

Raw PCM input:

```bash
curl -sS \
  -H "Authorization: Bearer $VOXTRAL_API_KEY" \
  -H "Content-Type: application/octet-stream" \
  -H "X-Audio-Format: pcm_s16le" \
  -H "X-Sample-Rate: 16000" \
  -H "X-Audio-Channels: 1" \
  --data-binary @speech.pcm \
  http://127.0.0.1:8080/v1/audio/transcriptions
```

Raw PCM bodies must have a non-zero even byte count. The default response is:

```json
{
  "id": "tr_...",
  "object": "audio.transcription",
  "model": "voxtral-mini",
  "text": "Full transcript...",
  "duration_ms": 3578,
  "processing_ms": 2521.0,
  "realtime_factor": 0.7046
}
```

`?response_format=text` returns only UTF-8 transcript text with
`Content-Type: text/plain; charset=utf-8`. Any other response format is a
`400`. MP3, M4A, AAC, FLAC, OGG, multipart bodies, resampling, and channel
mixing are not supported and return `415`.

The declared and received body are both bounded by `--max-upload-mib`. WAV
chunk arithmetic, PCM alignment, and size conversions are checked before
acquiring the GPU lease. Batch audio is then fed in bounded chunks with exact
partial-consumption and `QUEUE_FULL` handling. If the client disconnects, the
worker calls `voxtral_stream_cancel` at the next safe public-API boundary,
destroys the stream, and releases the lease without retaining a result.

## Realtime WebSocket protocol v1

Connect to:

```text
ws://127.0.0.1:8080/v1/realtime/transcription
```

The Bearer header and free GPU lease are checked before the HTTP upgrade.
The first application message must arrive within five seconds and be text
JSON:

```json
{
  "type": "session.configure",
  "audio": {
    "format": "pcm_s16le",
    "sample_rate": 16000,
    "channels": 1
  },
  "events": {
    "token": false,
    "partial": true
  }
}
```

`language` may be absent or `"auto"`. After validation the server creates the
public stream and sends `session.created`. Audio is sent as WebSocket binary
messages containing raw signed PCM16LE mono 16 kHz—never WAV, Base64, or a JSON
envelope. A binary message may contain at most 64 KiB and must have an even byte
count. An empty binary message is a documented no-op.

Client text messages:

| Type | Effect |
|---|---|
| `input_audio.end` | Finish input, emit final/completed, and close normally |
| `session.cancel` | Cancel without a final transcript, emit `session.cancelled`, and close normally |
| `ping` with string `id` | Emit application-level `pong` with the same `id` |

Unsupported messages and invalid JSON are fatal protocol errors. Beast also
handles WebSocket protocol ping/pong frames.

Server text events:

| Event | Meaning |
|---|---|
| `session.created` | Configuration accepted; contains session ID and audio contract |
| `transcript.token` | Token piece; emitted only when `events.token` is true |
| `transcript.partial` | Latest cumulative stable prefix |
| `transcript.final` | Complete final transcript |
| `session.completed` | Duration, realtime factor, final backlog, and backlog slope |
| `session.warning` | Rate-limited `processing_lag` warning |
| `session.cancelled` | Explicit client cancellation acknowledged |
| `error` | Safe code/message and `fatal` flag |
| `pong` | Application ping response |

Transcript event sequence values come from the public stream and are strictly
increasing even when token events were not requested.

## Busy behavior and backpressure

There is one process-global lease with `FREE`, `BUSY_BATCH`, `BUSY_REALTIME`,
and `SHUTTING_DOWN` states. It never waits and there is no server request
queue. A busy batch request or WebSocket upgrade receives immediately:

```http
HTTP/1.1 503 Service Unavailable
Retry-After: 1
Content-Type: application/json
```

```json
{
  "error": {
    "code": "server_busy",
    "message": "The transcription engine is currently busy."
  }
}
```

The realtime PCM queue holds `--realtime-buffer-ms` of audio (32,000 bytes per
second). A single already-read frame is separately bounded by 64 KiB; when the
PCM queue is full the session stops starting reads and lets TCP/WebSocket
backpressure slow the client. It never drops an accepted frame.

The outbound queue is bounded to 256 messages and 4 MiB. Pending cumulative
partials are coalesced to the newest value. Requested token events and
mandatory final/error/completed events apply backpressure instead of being
silently dropped.

Lag combines received-but-not-accepted audio with the public stream backlog.
The soft threshold emits at most one warning per second and resets below
75 percent of the threshold. Reaching the hard threshold cancels the stream
with `realtime_capacity_exceeded`; a stream cannot silently turn into an
unbounded delayed batch.

Idle means no client text, binary, ping, or pong frames. PCM silence still
counts as activity. An idle session emits `idle_timeout`, closes, and releases
the lease.

## Limits and errors

The server enforces:

- 32 KiB HTTP headers;
- the configured HTTP body maximum;
- 16 KiB JSON control messages;
- 64 KiB WebSocket binary messages;
- five-second WebSocket configuration/handshake timeouts;
- the configured client-frame idle timeout;
- bounded PCM and outbound event queues.

HTTP errors use `{"error":{"code":"...","message":"..."}}`. WebSocket errors
use `{"type":"error","code":"...","message":"...","fatal":true}`. Stable codes
include `unauthorized`, `server_busy`, `invalid_json`,
`invalid_configuration`, `unsupported_audio_format`, `invalid_audio_frame`,
`input_overflow`, `realtime_capacity_exceeded`, `idle_timeout`,
`shutting_down`, `out_of_memory`, `model_error`, `backend_error`, and
`internal_error`. Raw C++ exception text is logged locally but never returned
to a client.

## C# examples

[`examples/csharp/VoxtralServerClient.cs`](../examples/csharp/VoxtralServerClient.cs)
uses only `HttpClient`, `ClientWebSocket`, `System.Text.Json`, and
`CancellationToken`. The accompanying SDK project has no package references.

```bash
dotnet run --project examples/csharp/VoxtralServerExample.csproj -- \
  batch http://127.0.0.1:8080/v1/audio/transcriptions "$VOXTRAL_API_KEY" speech.wav

dotnet run --project examples/csharp/VoxtralServerExample.csproj -- \
  realtime ws://127.0.0.1:8080/v1/realtime/transcription \
  "$VOXTRAL_API_KEY" speech.pcm
```

The realtime example reads a raw `.pcm` file and paces it in 80 ms chunks:

```text
CHUNK_SAMPLES = 1280
CHUNK_BYTES   = 2560
```

It deliberately does not capture a microphone.

## TLS and limitations

`voxtral-server` itself serves HTTP and WebSocket without TLS. For public
Internet deployment place it behind Caddy, nginx, or HAProxy.

Version 1 intentionally has no compressed audio decoding, multipart form
parsing, jobs, persistence, server queue, multi-stream/continuous batching,
VAD, diarization, translation, Hy-MT2, resampling, channel mixing, browser UI,
CORS policy, Prometheus endpoint, or embedded deployment configuration. One
RX 6600 workload is active at a time.

# Node.js acceptance harness

This directory is the reproducible acceptance layer for the current batch CLI and the future
incremental runtime. It does not implement or simulate streaming inference. The base
`StreamingTransport` deliberately throws `Not implemented`; concrete stdio and WebSocket
transports belong to later runtime/API sessions.

## Commands

Run from `tests/node`:

```bash
npm ci
npm run test:unit
npm test
npm run test:environment
npm run test:gpu
npm run acceptance:baseline
```

The first two commands are local and deterministic. `npm test` discovers the whole suite but
remote suites remain skipped unless their explicit scripts set the opt-in environment flags.
`test:environment` checks local tools plus SSH, model presence and `vulkaninfo`.
`test:gpu` ensures the remote Vulkan build is current and runs the short 3.58-second fixture.
`acceptance:baseline` runs environment, unit tests, local build, source-only rsync, remote build
and GPU inference sequentially, then writes a summary artifact.

Individual orchestration commands are also available:

```bash
npm run build:local
npm run sync:gpu
npm run build:gpu
```

Configuration is reloaded per call from `VOXTRAL_LOCAL_REPO`, `VOXTRAL_GPU_HOST`,
`VOXTRAL_GPU_USER`, `VOXTRAL_GPU_PASSWORD`, `VOXTRAL_REMOTE_REPO`, `VOXTRAL_REMOTE_MODEL`,
`VOXTRAL_REMOTE_BUILD` and `VOXTRAL_ARTIFACT_DIR`. Environment values override the checked local
defaults. Diagnostics never serialize the password. SSH uses `SSHPASS`/`sshpass -e`, keeping the
secret out of argv. Source synchronization rejects every destructive destination except the fixed
`/root/voxtral.cpp` tree and excludes Git, builds, dependencies, artifacts, caches and GGUF files.

## Reusable helpers

- `config/environment.js`: validation, derived local/remote paths and redacted descriptions.
- `helpers/exec.js`: shell-free `spawn`, stdout/stderr draining, callbacks, timeout, abort, signal
  and detailed error results.
- `helpers/remote.js`, `helpers/build.js`: guarded SSH/rsync and incremental CMake orchestration.
- `helpers/wav.js`, `helpers/audio.js`: chunk-aware PCM16 RIFF parsing, SHA-256 metadata, optional
  ffmpeg normalization and raw PCM extraction. Streaming input is strictly mono 16 kHz PCM16 LE.
- `helpers/chunks.js`: deterministic sample-based plans for full, 80/160/320/480/1000 ms,
  single-sample, seeded-random and custom sizes, including explicit zero-length feed events.
- `helpers/artifacts.js`: unique run directories with metadata, command, stdout, stderr and result.
- `helpers/inference.js`: structured transcript, backend/device evidence and timing extraction for
  the existing batch CLI. The CLI has useful summaries but no stable machine-readable result mode;
  a JSON/event mode remains desirable for the future streaming CLI.
- `helpers/streaming-transport.js`: intentionally abstract transport contract and timestamped event
  collector with timeout/abort support.

Artifacts are written under `.artifacts/<timestamp>-<uuid>/`, ignored by Git. Failures retain full
stdout/stderr; successful results may remain compact. Every bundle records commit, host, backend,
binary/model/audio paths, exit status, wall time and test name.

## Future streaming acceptance matrix

The rows below are requirements, not skipped placeholder tests. They become executable only after
the corresponding runtime feature exists.

### Chunk invariance

- Full PCM in one feed.
- Fixed 80 ms, 160 ms and 480 ms feeds.
- Reproducible seeded-random boundaries.
- One sample per feed for short fixtures.
- Explicit zero-length feed events.
- Assert identical sample stream first, then Mel/token/transcript parity with batch.

### Lifecycle

- Create and configure.
- Empty finish and repeated finish.
- Feed after finish.
- Cancel, reset, destroy and reconnect.
- Deterministic terminal error/completed ordering.

### Numerical parity

- Incremental vs batch Mel maximum absolute delta.
- Encoder output comparison at equivalent completed-frame boundaries.
- Exact token sequence and transcript parity.
- Batch vs stream parity on CPU and Vulkan.

### Runtime

- CPU and Vulkan paths.
- Long session crossing cache policy thresholds.
- Bounded memory growth, cache limit and repeated stream cleanup.
- Concurrent streams and multiple users without mutable-model races.

### Network

- Binary PCM and documented JSON fallback.
- Malformed JSON and oversized frames.
- Slow client, fast producer and bounded backpressure.
- Disconnect during upload/inference and reconnect semantics.

The 31-second synthetic fixture is intentionally not in the default baseline. Long-session,
concurrency and WebSocket cases remain opt-in until their runtime surfaces are implemented.

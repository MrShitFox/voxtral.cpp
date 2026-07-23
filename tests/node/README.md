# Node.js acceptance harness

This directory is the reproducible acceptance layer for the batch CLI and the production
incremental Mel / encoder-KV / adapter / decoder runtime. The scheduler harness drives one native
process per case; that process owns PCM pacing and no SSH round-trip occurs per chunk. It is a test
harness, not a public Node.js or network API.

## Commands

Run from `tests/node`:

```bash
npm ci
npm run test:unit
npm test
npm run test:environment
npm run test:gpu
npm run acceptance:baseline
npm run benchmark:encoder-scheduler
npm run acceptance:encoder-realtime
npm run test:encoder-kv-manual
npm run acceptance:decoder-kv-ring
npm run acceptance:kv-fp16
npm run benchmark:realtime-pipeline
npm run acceptance:realtime-sustained
npm run test:realtime-rollover
npm run test:realtime-soak-30m
```

The first two commands are local and deterministic. `npm test` discovers the whole suite but
remote suites remain skipped unless their explicit scripts set the opt-in environment flags.
`test:environment` checks local tools plus SSH, model presence and `vulkaninfo`.
`test:gpu` ensures the remote Vulkan build is current and runs the short 3.58-second fixture.
`acceptance:baseline` runs environment, unit tests, local build, source-only rsync, remote build
and GPU inference sequentially, then writes a summary artifact.

`benchmark:encoder-scheduler` performs the staged logical/physical matrix
(1/2/4/8/16/32/64/128 against every 32/64/128 shape it fits), a deterministic synthetic soak
and the optional two-minute spoken fixture. Set
`VOXTRAL_LONG_AUDIO` to the private M4A to enable the spoken stage. The fixture is transferred
to `/root/voxtral-test-data` separately, normalized once, and never enters source rsync or Git.
The helper compares local, transferred and normalized source SHA-256 values before any run.
`acceptance:encoder-realtime` gates the selected low-query default against 128/128 on short plans and,
when the fixture is available, 80/160/480 ms plus seeded-random paced long-form input. Every
long plan decodes the complete token sequence/transcript, and the compute run also executes the
independent global-batch encoder tensor oracle. `test:encoder-kv-manual` is an opt-in fused-flash
versus explicit-manual-attention oracle with full token/text and pure-mode runtime comparison.

The Session 8 commands are strict. `acceptance:decoder-kv-ring` compares the reusable production
ring, a dynamic physical ring and a logical-order attention oracle while a reduced test-only
capacity wraps inside the spoken fixture. `acceptance:kv-fp16` runs sequential FP16/F32 plans and
requires exact token/transcript parity. `benchmark:realtime-pipeline` records the F32, decoder-only
FP16 and dual-FP16 shape matrix. `acceptance:realtime-sustained` always runs a full 30-minute paced
80 ms stream; `VOXTRAL_SOAK_SECONDS` may lengthen it but cannot shorten it. Its rollover and soak
aliases use the same non-relaxed hard gates: RTF below 0.95, zero final backlog, non-positive slope,
zero full-buffer KV movement, FP16 memory limits, bounded graph allocation, zero dropped events and
the published latency limits.

For optional long-form regression tests:

```bash
VOXTRAL_LONG_AUDIO=/home/glebus/Desktop/Code/cppShit/voxtral.cpp/voxTest2min.m4a \
npm run test:encoder-realtime:long
npm run test:encoder-realtime:soak
```

Set `VOXTRAL_ENCODER_TRACE_JSONL=/absolute/path/trace.jsonl` in a native harness environment to
capture compact per-frame dependency and residence timestamps. Ordinary JSON remains aggregate
and does not contain a large timestamp array.

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
- `helpers/artifacts.js`: unique run directories with metadata, command, stdout, stderr and result;
  scheduler runs can add a redacted CSV summary, transcript and approximate token-boundary map.
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

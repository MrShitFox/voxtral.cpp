# Session 9 — RX 6600 hardware-ceiling analysis and production hardening

Branch `perf/session9-hardware-ceiling`, base `45e0605` (= `origin/main`).
All measurements use the RX 6600 Vulkan/RADV host and the production incremental
pipeline: **variant C — encoder KV F32 + decoder KV FP16, encoder shape 4/4.**
No precision/shape overrides unless a row explicitly names one.

## Verdict

**The current production pipeline is close to the practical RX 6600 ceiling
under the existing GGML/Vulkan architecture.** It is GPU-compute-bound at
~93 % GPU utilisation with the core clock already at the card's real sustained
ceiling. No material single-stream software bottleneck remains: submission,
synchronization, graph reuse, allocations and device↔host traffic are already
at their optimum, and neither a forced high-performance DPM state, a larger
encoder batch shape, nor reduced synchronization produces a material gain.

This is **Outcome B**. No performance code change is shipped (an artificial
perf commit would be churn); the benchmark tooling, telemetry and this evidence
are the deliverable, plus the production-lifecycle / sequential-reuse hardening.

## Hardware / runtime

| Item | Value |
|---|---|
| GPU | AMD Radeon RX 6600 (Navi 23, PCI `1002:73ff`), RADV |
| VRAM | 8.0 GB total; visible BAR only 256 MB (small-BAR host) |
| Core clock (sclk) DPM | states {sleep, 500, 2750} MHz; **real sustained ≈ 2530–2567 MHz** |
| Memory clock (mclk) | max 875 MHz; pinned at max under load |
| Driver | Mesa 26.1.5, RADV, Vulkan 1.4.354 |
| Kernel | 7.1.4-arch1-1 (Arch Linux) |
| CPU | Intel Xeon E5-2690 v2 @ 3.0 GHz, 10 cores, governor `schedutil` |
| RAM | 16 GB |
| Power / thermal | cap 100 W; under load ~75 W, 57–60 °C (no power or thermal throttling) |
| Tooling | vulkaninfo, strace, ffmpeg present; **perf / amdgpu_top / radeontop absent** → GPU telemetry sampled from amdgpu sysfs |

`power_dpm_force_performance_level` was `auto` at start; the ceiling sweep
temporarily forced `high` and **restored `auto`**. No system configuration was
committed or left changed.

## Baseline stage profile (2-min fixture, unpaced, warm, `auto`)

1531 tokens (exact match to the Session 8 production reference). pipeline RTF
**0.7165**, wall/audio **0.7166**.

| stage | count | total ms | mean | p50 | p95 | p99 | % of GPU |
|---|---:|---:|---:|---:|---:|---:|---:|
| encoder_graph_execute | 1530 | 48811 | 31.90 | 31.87 | 32.13 | 32.36 | **58.0** |
| decoder_step_graph_execute | 1531 | 34983 | 22.85 | 22.85 | 23.04 | 23.13 | **41.6** |
| mel_compute | 1522 | 2407 | 1.58 | 1.66 | 1.94 | 2.05 | 2.9 |
| adapter_graph_execute | 1538 | 327 | 0.21 | 0.22 | 0.24 | 0.26 | 0.4 |
| decoder_prefill_graph_execute | 1 | 53 | 52.8 | — | — | — | 0.06 |
| token_readback | 1531 | 255 | 0.17 | 0.11 | 0.35 | 0.36 | — |
| backend_synchronize (subset of execute) | 4611 | 52449 | 11.37 | 5.84 | 28.95 | 29.29 | — |

`pipeline_feed` mean 56.75 ms ≈ Σ per-feed GPU stages (enc 31.9 + dec 22.85 +
mel 1.58 + adapter 0.21 ≈ 56.5) ⇒ **non-GPU per-feed overhead ≈ 0.2 ms**.

Structure (all already optimal):
- **submit : sync = 4611 : 4611 = 1.0** — one full backend sync per graph.
- steady graph builds enc/adapter/decoder = **2 / 0 / 1**; steady allocations
  **2 / 0 / 1** — no per-step build or allocation.
- device→host = **token IDs only** (`tensorGetCount` = 1531, `adapterInputD2hBytes` = 0).
- `temporaryF32KvBytes` = 0; decoder KV FP16 (element size 2), 872 MB; peak VRAM 4.319 GB.

## CPU vs GPU diagnosis: **GPU-compute-bound**

| Evidence | Measurement |
|---|---|
| GPU busy (unpaced, 2-min / 4-min) | **92.9 % / 93.9 %** (idle only ~7 %) |
| Core clock under load | sclk ~2530 MHz = **96–99 % of the card's real sustained max** |
| Memory clock under load | mclk ~862/875 MHz (**effectively pinned at max**) |
| Process CPU (4-min unpaced) | **0.49 cores** of 10 → not CPU-bound |
| Memory-controller busy | ~20 % → single-token decode is latency/occupancy-bound, not BW-saturated |
| Non-GPU per-feed overhead | ~0.2 ms → no dispatch/submission slack to reclaim |

The two dominant stages are irreducible model work: the encoder (32 layers,
window 750) attends 4 new frames over the full causal window every call; the
decoder (26 layers, dim 3072, vocab 131072) streams ~2.25 GB of Q4 weights per
token. The 1:1 submit:sync costs only ~7 % idle because each 23–32 ms graph
dwarfs submission overhead — so removing synchronization has a <7 % ceiling and
the decoder sync is fundamentally required (autoregressive: step N+1 needs
step N's argmax).

## Paced vs unpaced (2-min)

| Mode | pipeline RTF | wall RTF | GPU busy | sclk MHz | encoder ms | CPU cores |
|---|---:|---:|---:|---:|---:|---:|
| unpaced (GPU never idle) | 0.7165 | 0.7166 | 92.9 % | 2533 | 31.90 | 0.49 |
| paced 80 ms (production) | 0.7682 | 1.0076 | 70.2 % | 2042 | 35.63 | 0.38 |

The paced/unpaced gap is fully explained by **DPM downclocking during pacing
idle gaps** (sclk 2533→2042, so the encoder runs slower, 31.9→35.6 ms), not by
serialization. This is why production paced RTF (~0.77–0.79) exceeds the unpaced
compute ratio (~0.71). It does not threaten realtime: wall/audio 1.0076 with
backlog draining to 0.

## Hardware-ceiling sweep (`auto` vs forced `high`, 2-min unpaced, 3 repeats)

| Power state | pipeline RTF (median warm) | sclk MHz | GPU busy | power | temp |
|---|---:|---:|---:|---:|---:|
| auto (default) | 0.7170 | 2528 | 93 % | 75 W | 60 °C |
| high (forced)  | 0.7170 | 2528 | 93 % | 76 W | 60 °C |

**high vs auto = 0.01 % (noise).** Forcing high does not help: even in `high`,
`pp_dpm_sclk` selects the 2600 MHz state (not the 2750 peak) and the actual
hwmon clock is ~2567 MHz — the RDNA2 power management will not sustain 2750 MHz
under this envelope, and `auto` already reaches ~2530 MHz (within ~1.5 %). There
is **no DPM headroom** to reclaim. Not thermally or power throttled.

## Encoder batch-shape experiment (2-min unpaced)

| Encoder physical rows | wall RTF | note |
|---|---:|---|
| 4 (production) | 0.7127 | zero padding; segmented attention reads the ring in place |
| 8 | 0.8528 | ~20 % slower — pads 4 real query rows to 8 |
| 16 | 0.9366 | slower still (monotonic padding penalty) |

The streaming cadence feeds exactly 4 encoder frames (one adapter group) per
drain for feed-plan invariance and low latency; larger physical rows only add
padding, and because segmented attention does not materialise the window,
batching more query frames amortises nothing. **4/4 is optimal.**

## Optimization candidates considered

| Candidate | Expected | Measured | Decision |
|---|---|---|---|
| Force high-performance DPM | up to ~8 % (short-clip hypothesis) | 0.01 % | Reject — no sustained-clock headroom; also a host setting, not shippable |
| Reduce inter-stage synchronization | ≤7 % (GPU idle ceiling) | not pursued | Reject — GPU already 93 % busy, decoder sync is fundamental, high correctness risk |
| Larger encoder batch (8/8, 16/16) | amortise per-call cost | 8 → 20 % slower | Reject — padding overhead; also raises residence latency |
| Graph/command-buffer reuse | fewer builds | already 2/0/1 steady | Nothing to do |
| Remove allocations / D2H | fewer copies | already token-ID-only, temp F32 KV = 0 | Nothing to do |

No candidate meets the ≥5 % material-gain bar; nothing is shipped as a perf change.

## Production result (unchanged production variant C)

| Fixture | pipeline RTF | tokens | peak VRAM |
|---|---:|---:|---:|
| 2-min unpaced | 0.7165 | 1531 (exact) | 4.319 GB |
| 4-min unpaced | 0.7126 | 3072 (exact) | 4.319 GB |
| 2-min paced 80 ms | 0.7682 | 1531 (exact) | 4.319 GB |
| 4-min paced 80 ms | 0.7657 | 3072 (exact) | 4.319 GB |

## Production lifecycle hardening

Added `test:sequential-stream-reuse` (>=100 short streams through one reused
context via `reset()`) and `acceptance:production-lifecycle`. Both drive a new
`--sequential-streams N` mode in the stream test binary.

100 short streams reused through one context via `reset()`: **byte-identical 49
tokens every stream** (no stale KV / token history / event sequence), a clean
`reset → created` transition each time, and a flat tail — **VRAM range 0 B, RSS
range 0 KiB** over the last 20. All seven state-machine edges hold: finish is
idempotent (`ok/ok`), feed-after-finish is rejected (`invalid_state`),
cancel → `cancelled`, finish-after-cancel → `ok`, reset-after-cancel →
`created`, destroy-under-backpressure does not crash/deadlock, reset-from-created
is idempotent. The reference (non-incremental) decode path still completes.

## Next meaningful gain

Single-stream software is at the RX 6600 ceiling. Further gains require:
- **custom fused Vulkan kernels** for the decoder single-token matvec / encoder
  windowed attention (beyond stock GGML) — highest risk/effort;
- **Q8 (or lower) decoder KV** or a different model quantization — trades quality;
- a **faster GPU** (more bandwidth for the decoder, more compute for the encoder);
- **multi-stream / continuous batching** — raises *aggregate* throughput on the
  same GPU, not single-stream latency.

## Regression

**No production runtime source (`src/`) was changed** — the production library is
byte-identical to `45e0605`, so production RTF / token parity / memory behaviour
is unchanged by construction. All changes are test-harness (`tests/node` helpers
+ scripts) and a new, default-off `--sequential-streams` mode in the stream test
binary; the existing test-binary modes and the `monitorMemory` harness path are
byte-for-byte unchanged.

| Check | Result |
|---|---|
| Local RelWithDebInfo build (incl. new C++ mode) | PASS |
| Local Release build (incl. new C++ mode) | PASS |
| Remote RX 6600 / Vulkan build | PASS |
| CTest RelWithDebInfo / Release | 5/5 / 5/5 PASS |
| Node unit (`npm run test:unit`) | 45/45 PASS |
| `test:sequential-stream-reuse` (100 streams) | PASS — identical 49 tokens/stream, all 7 edges pass, VRAM tail range 0 B, RSS tail range 0 KiB |
| `acceptance:production-lifecycle` | PASS — all 11 checks (7 state-machine edges + reuse consistency + reset pristine + reference mode completes) |
| `benchmark:pipeline-profile` | PASS — classifier returns compute-bound for 2-min (0.7156, 92.7 %) and 4-min (0.7128, 94 %) |
| `benchmark:incremental-unpaced` | PASS — unpaced 2min 0.716/4min 0.713 (busy 93/94 %); paced 2min 0.769/4min 0.766 (busy 71 %) |
| `benchmark:hardware-ceiling` | PASS (power restored to `auto`) |
| `acceptance:end-to-end-realtime` (unchanged prod path spot-check) | PASS |

The Session 8 production matrix (30-min soak, precision matrix, rollover, RSS
plateau, real-fixture repeats) was **not** re-run: it validates the production
runtime, which is unchanged here; those results stand from Session 8 on the
identical `45e0605` tree.

## Changed files

| File | Purpose |
|---|---|
| `tests/node/helpers/stream.js` | additive `gpuTelemetry` amdgpu sysfs sampler (busy%/clocks/power/temp) + process-CPU capture + `--sequential-streams` plumbing; existing paths unchanged |
| `tests/node/helpers/gpu-power.js` | new — record/set/**restore** `power_dpm_force_performance_level` (allowlisted levels only) |
| `tests/node/helpers/profile-report.js` | new — stage-table aggregation + rule-based bottleneck classifier |
| `tests/node/helpers/benchmark-run.js` | new — shared profiling run + repeat statistics (median/CV) |
| `tests/node/scripts/benchmark-hardware-ceiling.js` | new — auto-vs-high DPM sweep with restore |
| `tests/node/scripts/benchmark-pipeline-profile.js` | new — per-stage profile of production-C |
| `tests/node/scripts/benchmark-incremental-unpaced.js` | new — unpaced max-throughput + paced contrast |
| `tests/node/scripts/test-sequential-stream-reuse.js` | new — 100 reused streams, plateau + edges |
| `tests/node/scripts/acceptance-production-lifecycle.js` | new — state-machine edge cases + reference mode |
| `tests/node/package.json` | new npm scripts |
| `tests/cpp/voxtral_stream_test.cpp` | new default-off `--sequential-streams N` lifecycle/reuse mode |
| `docs/session9-findings.md` | this report |

## Git

- Branch `perf/session9-hardware-ceiling`, base `45e0605` (= `origin/main`).
- One commit of test/benchmark tooling + docs; **no `src/` change**, so the
  production runtime is byte-identical to `45e0605`.
- Pushed to `origin/main` (fast-forward, no force). Upstream untouched.
- Working tree clean; private fixtures never staged.

## Fixture safety

`voxTest2min.m4a` and `voxTest4min.m4a` were used only as local test fixtures.
Neither file nor any generated WAV/PCM derivative was tracked or committed.

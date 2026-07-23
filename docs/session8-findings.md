# Session 8.1 — FP16 quality, bounded memory and sustained realtime

Branch `wip/session8-sustained-realtime`, WIP base `0f72eae` on parent
`68d002f`. Measurements use the RX 6600 Vulkan/RADV server and the internal
streaming production path at encoder logical/physical shape 4/4.

## Production verdict

The production default is **variant C: encoder KV F32 + decoder KV FP16**.

- It is token- and transcript-exact against F32/F32 on both real fixtures:
  0/4,603 token edits, WER 0, CER 0.
- The only observed FP16 drift follows encoder FP16: variants B and D have the
  same local four-token proper-name spelling change on the two-minute fixture;
  decoder FP16 alone does not change either recording.
- A corrected 30-minute paced run has pipeline RTF **0.767796**, final backlog
  0, slope `-0.00000569815 ms/s`, zero dropped events and zero decoder KV moves.
- The decoder KV uses 872,415,232 bytes; encoder KV uses 460,324,864 bytes;
  peak VRAM in the 30-minute run is 4,319,039,488 bytes.
- An accelerated production-path rollover test captured 48 decoder wraps and
  shows a flat 20-wrap tail for RSS and VRAM.

Dual FP16 (D) also passes the fixture quality gates and is faster/smaller, but
it is rejected as the default because encoder FP16 is the isolated source of
the only measured text drift. F32/F32 (A) is exact but uses substantially more
VRAM. Encoder FP16 + decoder F32 (B) retains the same drift as D while giving up
the dominant decoder-KV memory saving.

The two local fixtures show no material quality regression, but they are not a
complete multilingual quality corpus.

## Fixtures

| Fixture | Duration | SHA-256 | Language/content | Plans | Git |
|---|---:|---|---|---|---|
| `voxTest2min.m4a` | 121.600 s | `d66181fd86a94d04156cb0d1e1aacaae3d2f97e131af0e38a6fcf29c04ab45a4` | Russian monologue about a composition and the Oppenheimer quotation | full cold/warm, paced 80/160/480 ms, seeded random | ignored locally, not tracked |
| `voxTest4min.m4a` | 244.821 s | `e662476be717c53a50bee67a03c0463b5712447c29178952d95285dc887328c3` | Russian biographical monologue about Albert Wesker | full cold/warm, paced 80/160/480 ms, seeded random | ignored locally, not tracked |

The sources and normalized WAV/PCM derivatives live outside the remote Git
repository. Full transcripts, token IDs and transcript diffs are test artifacts;
audio is never copied into an artifact bundle.

## Precision isolation matrix

The aggregate corpus contains 4,603 oracle tokens, 626 reference words and
4,425 reference characters.

| Variant | Encoder KV | Decoder KV | Token edits | WER | CER | Max pipeline RTF | Peak VRAM | Result |
|---|---|---|---:|---:|---:|---:|---:|---|
| A | F32 | F32 | 0/4,603 | 0 | 0 | 0.850668 | 5,191,294,976 | PASS, oracle |
| B | FP16 | F32 | 4/4,603 (0.0869%) | 0.1597% | 0.0452% | 0.791996 | 4,937,932,800 | PASS, rejected |
| C | F32 | FP16 | **0/4,603** | **0** | **0** | 0.789142 | 4,318,875,648 | **selected** |
| D | FP16 | FP16 | 4/4,603 (0.0869%) | 0.1597% | 0.0452% | **0.728859** | **4,065,517,568** | PASS, rejected |

Every cold/warm repetition and every chunk plan is deterministic. All cases
finish with backlog 0, non-positive slope, zero dropped events and zero
full-buffer decoder moves. F32/F32 4/4 matches the independent F32 4/32 global
batch encoder oracle exactly.

The original immutable matrix artifact was written before the production-rule
review and recorded B under the former “prefer B on any quality pass” rule.
`deriveProductionDecision()` now applies the literal rule: B must be exact or
strictly less divergent than D; otherwise exact C is preferred. All downstream
quality, latency, real-fixture and sustained artifacts explicitly record C.
The measured A–D runs themselves were not changed or replayed.

## Divergence analysis

B and D have the same single region; C has no divergent region.

| Timestamp | Token indices | F32 text | Candidate text | Classification | Meaning changed | Re-converged |
|---:|---|---|---|---|---|---|
| 39.200 s | 483–486 | `Аламагаруда.` | `Аламагорода.` | proper-name spelling | no material semantic change | yes, immediately |

Reference IDs/pieces are `4383/га`, `2895/ру`, `2726/да`, `1046/.`;
candidate IDs/pieces are `22732/гор`, `8107/ода`, `1046/.`, `32/<special>`.
The five preceding tokens are identical and the following stream is identical
from token 487 onward. There are no changed numbers, negations, sentence counts,
commands, deletions, insertions or sustained desynchronization. The four-minute
fixture is exact for A, B, C and D.

## Real-fixture production repeats

| Fixture | Repeat | Tokens | Pipeline RTF | Backlog p95/max/final | Slope | Peak VRAM | Result |
|---|---:|---:|---:|---|---:|---:|---|
| 2 min | 1 | 1,531 | 0.788371 | 0 / 28.936 / 0 ms | -0.00145374 | 4,318,875,648 | PASS |
| 2 min | 2 | 1,531 | 0.788052 | 0 / 28.922 / 0 ms | -0.00157267 | 4,318,875,648 | PASS |
| 4 min | 1 | 3,072 | 0.779953 | 0 / 31.471 / 0 ms | -0.000404098 | 4,325,834,752 | PASS |
| 4 min | 2 | 3,072 | 0.780165 | 0 / 29.794 / 0 ms | -0.000384520 | 4,318,679,040 | PASS |

Within each fixture, token IDs, transcript, encoder SHA-256 and adapter SHA-256
match exactly between repeats and the matrix reference. Encoder output
accumulation and dropped events are zero.

## Sustained realtime

| Duration | Pipeline RTF | Backlog p95/max/final | Slope | Deadline misses | Decoder wraps | Result |
|---:|---:|---|---:|---:|---:|---|
| 30 min synthetic, paced 80 ms | **0.767796** | **0 / 19.565 / 0 ms** | **-0.00000569815 ms/s** | 2/22,500 (0.00889%) | 2 | **PASS** |
| 2 min real, paced 80 ms, repeat 1 | 0.788371 | 0 / 28.936 / 0 ms | -0.00145374 | 2/1,520 | 0 | PASS |
| 4 min real, paced 80 ms, repeat 1 | 0.779953 | 0 / 31.471 / 0 ms | -0.000404098 | 2/3,060 | 0 | PASS |

The 30-minute run has compute headroom 23.22%, wall/audio factor 1.00053,
events dropped 0, decoder bytes moved 0, full-buffer moves 0 and
`encoderOutputAccumulatedBytes=0`. Its two wraps evict 14,357 positions; wrap-step
latency is 22.066 ms versus pre-wrap p99 24.806 ms.

### Harness telemetry defect found during the soak

The first Session 8.1 C soak produced pipeline RTF 0.742696 but a false 312-second
backlog. The harness built a full percentile summary of the profiler reservoirs
before and after every one of 22,500 feeds. This O(n log n) diagnostic work sat
outside the internal `pipeline_feed` metric: wrapper feed mean was 75.734 ms
versus internal 59.380 ms and grew with duration.

Per-feed full profile snapshots are now executed only when
`--capture-rollover-memory` explicitly needs wrap attribution. Normal paced runs
take one steady and one final profile snapshot. A five-minute diagnostic then
measured wrapper/internal means 62.210/62.204 ms, final backlog 0 and negative
slope. The full repeated soak above is the production proof. No production
threshold was relaxed. The external timeout was separately widened only to
cover deterministic synthetic PCM generation, model load and warmup before the
unchanged 1,800-second paced interval.

## RSS and VRAM plateau

The accelerated test uses the production decoder/KV path and a reduced
test-only ring capacity. It captured 48 wraps without trim and 48 wraps in the
post-destroy trim diagnostic. The plateau analysis excludes one-time warmup and
fault-in steps and evaluates the last 20 wraps (29–48).

| Wrap | Absolute position | RSS before/after/settled KiB | Anonymous settled KiB | VRAM bytes | Graph objects/allocs | Decoder allocs | KV moved |
|---:|---:|---|---:|---:|---|---:|---|
| 1 | 64 | 233808 / 233808 / 233808 | 87280 | 4,311,310,336 | 4 / 4 | 2 | 0 |
| 5 | 320 | 233840 / 233840 / 233840 | 87312 | 4,311,310,336 | 4 / 4 | 2 | 0 |
| 10 | 640 | 233904 / 233904 / 233904 | 87376 | 4,311,310,336 | 4 / 4 | 2 | 0 |
| 20 | 1280 | 235956 / 235956 / 235956 | 89428 | 4,311,310,336 | 4 / 4 | 2 | 0 |
| 29 | 1856 | 255788 / 255788 / 255788 | 109260 | 4,311,310,336 | 4 / 4 | 2 | 0 |
| 30 | 1920 | 255792 / 255792 / 255792 | 109264 | 4,311,310,336 | 4 / 4 | 2 | 0 |
| 40 | 2560 | 255924 / 255924 / 255924 | 109396 | 4,311,310,336 | 4 / 4 | 2 | 0 |
| 48 | 3072 | 256048 / 256048 / 256048 | 109520 | 4,311,310,336 | 4 / 4 | 2 | 0 |

For wraps 29–48, RSS slope is 13.844 KiB/wrap with only 260 KiB total
delta/range and is classified non-linear/plateaued; anonymous RSS is identical.
VRAM slope/delta/range are zero. Graph objects, graph allocations and decoder
allocations remain 4/4/2. Decoder rollover bytes and full-buffer moves remain
zero at every wrap.

After stream destruction, RSS is 248,868 KiB. A diagnostic-only
`malloc_trim(0)` reduces it to 184,408 KiB (64,460 KiB allocator retention
released). The trim is never called in the production steady-state path. Child
exit is observed and releases process memory.

## Eligibility latency

Absolute time remains diagnostic because both fixtures contain non-lexical
intro material. Hard gates use runtime overhead after the required audio is
available.

| Fixture | Metric | Absolute | Eligibility | Runtime overhead | Gate | Result |
|---|---|---:|---:|---:|---:|---|
| 2 min | first decoder step | 717.975 ms | 658.436 ms | 59.540 ms | <200 ms | PASS |
| 2 min | first lexical token | 2222.220 ms | 2160.070 ms | 62.151 ms | <250 ms | PASS |
| 2 min | first partial | 2222.220 ms | 2160.070 ms | 62.153 ms | <350 ms | PASS |
| 4 min | first decoder step | 718.608 ms | 659.007 ms | 59.601 ms | <200 ms | PASS |
| 4 min | first lexical token | 2302.870 ms | 2240.100 ms | 62.770 ms | <250 ms | PASS |
| 4 min | first partial | 2302.870 ms | 2240.100 ms | 62.771 ms | <350 ms | PASS |

Lifecycle is reported separately: model load 1.995/2.011 s, context creation
12.433/11.771 ms, Vulkan warmup 408.020/406.291 ms and stream start
80.107/80.097 ms for the two fixtures.

## Memory attribution

- Encoder KV: 460,324,864 bytes, F32, fixed ring.
- Decoder KV: 872,415,232 bytes, FP16, fixed ring.
- Temporary F32 KV: 0.
- 30-minute peak VRAM: 4,319,039,488 bytes; after stream destroy:
  3,842,646,016 bytes; after child exit: 990,298,112 bytes including server
  baseline/other mappings.
- 30-minute peak RSS: 565,548 KiB; after finish: 316,832 KiB; after stream
  destroy: 310,672 KiB; child exit observed.
- PCM peak retained samples: 360. Encoder output accumulated bytes: 0.
- Steady graph builds/allocations: encoder 2/2, adapter 0/0, decoder 2/2.
- Device-to-host traffic in steady inference is token IDs only (90,044 bytes
  over 30 minutes); encoder, adapter and logits D2H bytes are zero.
- The soak harness retains 45,024 drained events solely to write token-window
  evidence; the production event queue high-watermark is bounded and the
  dedicated RSS test disables retained event history.

## Regression status

| Check | Result |
|---|---|
| RelWithDebInfo local + RX 6600/Vulkan builds | PASS |
| Release local build | PASS |
| CTest RelWithDebInfo / Release | 5/5 / 5/5 PASS |
| Node unit / `npm test` | 45/45 PASS; 45 PASS + 12 expected GPU-env skips |
| `acceptance:precision-matrix` | PASS |
| `acceptance:fp16-quality` | PASS |
| `acceptance:real-fixtures` | PASS |
| `test:rss-rollover-plateau` | PASS |
| `acceptance:latency-eligibility` | PASS |
| `test:realtime-soak-30m` | PASS |
| `acceptance:decoder-kv-ring` | PASS |
| `acceptance:kv-fp16` | PASS |
| `acceptance:end-to-end-realtime` | PASS |

The final harness-only fix conditionally removes expensive full-profile
snapshots from ordinary paced feeds. The matrix inference, quality analysis,
real-fixture determinism and RSS capture path were not replayed: the first three
do not depend on profiler snapshot frequency, while the RSS test explicitly
enables `capture_rollover_memory` and therefore executes the unchanged branch.
The corrected full soak and the three targeted component/end-to-end suites were
rerun on the final tree.

## Evidence artifacts

- Precision matrix:
  `tests/node/.artifacts/2026-07-23T17-08-47.981Z-d1940821-a87d-4f65-95f9-d43a1a37fb67`
- FP16 quality:
  `tests/node/.artifacts/2026-07-23T17-09-26.759Z-6c933c48-804c-49f3-acf8-83d31d46093a`
- Eligibility latency:
  `tests/node/.artifacts/2026-07-23T17-09-27.069Z-0c56b137-2bf9-41e3-a7eb-d1cdb3c9a13e`
- RSS plateau:
  `tests/node/.artifacts/2026-07-23T17-26-04.328Z-6eab26ee-0749-4d80-9dd0-2d6c4231c114`
- Real production repeats:
  `tests/node/.artifacts/2026-07-23T17-39-10.505Z-e5ae1428-a001-477b-939f-b7c5b7a2bd7a`
- Corrected 30-minute soak:
  `tests/node/.artifacts/2026-07-23T18-55-04.866Z-24bfed4f-a38e-4faf-bcbe-66926d1d145c`
- Decoder KV ring:
  `tests/node/.artifacts/2026-07-23T18-55-45.212Z-8dc0b812-9c18-4a1e-82a3-bac0aa019de6`
- FP16 KV component:
  `tests/node/.artifacts/2026-07-23T18-56-25.217Z-1e9bbb29-e7bf-4a04-8143-c736839e6256`
- Final end-to-end realtime:
  `tests/node/.artifacts/2026-07-23T19-12-21.232Z-937919e8-75d5-4d07-9a82-db0f57694a2e`

## Fixture safety

`voxTest2min.m4a` and `voxTest4min.m4a` were used only as local test
fixtures. Neither file nor any generated WAV/PCM derivative is tracked or
committed.

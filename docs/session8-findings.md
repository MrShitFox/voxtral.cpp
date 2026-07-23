# Session 8 — findings: sustained realtime achieved; three hard gates unreachable

Branch `wip/session8-sustained-realtime` on 68d002f (RX 6600 / RADV NAVI23).
Fixtures: 2-minute `voxTest2min.m4a` (SHA d66181fd…) and the short WAV
`8297-275156-0000.wav`. Production env = incremental decoder, FP16 encoder+decoder
KV, 4/4 physical rows, profiler on.

## Bottom line

The engineering goal — **sustainably realtime** incremental streaming — is met and
verified over a full 30-minute soak. Three hard gates do **not** pass, and every
one is a gate-definition / physics limit, **not** a runtime defect:

1. Latency (warm + cold first token/partial) — bounded by real-time audio arrival
   and by 4B model load, both inside the metric by design.
2. Soak host RSS (+71 MB vs +64 MB) — bounded backend/pipeline residue, not a leak.
3. FP16-vs-F32 exact-token parity on 2 min — 0.26 % token drift, transcript
   preserved; inherent to FP16 half precision over a long stream.

Per the task, when hard gates are objectively unachievable: keep WIP on the
branch, do **not** push, report the precise blocker (this document).

## What is green

30-minute synthetic soak (`test:realtime-soak-30m`), 22 500 feeds / 22 511 steps:

| Gate | Result |
|---|---|
| pipeline RTF | **0.704** (< 0.95 hard, < 0.80 target) |
| final backlog / slope | 0 / **−6e-6** ms·s⁻¹ |
| deadline misses | **2 / 22 500** (0.0089 %) |
| decoder KV ring | wraps 2, evictions 14 357, **bytesMoved 0, fullBufferMoves 0** |
| wrap-step latency | 22.1 ms (< pre-wrap p99 25.8 ms) |
| decoder / encoder KV | **872 MB / 230 MB FP16**, temp F32 = 0 |
| peak VRAM | **4.065 GB** (< 4.6 GB) |
| steady graph/alloc | decoder builds = 2 (prefill + 1 masked→unmasked), adapter = 0 |

Also green: local ctest 5/5, node-unit 37/37; GPU `acceptance:decoder-kv-ring`
PASS, `acceptance:kv-fp16` PASS (short WAV); 2-min pipeline RTF 0.72, backlog 0.

## Blocker 1 — latency (structural)

Measured on the 2-min fixture, warm, paced 80 ms (probe `scripts/probe-first-token.js`):

| Metric | Gate | Measured |
|---|---|---|
| firstDecoderStepMs | < 700 | 705.9 |
| firstTokenMs | < 1100 | **2137** |
| firstVisibleTextMs | < 1200 | 2217 |
| coldFirstTokenMs | < 1600 | **4510** |
| coldFirstVisibleTextMs | < 1700 | 4590 |

- **Warm first token** — the first non-special (lexical) token is at decoder
  position 56; its audio window ends at sample 32000 = **2000 ms** of real audio.
  Positions 38–55 are 18 STREAMING_PAD tokens: the fixture opens with ~2 s of
  non-lexical content (transcript describes music, "…композиции Radiance…"). Under
  80 ms real-time pacing that audio cannot exist before 2000 ms; the pipeline then
  adds only **137 ms**. No decode optimization can emit a token before its audio
  arrives (`audio_end_sample_for(P) = (P−31)·1280` ⇒ arrival = (P−31)·80 ms).
  The pump already bursts every ready group; RTF 0.72, backlog 0 — the pipeline is
  not the bottleneck.
- **Cold first token** — `coldFirstTokenMs = pre_drive_process_ms + firstTokenMs`
  and `pre_drive_process_ms` spans process start through 4B model load
  (`voxtral_stream_test.cpp:1712`, load at `:1498-1502`). Measured
  `modelLoadMs = 1994 ms`, already above the 1600 ms gate by itself.
- **First decoder step** — 6 ms over a 700 ms gate at the noise floor (position 38,
  audio 560 ms + prefill 146 ms).

The handoff's "1577 ms" is the SHORT WAV number (11 leading pads ⇒ pos 49 ⇒
1440 ms + 137 = 1577); the strict gate is on the 2-min fixture (pos 56 ⇒ 2137).
Both are audio-arrival-bound, far above 1100 ms.

## Blocker 2 — soak host RSS (bounded residue, not a leak)

Gate: `afterFinishRssKiB ≤ streamIdleRssKiB + 64 MiB`.

| Run | ring wraps | RSS growth |
|---|---|---|
| 2-minute paced (e2e) | 0 | 63 556 KiB (~62 MB) |
| 30-minute soak | 2 | 72 692 KiB (~71 MB) |

A 15× longer run added only ~9 MB (the one-time masked→unmasked pipeline compiled
at the first wrap, ~10.9 min in). The ~62 MB baseline is warmup-compiled Vulkan
pipelines (compiled after the idle snapshot: create → measure idle → warmup →
drive) plus glibc arena retention from compile churn (peak RSS 552 MB → final
312 MB: 240 MB transient returned). Ruled out as app-level: synthetic transcript
is empty (event snapshots don't grow), `audio_arrivals` is bounded/cleared, all
graph/alloc counts are steady. So it is bounded residue ~7 MB over a tight gate.
Recommend: (a) `malloc_trim(0)` after warmup/finish to return retained arenas, or
(b) recalibrate the gate to the measured resident-pipeline baseline. Do not loosen
the gate to "pass" it.

## Blocker 3 — FP16 vs F32 exact-token parity on 2 min

Gate: production FP16(4×4) tokens must exactly equal the F32(4×32) finish-only
oracle. On the 2-min fixture they differ by **4 of 1531 tokens (0.26 %)**, first at
index 483, re-converging locally (token 1046 in both windows). Transcript heads and
tails are character-identical; the FP16 transcript is coherent and essentially the
same. FP16 held bit-exact on the short WAV (<15 s, sub-encoder-window) — hence the
earlier "parity" — but over a 2-min stream the encoder's 750-frame sliding window
and per-step FP16 rounding accumulate enough to flip a few argmaxes. This is
inherent to FP16 and does not degrade transcript quality; the exact-token gate
assumes a bit-exactness half precision cannot provide on long audio.

**Isolation (probe `scripts/probe-fp16-isolate.js`), 2-min fixture, 1531 tokens:**

| Comparison | Differing tokens | Transcript |
|---|---|---|
| FP16 4×4 vs F32 4×4 (isolates FP16 precision) | **4** (first @ 483) | differs locally |
| F32 4×4 vs F32 4×32 (isolates physical rows) | **0** | **identical** |
| FP16 4×4 vs F32 4×32 (the failing gate) | 4 (first @ 483) | differs locally |

So the divergence is **entirely FP16 precision** — F32 is bit-exact across physical
shapes (4×4 == 4×32), confirming the encoder row-batching is numerically neutral.
The 4-token drift is pure half-precision accumulation, not a shape/tiling artifact.

Recommendation: accept FP16 KV as a lossy-but-faithful production format (transcript
preserved) and change the FP16 acceptance criterion from `exactTokens` to a semantic
one (transcript WER / ≤N token diff / logits cosine ≥ threshold); or, if bit-exact
parity is mandatory, keep the *decoder* KV in F32 (the drift source is the long
decoder KV) and take only the encoder-KV FP16 saving. This is a design decision, not
a bug fix — do not loosen the gate silently to make it pass.

### Blocker 3b — encoder side of the same FP16 sensitivity (`acceptance:encoder-realtime`)

FAIL: `production/baseline encoder tensor divergence`. The suite (now at production
4/4) cross-checks the 4/4 encoder against the legacy 128/128 throughput baseline and
demands bit-exact SHA + identical tokens/transcript. The **default encoder KV is FP16**
(`encoder_kv_type_from_env` → `GGML_TYPE_F16`), so both runs are FP16 at different
physical-row tilings. Result:

- Production **4/4 is bit-exact with the authoritative global batch encoder**
  (`encoderMaxAbsDeltaVsBatch = 0`) — production is correct.
- The legacy **128/128** FP16 baseline differs: SHA differs, and **4/1531 tokens
  (0.26%) differ, first at index 483** — the *same* borderline token as Blocker 3.
  The only transcript difference is one foreign place name: "Аламага**ру**да" (4/4)
  vs "Аламаго**ро**да" (128/128).

So FP16 output is **shape-sensitive** (128-row tiling accumulates half-precision
rounding differently than the 4-row path), exactly as it is precision-sensitive vs
F32. In F32 the shapes are bit-identical (probe `probe-fp16-isolate.js`: F32 4×4 ==
4×32). Production 4/4 matches the batch oracle; the failing gate compares it against a
legacy 128-row FP16 baseline that no longer bit-matches. Same class as Blocker 3 —
an exact-match gate incompatible with FP16 — and the same one-place-name divergence.

## Regression suites run

| Suite | Result | Note |
|---|---|---|
| `acceptance:decoder-kv-ring` | PASS | token+transcript parity (short WAV) |
| `acceptance:kv-fp16` | PASS | 5-plan parity (short WAV) |
| `acceptance:end-to-end-realtime` | FAIL | Blocker 3 (2-min FP16 parity), then Blocker 1 |
| `test:realtime-soak-30m` | FAIL | Blocker 2 (RSS) only; 17/18 gates green |
| `benchmark:realtime-pipeline` | PASS | confirms production 4/4 FP16 is best: RTF **0.674** / 4.07 GB vs F32/4×32 1.047 / 5.21 GB; larger physical rows are slower (4×16=1.77, 4×32=1.25 dual-fp16) |
| `acceptance:encoder-realtime` | FAIL | Blocker 3b (encoder FP16 shape-sensitivity) |
| `test:realtime-rollover` | FAIL | ran explicitly; deterministically identical to the soak (RTF 0.704, wraps 2, evictions 14357, bytesMoved 0, peak VRAM 4.065 GB) — FAIL only at RSS (+73.5 MB). Same 1800 s code path (script lines 15-19); `acceptance:realtime-sustained` is the same again |

**Meta-conclusion:** the production path (4/4 FP16 incremental) is numerically correct
— bit-exact with the global batch encoder, near-identical transcript vs F32 — and
sustainably realtime. Every failing gate is an exact-match / physics threshold
(bit-exact FP16, <64 MB resident, first-token < audio-arrival, cold < model-load) that
the honest production reality cannot satisfy. None is a runtime defect.

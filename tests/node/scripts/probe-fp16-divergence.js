// DIAGNOSTIC (untracked): quantify the FP16(4x4) vs F32(4x32) divergence on the
// 2-min fixture. Reproduces the exact e2e oracle and diffs it against the FP16
// production tokens/transcript captured in the failing e2e artifact.
import fs from "node:fs";
import path from "node:path";

import { loadEnvironment } from "../config/environment.js";
import { normalizeFixtureOnGpu, syncFixture } from "../helpers/remote.js";
import { runStreamSession } from "../helpers/stream.js";
import { SESSION8_PRODUCTION_ENV } from "../helpers/session8.js";

const config = loadEnvironment();
const artifactDir = process.argv[2];
const fp16 = JSON.parse(fs.readFileSync(path.join(artifactDir, "result.json"), "utf8")).paced;

const localFixture = path.join(config.localRepo, "voxTest2min.m4a");
await syncFixture(localFixture, { config });
const fixture = await normalizeFixtureOnGpu({ config });

// Exact e2e oracle: F32 KV, 4/32 physical, reference decoder, 80ms chunk plan.
const ref = await runStreamSession({
  config,
  planName: "probe-f32-4x32-oracle",
  audioPath: fixture.wavPath,
  mode: "80ms",
  env: {
    ...SESSION8_PRODUCTION_ENV,
    VOXTRAL_STREAM_DECODER: "reference",
    VOXTRAL_ENCODER_KV_TYPE: "f32",
    VOXTRAL_DECODER_KV_TYPE: "f32",
    VOXTRAL_ENC_KV_PHYSICAL_ROWS: "32",
  },
  timeoutMs: 600_000,
});

const a = fp16.tokens, b = ref.tokens;
let firstDiff = -1;
const n = Math.min(a.length, b.length);
for (let i = 0; i < n; i++) if (a[i] !== b[i]) { firstDiff = i; break; }
let nDiff = 0;
for (let i = 0; i < n; i++) if (a[i] !== b[i]) nDiff++;

const out = {
  fp16: { nTokens: a.length, transcriptLen: (fp16.transcript || "").length },
  f32ref: { nTokens: b.length, transcriptLen: (ref.text || "").length },
  transcriptsIdentical: (fp16.transcript || "") === (ref.text || ""),
  tokenCountsEqual: a.length === b.length,
  firstDivergenceIndex: firstDiff,
  differingTokensInCommonPrefix: nDiff,
  windowAtFirstDiff: firstDiff >= 0
    ? { fp16: a.slice(Math.max(0, firstDiff - 3), firstDiff + 6), f32: b.slice(Math.max(0, firstDiff - 3), firstDiff + 6) }
    : null,
  fp16TranscriptHead: (fp16.transcript || "").slice(0, 160),
  f32TranscriptHead: (ref.text || "").slice(0, 160),
  fp16TranscriptTail: (fp16.transcript || "").slice(-160),
  f32TranscriptTail: (ref.text || "").slice(-160),
};
console.log("FP16_DIVERGENCE " + JSON.stringify(out));

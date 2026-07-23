// DIAGNOSTIC (untracked): isolate whether the FP16(4x4)-vs-F32(4x32) divergence
// comes from FP16 precision or from the 4->32 physical-rows change. Runs F32 at
// BOTH physical shapes (4 and 32) and diffs each against the FP16(4x4) tokens
// captured in the failing e2e artifact.
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

async function f32(rows) {
  return runStreamSession({
    config,
    planName: `probe-f32-4x${rows}`,
    audioPath: fixture.wavPath,
    mode: "80ms",
    env: {
      ...SESSION8_PRODUCTION_ENV,
      VOXTRAL_STREAM_DECODER: "reference",
      VOXTRAL_ENCODER_KV_TYPE: "f32",
      VOXTRAL_DECODER_KV_TYPE: "f32",
      VOXTRAL_ENC_KV_PHYSICAL_ROWS: String(rows),
    },
    timeoutMs: 600_000,
  });
}

function diff(a, b) {
  const n = Math.min(a.length, b.length);
  let first = -1, count = 0;
  for (let i = 0; i < n; i++) if (a[i] !== b[i]) { if (first < 0) first = i; count++; }
  return { lenA: a.length, lenB: b.length, firstDivergenceIndex: first, differing: count };
}

const f32_4 = await f32(4);
const f32_32 = await f32(32);

const out = {
  "FP16_4x4_vs_F32_4x4  (isolates FP16 precision)": {
    ...diff(fp16.tokens, f32_4.tokens),
    transcriptsIdentical: (fp16.transcript || "") === (f32_4.text || ""),
  },
  "F32_4x4_vs_F32_4x32  (isolates physical rows)": {
    ...diff(f32_4.tokens, f32_32.tokens),
    transcriptsIdentical: (f32_4.text || "") === (f32_32.text || ""),
  },
  "FP16_4x4_vs_F32_4x32 (both axes; the failing gate)": {
    ...diff(fp16.tokens, f32_32.tokens),
    transcriptsIdentical: (fp16.transcript || "") === (f32_32.text || ""),
  },
};
console.log("FP16_ISOLATE " + JSON.stringify(out, null, 1));

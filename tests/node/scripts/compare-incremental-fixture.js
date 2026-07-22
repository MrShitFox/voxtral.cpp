// Session 7 Stage-1 long-stream parity: finish-only reference vs device-resident
// incremental over the 2-minute fixture, which wraps the audio-embedding ring
// (~1500 groups vs cap 256) and exercises modulo indexing + backpressure.
import path from "node:path";
import { loadEnvironment } from "../config/environment.js";
import { runStreamSession } from "../helpers/stream.js";
import { normalizeFixtureOnGpu, syncFixture } from "../helpers/remote.js";

const config = loadEnvironment();
const base = { VOXTRAL_ENC_KV_LOGICAL_BATCH: "4", VOXTRAL_ENC_KV_PHYSICAL_ROWS: "32", VOXTRAL_ENCODER_TELEMETRY: "1" };
const longFixture = path.join(config.localRepo, "voxTest2min.m4a");

await syncFixture(longFixture, { config });
const fixture = await normalizeFixtureOnGpu({ config });
console.log(`fixture: ${fixture.durationMs.toFixed(0)}ms, ${fixture.sampleCount} samples`);

async function run(label, incremental, opts) {
  return runStreamSession({
    config, planName: `${label}`, audioPath: fixture.wavPath, maxTokens: 0, skipParity: true,
    // Session 7.1: incremental is the default; the oracle must ask for reference.
    env: { ...base, VOXTRAL_STREAM_DECODER: incremental ? "incremental" : "reference" },
    timeoutMs: 420_000, ...opts,
  });
}

const cases = [
  { label: "full", opts: { mode: "full" } },
  { label: "paced-80ms", opts: { realtimeMs: 80, mode: "80ms" } },
];

let allOk = true;
// One shared finish-only reference (mode full).
const ref = await run("ref-full", false, { mode: "full" });
console.log(`ref-full: state=${ref.state} nTokens=${ref.tokens?.length} text[0..60]=${JSON.stringify((ref.text||"").slice(0,60))}`);

for (const c of cases) {
  const inc = await run(`incr-${c.label}`, true, c.opts);
  const tok = JSON.stringify(ref.tokens) === JSON.stringify(inc.tokens);
  const txt = ref.text === inc.text;
  const ok = inc.state === "completed" && tok && txt;
  allOk = allOk && ok;
  console.log(`\n=== incr ${c.label} === ${ok ? "PASS" : "FAIL"}`);
  console.log(`  state=${inc.state} nTokens=${inc.tokens?.length} tokensMatch=${tok} textMatch=${txt}`);
  console.log(`  groupsCommitted=${inc.adapterGroupsCommitted} decoderSteps=${inc.decoderSteps} tokensEmitted=${inc.decoderTokensEmitted} tokensBeforeFinish=${inc.tokensBeforeFinish}`);
  console.log(`  firstDecoderStepMs=${inc.firstDecoderStepMs} firstTokenMs=${inc.firstTokenMs} firstVisibleTextMs=${inc.firstVisibleTextMs}`);
  if (!tok) {
    const a = ref.tokens || [], b = inc.tokens || [];
    for (let i = 0; i < Math.max(a.length, b.length); i++) if (a[i] !== b[i]) { console.log(`  first diff @${i}: ref=${a[i]} incr=${b[i]} (ref.len=${a.length} incr.len=${b.length})`); break; }
  }
}
console.log(`\nOVERALL: ${allOk ? "PASS" : "FAIL"}`);
process.exitCode = allOk ? 0 : 1;

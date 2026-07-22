// Session 7 Stage-1 parity check: finish-only reference vs device-resident
// incremental adapter+decoder, over the same feed plans. Tokens + transcript must
// be byte-identical.
import { loadEnvironment } from "../config/environment.js";
import { runStreamSession } from "../helpers/stream.js";

const config = loadEnvironment();
const base = { VOXTRAL_ENC_KV_LOGICAL_BATCH: "4", VOXTRAL_ENC_KV_PHYSICAL_ROWS: "32", VOXTRAL_ENCODER_TELEMETRY: "1" };
const plans = process.argv.slice(2).length ? process.argv.slice(2) : ["full", "80ms", "480ms"];

async function run(mode, incremental) {
  return runStreamSession({
    config,
    planName: `${incremental ? "incr" : "finish"}-${mode}`,
    mode,
    maxTokens: 0,
    env: incremental ? { ...base, VOXTRAL_STREAM_DECODER: "incremental" } : base,
    timeoutMs: 300_000,
  });
}

let allOk = true;
for (const mode of plans) {
  const ref = await run(mode, false);
  const inc = await run(mode, true);
  const tokMatch = JSON.stringify(ref.tokens) === JSON.stringify(inc.tokens);
  const txtMatch = ref.text === inc.text;
  const ok = ref.state === "completed" && inc.state === "completed" && tokMatch && txtMatch;
  allOk = allOk && ok;
  const evTypes = (inc.events ?? []).map((e) => e.type);
  const evCounts = evTypes.reduce((m, t) => ((m[t] = (m[t] || 0) + 1), m), {});
  console.log(`\n=== ${mode} === ${ok ? "PASS" : "FAIL"}`);
  console.log(`  ref:  state=${ref.state} nTokens=${ref.tokens?.length} text=${JSON.stringify(ref.text)}`);
  console.log(`  incr: state=${inc.state} nTokens=${inc.tokens?.length} text=${JSON.stringify(inc.text)}`);
  console.log(`  tokens match=${tokMatch}  text match=${txtMatch}  incr events=${JSON.stringify(evCounts)}`);
  if (!tokMatch) {
    // Show first divergence.
    const a = ref.tokens || [], b = inc.tokens || [];
    const n = Math.max(a.length, b.length);
    for (let i = 0; i < n; i++) if (a[i] !== b[i]) { console.log(`  first token diff @${i}: ref=${a[i]} incr=${b[i]}`); break; }
  }
}
console.log(`\nOVERALL: ${allOk ? "PASS" : "FAIL"}`);
process.exitCode = allOk ? 0 : 1;

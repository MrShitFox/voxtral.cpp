// Session 7 baseline oracle capture. Runs the CURRENT finish-only path on the
// checked-in short WAV across feed plans and prints tokens/text + the telemetry
// fields Session 7 will need to hold invariant. Read-only; no source changes.
import { loadEnvironment } from "../config/environment.js";
import { runStreamSession } from "../helpers/stream.js";

const config = loadEnvironment();
const envFor = (logical, physical) => ({
  VOXTRAL_ENC_KV_LOGICAL_BATCH: String(logical),
  VOXTRAL_ENC_KV_PHYSICAL_ROWS: String(physical),
  VOXTRAL_ENCODER_TELEMETRY: "1",
});

const plans = process.argv.slice(2).length ? process.argv.slice(2) : ["full", "80ms", "480ms"];
const out = [];
for (const mode of plans) {
  const r = await runStreamSession({
    config,
    planName: `baseline-${mode}`,
    mode,
    maxTokens: 0,
    env: envFor(4, 32),
    timeoutMs: 300_000,
  });
  out.push({
    mode,
    state: r.state,
    finishStatus: r.finishStatus,
    nTokens: r.tokens?.length ?? 0,
    tokens: r.tokens,
    text: r.text,
    encoderFrames: r.encoderFrames,
    encoderUniqueFrames: r.encoderUniqueFrames,
    encoderSha256: r.encoderSha256,
    finishFrontendMs: r.finishFrontendMs,
    finishEncoderMs: r.finishEncoderMs,
    finishDecoderMs: r.finishDecoderMs,
    finishLatencyMs: r.finishLatencyMs,
    pendingEvents: r.events?.length ?? 0,
    eventTypes: (r.events ?? []).map((e) => e.type),
  });
  console.error(`[baseline] ${mode}: state=${r.state} tokens=${r.tokens?.length} finishDecoderMs=${r.finishDecoderMs?.toFixed?.(1)}`);
}

// Cross-plan identity (the current hard invariant we must not regress).
const first = out[0];
const identical = out.every(
  (o) => JSON.stringify(o.tokens) === JSON.stringify(first.tokens) && o.text === first.text,
);
console.log(JSON.stringify({ identical, runs: out }, null, 2));

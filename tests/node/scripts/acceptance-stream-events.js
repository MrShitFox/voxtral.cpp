// Acceptance gate for the Session 7.1 event lifecycle and explicit backpressure.
// Part A (normal consumer): the incremental default emits a well-formed lifecycle —
//   TOKEN* PARTIAL_TEXT* FINAL_TEXT COMPLETED — with strictly increasing token
//   sequences (no loss/gap), monotonic partial revisions, a non-shrinking stable
//   prefix, exactly one FINAL_TEXT and one COMPLETED (COMPLETED last), nothing after
//   COMPLETED, and events_dropped == 0.
// Part B (stalled consumer): with a deliberately tiny event queue and no draining
//   until feed reports queue_full, feed raises explicit backpressure, NO mandatory
//   event is ever dropped (events_dropped == 0), and after draining the transcript
//   and token stream are byte-identical to Part A.
import { loadEnvironment } from "../config/environment.js";
import { writeArtifactBundle } from "../helpers/artifacts.js";
import { runProcess } from "../helpers/exec.js";
import { checkRemoteConnection, syncSources } from "../helpers/remote.js";
import { buildRemoteVulkan } from "../helpers/build.js";
import { runStreamSession } from "../helpers/stream.js";

const config = loadEnvironment();
const nodeCwd = new URL("..", import.meta.url).pathname;
const base = { VOXTRAL_ENC_KV_LOGICAL_BATCH: "4", VOXTRAL_ENC_KV_PHYSICAL_ROWS: "32", VOXTRAL_ENCODER_TELEMETRY: "1" };
const summary = { startedAt: new Date().toISOString(), command: "npm run acceptance:stream-events", steps: [] };

function gate(cond, msg) { if (!cond) throw new Error(msg); }

async function localStep(name, command, args, cwd = nodeCwd, timeoutMs = 1_200_000) {
  const r = await runProcess(command, args, { cwd, timeoutMs, onStdout: (c) => process.stdout.write(c), onStderr: (c) => process.stderr.write(c) });
  summary.steps.push({ name, exitCode: r.exitCode, wallMs: r.wallMs });
  gate(r.exitCode === 0, `${name}: exit ${r.exitCode}`);
}

// Verify the ordered event stream obeys the documented lifecycle contract.
function checkLifecycle(r, label) {
  const ev = r.events ?? [];
  const finals = ev.filter((e) => e.type === "final_text");
  const completed = ev.filter((e) => e.type === "completed");
  gate(finals.length === 1, `${label}: expected exactly 1 final_text, got ${finals.length}`);
  gate(completed.length === 1, `${label}: expected exactly 1 completed, got ${completed.length}`);
  const finalIdx = ev.findIndex((e) => e.type === "final_text");
  const compIdx = ev.findIndex((e) => e.type === "completed");
  gate(compIdx === ev.length - 1, `${label}: COMPLETED is not the last event`);
  gate(compIdx > finalIdx, `${label}: COMPLETED (${compIdx}) not after FINAL_TEXT (${finalIdx})`);
  // No TOKEN/PARTIAL after FINAL_TEXT; sequences strictly increasing from 1.
  let seq = 0, rev = 0, stable = 0, tokenCount = 0, partialCount = 0;
  for (let i = 0; i < ev.length; i++) {
    const e = ev[i];
    if (e.type === "token" || e.type === "partial_text") {
      gate(i < finalIdx, `${label}: ${e.type} after FINAL_TEXT at index ${i}`);
    }
    if (e.type === "token") {
      gate(e.sequence === seq + 1, `${label}: token sequence gap: expected ${seq + 1}, got ${e.sequence}`);
      seq = e.sequence; tokenCount++;
    }
    if (e.type === "partial_text") {
      gate(e.revision > rev, `${label}: partial revision not increasing (${e.revision} <= ${rev})`);
      gate(e.stablePrefixBytes >= stable, `${label}: stable prefix shrank (${e.stablePrefixBytes} < ${stable})`);
      rev = e.revision; stable = e.stablePrefixBytes; partialCount++;
    }
  }
  gate(tokenCount === r.decoderSteps, `${label}: token events ${tokenCount} != decoderSteps ${r.decoderSteps}`);
  gate(r.eventsDropped === 0, `${label}: events dropped ${r.eventsDropped} != 0`);
  return { tokenCount, partialCount };
}

async function main() {
  await localStep("unit", "npm", ["run", "test:unit"]);
  await localStep("local-build", "npm", ["run", "build:local"]);
  await localStep("cpp-unit", "ctest", ["--test-dir", config.localBuild, "--output-on-failure"], config.localRepo);
  await checkRemoteConnection({ config });
  await syncSources({ config });
  await buildRemoteVulkan({ config });

  // Part A — normal consumer draining every feed.
  const normal = await runStreamSession({ config, planName: "events-normal", mode: "80ms", maxTokens: 0, env: base, timeoutMs: 300_000 });
  gate(normal.state === "completed", `normal: state ${normal.state}`);
  const la = checkLifecycle(normal, "normal");
  summary.normal = { tokens: normal.tokens.length, tokenEvents: la.tokenCount, partialEvents: la.partialCount,
    eventsEmitted: normal.eventsEmitted, eventsDropped: normal.eventsDropped,
    partialEventsCoalesced: normal.partialEventsCoalesced, eventQueueHighWatermark: normal.eventQueueHighWatermark,
    text: normal.text };

  // Part B — stalled consumer + tiny queue: feed must backpressure, drop nothing,
  // and still produce the identical transcript once drained.
  const bp = await runStreamSession({ config, planName: "events-backpressure", mode: "80ms", maxTokens: 0,
    env: base, maxEvents: 8, backpressure: true, timeoutMs: 300_000 });
  gate(bp.state === "completed", `backpressure: state ${bp.state}`);
  gate(bp.backpressureObserved === true, `backpressure: feed never reported queue_full`);
  gate(bp.feedQueueFullReturns > 0, `backpressure: feedQueueFullReturns ${bp.feedQueueFullReturns} == 0`);
  gate(bp.eventQueueOverflowAttempts > 0, `backpressure: no overflow attempts recorded`);
  gate(bp.eventsDropped === 0, `backpressure: events dropped ${bp.eventsDropped} != 0 (mandatory events must never drop)`);
  // The drained event stream is still well-formed and loses no token.
  const lb = checkLifecycle(bp, "backpressure");
  gate(JSON.stringify(bp.tokens) === JSON.stringify(normal.tokens), `backpressure: token divergence vs normal drain`);
  gate(bp.text === normal.text, `backpressure: transcript divergence vs normal drain`);
  gate(lb.tokenCount === la.tokenCount, `backpressure: token event count ${lb.tokenCount} != normal ${la.tokenCount}`);
  summary.backpressure = { maxEventsBound: bp.maxEventsBound, feedQueueFullReturns: bp.feedQueueFullReturns,
    eventQueueOverflowAttempts: bp.eventQueueOverflowAttempts, eventsDropped: bp.eventsDropped,
    backpressureObserved: bp.backpressureObserved, tokenEvents: lb.tokenCount, lastFeedStatus: bp.lastFeedStatus,
    text: bp.text };

  summary.transcript = normal.text;
  summary.exitCode = 0;
}

try { await main(); }
catch (e) { summary.exitCode = 1; summary.error = e.message; process.exitCode = 1; }
finally {
  summary.finishedAt = new Date().toISOString();
  const artifact = await writeArtifactBundle({ config, testName: "stream-events-acceptance", backend: "Vulkan",
    command: "npm run acceptance:stream-events", result: summary,
    textArtifacts: summary.transcript ? { "transcript.txt": summary.transcript } : {} });
  console.log(`[stream-events] ${summary.exitCode === 0 ? "PASS" : "FAIL"} summary: ${artifact.directory}`);
  if (summary.error) console.error(`[stream-events] error: ${summary.error}`);
}

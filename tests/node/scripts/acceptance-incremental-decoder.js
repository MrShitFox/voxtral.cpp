// Acceptance gate for the Session 7 device-resident incremental adapter + decoder.
// Proves, on the checked-in short WAV across every feed plan, that the incremental
// path is token/transcript-identical to the finish-only reference, emits a correct
// monotonic TOKEN / PARTIAL_TEXT / FINAL_TEXT / COMPLETED event stream, and keeps
// the adapter/decoder path device-resident (zero adapter D2H/H2D, zero full-logits
// D2H, ~4 bytes/token id readback).
import { loadEnvironment } from "../config/environment.js";
import { writeArtifactBundle } from "../helpers/artifacts.js";
import { runProcess } from "../helpers/exec.js";
import { checkRemoteConnection, syncSources } from "../helpers/remote.js";
import { buildRemoteVulkan } from "../helpers/build.js";
import { runStreamSession } from "../helpers/stream.js";

const config = loadEnvironment();
const nodeCwd = new URL("..", import.meta.url).pathname;
const base = { VOXTRAL_ENC_KV_LOGICAL_BATCH: "4", VOXTRAL_ENC_KV_PHYSICAL_ROWS: "32", VOXTRAL_ENCODER_TELEMETRY: "1" };
const summary = { startedAt: new Date().toISOString(), command: "npm run acceptance:incremental-decoder", plans: [], steps: [] };

function gate(cond, msg) { if (!cond) throw new Error(msg); }

async function localStep(name, command, args, cwd = nodeCwd, timeoutMs = 1_200_000) {
  const r = await runProcess(command, args, { cwd, timeoutMs, onStdout: (c) => process.stdout.write(c), onStderr: (c) => process.stderr.write(c) });
  summary.steps.push({ name, exitCode: r.exitCode, wallMs: r.wallMs });
}

function checkEvents(r, label) {
  const ev = r.events ?? [];
  const tokens = ev.filter((e) => e.type === "token");
  const partials = ev.filter((e) => e.type === "partial_text");
  const finals = ev.filter((e) => e.type === "final_text");
  const completed = ev.filter((e) => e.type === "completed");
  gate(finals.length === 1, `${label}: expected exactly 1 final_text, got ${finals.length}`);
  gate(completed.length === 1, `${label}: expected exactly 1 completed, got ${completed.length}`);
  // TOKEN sequences strictly monotonic starting at 1, none lost.
  for (let i = 0; i < tokens.length; i++) {
    gate(tokens[i].sequence === i + 1, `${label}: token sequence gap at ${i}: got ${tokens[i].sequence}`);
  }
  // One token event per decoder step (includes the terminal EOS token event).
  gate(tokens.length === r.decoderSteps, `${label}: token events ${tokens.length} != decoderSteps ${r.decoderSteps}`);
  // PARTIAL_TEXT revisions strictly monotonic and stable-prefix never shrinks.
  let lastRev = 0, lastStable = 0;
  for (const p of partials) {
    gate(p.revision > lastRev, `${label}: partial revision not increasing (${p.revision} <= ${lastRev})`);
    gate(p.stablePrefixBytes >= lastStable, `${label}: stable prefix shrank (${p.stablePrefixBytes} < ${lastStable})`);
    lastRev = p.revision; lastStable = p.stablePrefixBytes;
  }
  return { nToken: tokens.length, nPartial: partials.length };
}

function checkDeviceResident(r, label) {
  gate(r.usesIncrementalDecode === true, `${label}: not on incremental path`);
  gate(r.adapterInputD2hBytes === 0, `${label}: adapter input D2H ${r.adapterInputD2hBytes} != 0`);
  gate(r.adapterOutputD2hBytes === 0, `${label}: adapter output D2H ${r.adapterOutputD2hBytes} != 0`);
  gate(r.logitsD2hBytes === 0, `${label}: full-logits D2H ${r.logitsD2hBytes} != 0`);
  gate(r.tokenIdD2hBytes === 4 * r.decoderSteps, `${label}: token-id D2H ${r.tokenIdD2hBytes} != 4*${r.decoderSteps}`);
  // Each decoder position advances exactly once (no replay): steps == tokens + terminal EOS(0/1).
  const eos = r.decoderTokensEmitted < r.decoderSteps ? 1 : 0;
  gate(r.decoderSteps === r.decoderTokensEmitted + eos, `${label}: decoder replay (steps=${r.decoderSteps} emitted=${r.decoderTokensEmitted})`);
  gate(r.decoderTokensEmitted === (r.tokens?.length ?? 0), `${label}: emitted ${r.decoderTokensEmitted} != final tokens ${r.tokens?.length}`);
}

async function main() {
  await localStep("unit", "npm", ["run", "test:unit"]);
  await localStep("local-build", "npm", ["run", "build:local"]);
  await localStep("cpp-unit", "ctest", ["--test-dir", config.localBuild, "--output-on-failure"], config.localRepo);
  await checkRemoteConnection({ config });
  await syncSources({ config });
  await buildRemoteVulkan({ config });

  const plans = ["full", "80ms", "160ms", "480ms", "seeded-random:20260722"];
  const refs = [], incs = [];
  for (const mode of plans) {
    const planName = mode.replaceAll(":", "-");
    const ref = await runStreamSession({ config, planName: `ref-${planName}`, mode, maxTokens: 0, env: base, timeoutMs: 300_000 });
    const inc = await runStreamSession({ config, planName: `inc-${planName}`, mode, maxTokens: 0, env: { ...base, VOXTRAL_STREAM_DECODER: "incremental" }, timeoutMs: 300_000 });
    gate(ref.state === "completed" && inc.state === "completed", `${mode}: not completed (ref=${ref.state} inc=${inc.state})`);
    gate(JSON.stringify(ref.tokens) === JSON.stringify(inc.tokens), `${mode}: token divergence vs finish-only reference`);
    gate(ref.text === inc.text, `${mode}: transcript divergence vs finish-only reference`);
    checkDeviceResident(inc, mode);
    const ev = checkEvents(inc, mode);
    // FINAL_TEXT carries the full transcript.
    const finalEv = (inc.events ?? []).find((e) => e.type === "final_text");
    refs.push(ref); incs.push(inc);
    summary.plans.push({ mode, nTokens: inc.tokens.length, text: inc.text, decoderSteps: inc.decoderSteps,
      adapterGroupsCommitted: inc.adapterGroupsCommitted, adapterCommitCalls: inc.adapterCommitCalls,
      tokensBeforeFinish: inc.tokensBeforeFinish, tokensFlushedAtFinish: inc.tokensFlushedAtFinish,
      tokenIdD2hBytes: inc.tokenIdD2hBytes, adapterInputD2hBytes: inc.adapterInputD2hBytes,
      logitsD2hBytes: inc.logitsD2hBytes, partialTextRevision: inc.partialTextRevision,
      firstAdapterCommitMs: inc.firstAdapterCommitMs, firstDecoderStepMs: inc.firstDecoderStepMs,
      firstTokenMs: inc.firstTokenMs, firstVisibleTextMs: inc.firstVisibleTextMs, events: ev });
  }
  // Cross-plan identity (incremental output is invariant to feed chunking).
  for (const inc of incs) {
    gate(JSON.stringify(inc.tokens) === JSON.stringify(incs[0].tokens), `${inc.planName}: cross-plan token divergence`);
    gate(inc.text === incs[0].text, `${inc.planName}: cross-plan transcript divergence`);
  }
  summary.transcript = incs[0].text;
  summary.tokens = incs[0].tokens;
  summary.exitCode = 0;
}

try { await main(); }
catch (e) { summary.exitCode = 1; summary.error = e.message; process.exitCode = 1; }
finally {
  summary.finishedAt = new Date().toISOString();
  const artifact = await writeArtifactBundle({ config, testName: "incremental-decoder-acceptance", backend: "Vulkan",
    command: "npm run acceptance:incremental-decoder", result: summary,
    textArtifacts: summary.transcript ? { "transcript.txt": summary.transcript } : {} });
  console.log(`[incremental-decoder] ${summary.exitCode === 0 ? "PASS" : "FAIL"} summary: ${artifact.directory}`);
  if (summary.error) console.error(`[incremental-decoder] error: ${summary.error}`);
}

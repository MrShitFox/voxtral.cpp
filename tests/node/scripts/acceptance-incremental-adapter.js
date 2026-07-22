// Acceptance gate for the Session 7.1 device-resident incremental adapter.
// Proves, on the checked-in short WAV across every feed plan, that the adapter runs
// DURING feed in groups of four encoder frames, stays fully device-resident (zero
// adapter input/output D2H, zero host encoder accumulation), commits every group
// exactly once (adapter work ratio == 1.0, chunk-invariant), and yields tokens /
// transcript byte-identical to the finish-only reference. Incremental is the
// production default; the oracle side is the explicit reference decoder.
import { loadEnvironment } from "../config/environment.js";
import { writeArtifactBundle } from "../helpers/artifacts.js";
import { runProcess } from "../helpers/exec.js";
import { checkRemoteConnection, syncSources } from "../helpers/remote.js";
import { buildRemoteVulkan } from "../helpers/build.js";
import { runStreamSession } from "../helpers/stream.js";

const config = loadEnvironment();
const nodeCwd = new URL("..", import.meta.url).pathname;
const base = { VOXTRAL_ENC_KV_LOGICAL_BATCH: "4", VOXTRAL_ENC_KV_PHYSICAL_ROWS: "32", VOXTRAL_ENCODER_TELEMETRY: "1" };
const summary = { startedAt: new Date().toISOString(), command: "npm run acceptance:incremental-adapter", plans: [], steps: [] };

function gate(cond, msg) { if (!cond) throw new Error(msg); }

async function localStep(name, command, args, cwd = nodeCwd, timeoutMs = 1_200_000) {
  const r = await runProcess(command, args, { cwd, timeoutMs, onStdout: (c) => process.stdout.write(c), onStderr: (c) => process.stderr.write(c) });
  summary.steps.push({ name, exitCode: r.exitCode, wallMs: r.wallMs });
  gate(r.exitCode === 0, `${name}: exit ${r.exitCode}`);
}

// The adapter maps each group of DOWNSAMPLE=4 encoder frames to one audio embedding.
const DOWNSAMPLE = 4;

function checkAdapter(r, label) {
  gate(r.usesIncrementalDecode === true, `${label}: not on incremental path`);
  gate(r.decoderMode === "incremental", `${label}: decoderMode ${r.decoderMode} != incremental`);
  // Device residency: the adapter reads the encoder-output ring and writes the
  // audio-embedding ring entirely on-device.
  gate(r.adapterInputD2hBytes === 0, `${label}: adapter input D2H ${r.adapterInputD2hBytes} != 0`);
  gate(r.adapterOutputD2hBytes === 0, `${label}: adapter output D2H ${r.adapterOutputD2hBytes} != 0`);
  gate(r.encoderOutputD2hBytes === 0, `${label}: encoder-output D2H ${r.encoderOutputD2hBytes} != 0`);
  gate(r.encoderOutputAccumulatedBytes === 0, `${label}: host encoder accumulation ${r.encoderOutputAccumulatedBytes} != 0`);
  // Every complete group is committed exactly once (no replay): committed == unique
  // groups == floor(encoderFrames / 4), and each commit call advances >= 1 group.
  const uniqueGroups = Math.floor(r.encoderFrames / DOWNSAMPLE);
  gate(r.adapterGroupsCommitted === uniqueGroups,
    `${label}: adapter groups committed ${r.adapterGroupsCommitted} != unique ${uniqueGroups}`);
  gate(r.adapterCommitCalls > 0 && r.adapterCommitCalls <= r.adapterGroupsCommitted,
    `${label}: adapter commit calls ${r.adapterCommitCalls} out of range (groups ${r.adapterGroupsCommitted})`);
  // The adapter runs during feed: most groups are committed before finish().
  gate(r.firstAdapterCommitMs > 0, `${label}: adapter never committed a group during the run`);
  return { uniqueGroups, workRatio: r.adapterGroupsCommitted / Math.max(1, uniqueGroups) };
}

async function main() {
  await localStep("unit", "npm", ["run", "test:unit"]);
  await localStep("local-build", "npm", ["run", "build:local"]);
  await localStep("cpp-unit", "ctest", ["--test-dir", config.localBuild, "--output-on-failure"], config.localRepo);
  await checkRemoteConnection({ config });
  await syncSources({ config });
  await buildRemoteVulkan({ config });

  const plans = ["full", "80ms", "160ms", "480ms", "seeded-random:20260722"];
  const incs = [];
  for (const mode of plans) {
    const planName = mode.replaceAll(":", "-");
    const ref = await runStreamSession({ config, planName: `ref-${planName}`, mode, maxTokens: 0, env: { ...base, VOXTRAL_STREAM_DECODER: "reference" }, timeoutMs: 300_000 });
    const inc = await runStreamSession({ config, planName: `inc-${planName}`, mode, maxTokens: 0, env: base, timeoutMs: 300_000 });
    gate(ref.state === "completed" && inc.state === "completed", `${mode}: not completed (ref=${ref.state} inc=${inc.state})`);
    const a = checkAdapter(inc, mode);
    gate(Math.abs(a.workRatio - 1.0) < 1e-9, `${mode}: adapter work ratio ${a.workRatio} != 1.0`);
    gate(JSON.stringify(ref.tokens) === JSON.stringify(inc.tokens), `${mode}: token divergence vs reference`);
    gate(ref.text === inc.text, `${mode}: transcript divergence vs reference`);
    incs.push(inc);
    summary.plans.push({ mode, encoderFrames: inc.encoderFrames, uniqueGroups: a.uniqueGroups,
      adapterGroupsCommitted: inc.adapterGroupsCommitted, adapterCommitCalls: inc.adapterCommitCalls,
      adapterWorkRatio: a.workRatio, adapterInputD2hBytes: inc.adapterInputD2hBytes,
      adapterOutputD2hBytes: inc.adapterOutputD2hBytes, encoderOutputD2hBytes: inc.encoderOutputD2hBytes,
      firstAdapterCommitMs: inc.firstAdapterCommitMs, nTokens: inc.tokens.length });
  }
  // Chunk invariance: adapter group count + tokens identical across all feed plans.
  for (const inc of incs) {
    gate(inc.adapterGroupsCommitted === incs[0].adapterGroupsCommitted, `${inc.planName}: cross-plan adapter group divergence`);
    gate(JSON.stringify(inc.tokens) === JSON.stringify(incs[0].tokens), `${inc.planName}: cross-plan token divergence`);
  }
  summary.transcript = incs[0].text;
  summary.tokens = incs[0].tokens;
  summary.exitCode = 0;
}

try { await main(); }
catch (e) { summary.exitCode = 1; summary.error = e.message; process.exitCode = 1; }
finally {
  summary.finishedAt = new Date().toISOString();
  const artifact = await writeArtifactBundle({ config, testName: "incremental-adapter-acceptance", backend: "Vulkan",
    command: "npm run acceptance:incremental-adapter", result: summary,
    textArtifacts: summary.transcript ? { "transcript.txt": summary.transcript } : {} });
  console.log(`[incremental-adapter] ${summary.exitCode === 0 ? "PASS" : "FAIL"} summary: ${artifact.directory}`);
  if (summary.error) console.error(`[incremental-adapter] error: ${summary.error}`);
}

import path from "node:path";

import { loadEnvironment } from "../config/environment.js";
import { runRemote, shellQuote } from "./remote.js";

/** Absolute path of the streaming-skeleton harness on the GPU host. */
export function remoteStreamBinary(config = loadEnvironment()) {
  return path.posix.join(config.remoteBuild, "voxtral-stream-test");
}

/** Turn a createChunkPlan() result into the plain sample-count list the harness reads. */
export function planCounts(plan) {
  return plan.events.map((event) => event.sampleCount);
}

/** Write a feed plan (one sample count per line) to the remote repo over SSH stdin. */
export async function writeRemotePlan(remotePath, counts, { config = loadEnvironment() } = {}) {
  const text = `${counts.join("\n")}\n`;
  await runRemote(`cat > ${shellQuote(remotePath)}`, { config, input: text, timeoutMs: 30_000 });
  return remotePath;
}

function detectEvidence(stdout, stderr) {
  const combined = `${stdout}\n${stderr}`;
  return {
    vulkanEnabled: /backend:\s*VULKAN|runtime_backends:\s*[^\n]*Vulkan\([1-9]/iu.test(combined),
    rx6600Detected: /AMD Radeon RX 6600|RADV NAVI23/iu.test(combined),
    cpuOnlyFallbackDetected: /no GPU backend available, using CPU/iu.test(combined),
  };
}

/**
 * Drive the streaming skeleton once with a given feed plan and return its parsed
 * JSON result plus backend evidence. `counts` is authoritative; when omitted the
 * harness feeds the whole clip in one call (built-in "full" mode).
 */
export async function runStreamSession({
  config = loadEnvironment(),
  planName = "plan",
  counts = null,
  mode = null,
  audioPath = null,
  modelPath = null,
  maxTokens = 0,
  timeoutMs = 300_000,
} = {}) {
  const binary = remoteStreamBinary(config);
  const audio = audioPath ?? config.remoteSmokeAudio;
  const model = modelPath ?? config.remoteModel;

  const args = [
    shellQuote(binary),
    "--model", shellQuote(model),
    "--wav", shellQuote(audio),
    "--gpu", "vulkan",
    "--max-tokens", String(maxTokens),
  ];

  let planFile = null;
  if (counts) {
    planFile = path.posix.join(config.remoteRepo, `.stream-plan-${planName}.txt`);
    await writeRemotePlan(planFile, counts, { config });
    args.push("--plan-file", shellQuote(planFile));
  } else if (mode) {
    args.push("--mode", mode);
  }

  const command = `cd ${shellQuote(config.remoteRepo)} && ${args.join(" ")}`;
  const proc = await runRemote(command, { config, timeoutMs });

  const jsonLine = proc.stdout
    .split(/\r?\n/u)
    .map((line) => line.trim())
    .reverse()
    .find((line) => line.startsWith("{"));
  if (!jsonLine) {
    throw new Error(`stream harness produced no JSON\n--- stdout ---\n${proc.stdout}\n--- stderr ---\n${proc.stderr}`);
  }

  const result = JSON.parse(jsonLine);
  result.planName = planName;
  result.planFile = planFile;
  result.commandLine = command;
  result.wallMs = proc.wallMs;
  result.evidence = detectEvidence(proc.stdout, proc.stderr);
  result.stdout = proc.stdout;
  result.stderr = proc.stderr;
  return result;
}

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
 * Drive the incremental stream runtime once with a given feed plan and return its parsed
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
  ab = false,
  kvParity = false,
  warmup = false,
  env = null,
  realtimeMs = 0,
  skipParity = false,
  manualOracle = false,
  monitorMemory = false,
  maxEvents = 0,
  backpressure = false,
  captureRolloverMemory = false,
  mallocTrimAfter = false,
  discardEventHistory = false,
  syntheticSeconds = 0,
  maxTotalSamples = 0,
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

  // --ab: load the model once, run two streams (A then B) each owning its own
  // context, and emit both results plus a distinctContexts flag.
  if (ab) {
    args.push("--ab");
  }
  if (kvParity) {
    args.push("--kv-parity");
  }
  if (warmup) {
    args.push("--warmup");
  }
  if (realtimeMs > 0) {
    args.push("--realtime-ms", String(realtimeMs));
  }
  if (skipParity) {
    args.push("--skip-parity");
  }
  if (manualOracle) {
    args.push("--manual-oracle");
  }
  if (maxEvents > 0) {
    args.push("--max-events", String(maxEvents));
  }
  if (backpressure) {
    args.push("--backpressure");
  }
  if (captureRolloverMemory) {
    args.push("--capture-rollover-memory");
  }
  if (mallocTrimAfter) {
    args.push("--malloc-trim-after");
  }
  if (discardEventHistory) {
    args.push("--discard-event-history");
  }
  if (syntheticSeconds > 0) {
    args.push("--synthetic-seconds", String(syntheticSeconds));
  }
  if (maxTotalSamples > 0) {
    args.push("--max-total-samples", String(maxTotalSamples));
  }

  let planFile = null;
  if (counts) {
    planFile = path.posix.join(config.remoteRepo, `.stream-plan-${planName}.txt`);
    await writeRemotePlan(planFile, counts, { config });
    args.push("--plan-file", shellQuote(planFile));
  } else if (mode) {
    args.push("--mode", mode);
  }

  // Optional environment (e.g. VOXTRAL_ENCODER_STRATEGY=reference, VOXTRAL_ENC_KV_GRID).
  const envPrefix = env
    ? Object.entries(env).map(([k, v]) => `${k}=${shellQuote(String(v))}`).join(" ") + " "
    : "";
  const streamCommand = `${envPrefix}${args.join(" ")}`;
  let command = `cd ${shellQuote(config.remoteRepo)} && ${streamCommand}`;
  if (monitorMemory) {
    const safePlan = String(planName).replaceAll(/[^a-zA-Z0-9_-]/gu, "-");
    const stem = `/tmp/voxtral-memory-${process.pid}-${Date.now()}-${safePlan}`;
    const stdoutPath = `${stem}.stdout`;
    const stderrPath = `${stem}.stderr`;
    const vramPath = "/sys/class/drm/card1/device/mem_info_vram_used";
    command = [
      `cd ${shellQuote(config.remoteRepo)} && (`,
      `baseline_vram=$(cat ${shellQuote(vramPath)} 2>/dev/null || printf '0')`,
      `${streamCommand} >${shellQuote(stdoutPath)} 2>${shellQuote(stderrPath)} & stream_pid=$!`,
      "peak_rss=0; peak_vram=$baseline_vram",
      "while kill -0 $stream_pid 2>/dev/null; do",
      "rss=$(awk '/VmHWM:/ {print $2}' /proc/$stream_pid/status 2>/dev/null); rss=${rss:-0}",
      `vram=$(cat ${shellQuote(vramPath)} 2>/dev/null || printf '0'); vram=\${vram:-0}`,
      "if [ $rss -gt $peak_rss ]; then peak_rss=$rss; fi",
      "if [ $vram -gt $peak_vram ]; then peak_vram=$vram; fi",
      "sleep 0.05",
      "done",
      "wait $stream_pid; stream_status=$?",
      `final_vram=$(cat ${shellQuote(vramPath)} 2>/dev/null || printf '0'); final_vram=\${final_vram:-0}`,
      `cat ${shellQuote(stdoutPath)}`,
      `cat ${shellQuote(stderrPath)} >&2`,
      "printf '[VOXTRAL_MONITOR] baselineVramBytes=%s peakVramBytes=%s finalVramBytes=%s peakRssKiB=%s\\n' \"$baseline_vram\" \"$peak_vram\" \"$final_vram\" \"$peak_rss\" >&2",
      `rm -f ${shellQuote(stdoutPath)} ${shellQuote(stderrPath)}`,
      "exit $stream_status",
      ")",
    ].join("\n");
  }
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
  result.childExited = true;
  result.childExitRssKiB = 0;
  result.evidence = detectEvidence(proc.stdout, proc.stderr);
  const memory = proc.stderr.match(
    /\[VOXTRAL_MONITOR\] baselineVramBytes=(\d+) peakVramBytes=(\d+) finalVramBytes=(\d+) peakRssKiB=(\d+)/u,
  );
  if (memory) {
    result.baselineVramBytes = Number(memory[1]);
    result.peakVramBytes = Number(memory[2]);
    result.finalVramBytes = Number(memory[3]);
    result.peakRssKiB = Number(memory[4]);
  }
  result.stdout = proc.stdout;
  result.stderr = proc.stderr;
  return result;
}

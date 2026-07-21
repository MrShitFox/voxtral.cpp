import { loadEnvironment } from "../config/environment.js";
import { inspectWav } from "./audio.js";
import { writeArtifactBundle } from "./artifacts.js";
import { runRemote, shellQuote } from "./remote.js";

/** Normalize current human-readable CLI output into evidence suitable for assertions. */
export function parseInferenceOutput(result) {
  const combined = `${result.stdout}\n${result.stderr}`;
  const transcript = result.stdout
    .split(/\r?\n/u)
    .map((line) => line.trim())
    .find((line) => line && !line.startsWith("[tokens]") && line !== "[no-transcript]") ?? "";
  const timingMatches = [...result.stderr.matchAll(/(?:^|\n)(?:voxtral_[A-Z]: )?([^\n:]+):[^\n]*?([0-9]+(?:\.[0-9]+)?) ms/gmu)];
  const timings = Object.fromEntries(timingMatches.map((match) => [match[1].trim(), Number(match[2])]));
  const vulkanEnabled = /backend:\s*VULKAN|GGML_USE_VULKAN=ON|runtime_backends:\s*[^\n]*Vulkan\([1-9]/iu.test(combined);
  const rx6600Detected = /AMD Radeon RX 6600|RADV NAVI23/iu.test(combined);
  const cpuOnlyFallbackDetected = /no GPU backend available, using CPU|backend:\s*CPU(?:\s|$)/iu.test(combined);
  return {
    exitCode: result.exitCode,
    transcript,
    backend: vulkanEnabled ? "Vulkan" : cpuOnlyFallbackDetected ? "CPU" : "unknown",
    device: rx6600Detected ? "AMD Radeon RX 6600" : "unknown",
    wallMs: result.wallMs,
    timings,
    stdout: result.stdout,
    stderr: result.stderr,
    evidence: { vulkanEnabled, rx6600Detected, cpuOnlyFallbackDetected },
  };
}

/** Execute the current batch CLI remotely and optionally persist a full diagnostic bundle. */
export async function runBatchInference(options = {}) {
  const config = options.config ?? loadEnvironment();
  const binaryPath = options.binaryPath ?? config.remoteBinary;
  const modelPath = options.modelPath ?? config.remoteModel;
  const audioPath = options.audioPath ?? config.remoteSmokeAudio;
  const preflight = [
    `test -x ${shellQuote(binaryPath)}`,
    `test -s ${shellQuote(modelPath)}`,
    `test -s ${shellQuote(audioPath)}`,
  ].join(" && ");
  await runRemote(preflight, { config, timeoutMs: 30_000 });

  const command = [
    `cd ${shellQuote(config.remoteRepo)} &&`,
    shellQuote(binaryPath),
    "--model", shellQuote(modelPath),
    "--audio", shellQuote(audioPath),
    "--gpu vulkan",
    "--log-level info",
  ].join(" ");
  const processResult = await runRemote(command, { config, timeoutMs: options.timeoutMs ?? 180_000 });
  const result = parseInferenceOutput(processResult);
  result.commandLine = command;

  if (options.artifacts !== false) {
    const audioMetadata = options.audioMetadata ?? (audioPath === config.remoteSmokeAudio
      ? { ...(await inspectWav(config.localSmokeAudio)), remotePath: audioPath }
      : { remotePath: audioPath });
    result.artifact = await writeArtifactBundle({
      config,
      testName: options.testName ?? "gpu-smoke-inference",
      result,
      command,
      binaryPath,
      modelPath,
      audioMetadata,
      backend: result.backend,
    });
  }
  return result;
}

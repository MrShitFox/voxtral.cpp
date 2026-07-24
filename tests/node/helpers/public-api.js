import path from "node:path";

import { loadEnvironment } from "../config/environment.js";
import { buildRemoteVulkan, syncToGpu } from "./build.js";
import {
  checkRemoteConnection,
  runRemote,
  shellQuote,
} from "./remote.js";

const ENV_NAME = /^[A-Z_][A-Z0-9_]*$/u;

export const PUBLIC_PROFILE_ENV = Object.freeze({
  VOXTRAL_PROFILE: "1",
  VOXTRAL_ENCODER_KV_TYPE: "f32",
  VOXTRAL_DECODER_KV_TYPE: "f16",
  VOXTRAL_ENC_KV_LOGICAL_BATCH: "4",
  VOXTRAL_ENC_KV_PHYSICAL_ROWS: "4",
});

export function remotePublicApiBinary(config = loadEnvironment()) {
  return path.posix.join(
    config.remoteBuild,
    "voxtral_public_stream_api_test",
  );
}

export async function preparePublicApi({
  config = loadEnvironment(),
  forceConfigure = false,
} = {}) {
  const connection = await checkRemoteConnection({ config });
  const sync = await syncToGpu({ config });
  const build = await buildRemoteVulkan({ config, forceConfigure });
  const binary = remotePublicApiBinary(config);
  await runRemote(`test -x ${shellQuote(binary)}`, { config });
  return { connection, sync, build, binary };
}

function environmentPrefix(environment) {
  return Object.entries(environment).map(([name, value]) => {
    if (!ENV_NAME.test(name)) {
      throw new Error(`unsafe public API environment name: ${name}`);
    }
    return `${name}=${shellQuote(String(value))}`;
  }).join(" ");
}

function parseJsonResult(proc) {
  const jsonLine = proc.stdout
    .split(/\r?\n/u)
    .map((line) => line.trim())
    .reverse()
    .find((line) => line.startsWith("{"));
  if (!jsonLine) {
    throw new Error(
      `public API harness produced no JSON\n--- stdout ---\n${proc.stdout}` +
      `\n--- stderr ---\n${proc.stderr}`,
    );
  }
  return JSON.parse(jsonLine);
}

export async function runPublicApi({
  config = loadEnvironment(),
  audioPath = config.remoteSmokeAudio,
  syntheticSeconds = 0,
  modelPath = config.remoteModel,
  chunkSamples = 1280,
  eventCapacity = 4096,
  iterations = 1,
  randomSeed = null,
  delayedConsumer = false,
  lifecycle = false,
  paced = false,
  monitorMemory = false,
  env = {},
  timeoutMs = 300_000,
} = {}) {
  const binary = remotePublicApiBinary(config);
  const args = [
    shellQuote(binary),
    "--model", shellQuote(modelPath),
  ];
  if (syntheticSeconds > 0) {
    args.push("--synthetic-seconds", String(syntheticSeconds));
  } else {
    args.push("--audio", shellQuote(audioPath));
  }
  args.push(
    "--chunk-samples", String(chunkSamples),
    "--event-capacity", String(eventCapacity),
    "--iterations", String(iterations),
  );
  if (randomSeed !== null) {
    args.push("--random-seed", String(randomSeed));
  }
  if (delayedConsumer) args.push("--delayed-consumer");
  if (lifecycle) args.push("--lifecycle");
  if (paced) args.push("--paced");

  const environment = { ...PUBLIC_PROFILE_ENV, ...env };
  const launchCommand = [
    environmentPrefix(environment),
    "exec",
    args.join(" "),
  ].join(" ");
  const directCommand = [
    `cd ${shellQuote(config.remoteRepo)} &&`,
    launchCommand,
  ].join(" ");
  let command = directCommand;
  if (monitorMemory) {
    const stem =
      `/tmp/voxtral-public-api-${process.pid}-${Date.now()}`;
    const stdoutPath = `${stem}.stdout`;
    const stderrPath = `${stem}.stderr`;
    const vramPath = "/sys/class/drm/card1/device/mem_info_vram_used";
    command = [
      `baseline_vram=$(cat ${shellQuote(vramPath)} 2>/dev/null || printf '0')`,
      `${directCommand} >${shellQuote(stdoutPath)} 2>${shellQuote(stderrPath)} & child=$!`,
      "peak_rss=0; peak_vram=$baseline_vram",
      "while kill -0 $child 2>/dev/null; do",
      "rss=$(awk '/VmHWM:/ {print $2}' /proc/$child/status 2>/dev/null); rss=${rss:-0}",
      `vram=$(cat ${shellQuote(vramPath)} 2>/dev/null || printf '0'); vram=\${vram:-0}`,
      "if [ $rss -gt $peak_rss ]; then peak_rss=$rss; fi",
      "if [ $vram -gt $peak_vram ]; then peak_vram=$vram; fi",
      "sleep 0.05",
      "done",
      "wait $child; child_status=$?",
      `final_vram=$(cat ${shellQuote(vramPath)} 2>/dev/null || printf '0')`,
      `cat ${shellQuote(stdoutPath)}`,
      `cat ${shellQuote(stderrPath)} >&2`,
      "printf '[VOXTRAL_PUBLIC_MONITOR] baselineVramBytes=%s peakVramBytes=%s finalVramBytes=%s peakRssKiB=%s\\n' \"$baseline_vram\" \"$peak_vram\" \"$final_vram\" \"$peak_rss\" >&2",
      `rm -f ${shellQuote(stdoutPath)} ${shellQuote(stderrPath)}`,
      "exit $child_status",
    ].join("\n");
  }
  const proc = await runRemote(command, { config, timeoutMs });
  const result = parseJsonResult(proc);
  result.commandLine = command;
  result.wallMs = proc.wallMs;
  result.stdout = proc.stdout;
  result.stderr = proc.stderr;
  const memory = proc.stderr.match(
    /\[VOXTRAL_PUBLIC_MONITOR\] baselineVramBytes=(\d+) peakVramBytes=(\d+) finalVramBytes=(\d+) peakRssKiB=(\d+)/u,
  );
  if (memory) {
    result.baselineVramBytes = Number(memory[1]);
    result.peakVramBytes = Number(memory[2]);
    result.finalVramBytes = Number(memory[3]);
    result.peakRssKiB = Number(memory[4]);
  }
  return result;
}

export function expectedPublicEventSignature(tokenCount) {
  return `${"TP".repeat(tokenCount)}FC`;
}

export function publicRunChecks(run, { requireBackpressure = false } = {}) {
  return {
    harnessOk: run.ok === true,
    allAudioConsumed: run.samplesOffered === run.samplesConsumed,
    noUnaccountedRetry:
      Number.isSafeInteger(run.samplesRetried) && run.samplesRetried >= 0,
    backpressure:
      !requireBackpressure || (run.queueFullReturns > 0 && run.delayedConsumer),
    eventOrdering: run.eventOrderingOk === true,
    eventValue: run.eventValueOk === true,
    partialMonotonic: run.partialMonotonic === true,
    terminalOrdering: run.terminalOrderingOk === true,
    eventSignature:
      run.eventSignature === expectedPublicEventSignature(run.tokenCount),
    truncationSignalled:
      Buffer.byteLength(run.transcript ?? "", "utf8") <
        4096 ||
      run.truncatedEvents > 0,
    finishIdempotent: run.finishIdempotent === true,
    feedAfterFinishRejected: run.feedAfterFinishRejected === true,
    structuredStreamError: run.structuredStreamErrorOk === true,
    structuredQueueError: run.queueErrorOk === true,
    resetPristine: run.resetPristine === true,
    capabilities: run.capabilitiesOk === true,
    parameterValidation: run.parameterValidationOk === true,
    destroyStates:
      run.destroy?.nullSafe === true &&
      run.destroy?.created === true &&
      run.destroy?.active === true &&
      run.destroy?.cancelled === true &&
      run.destroy?.completed === true &&
      run.destroy?.underBackpressure === true &&
      run.destroy?.finishUnderBackpressure === true &&
      run.destroy?.cancelUnderBackpressure === true &&
      run.destroy?.leaseReleased === true,
    noKvBytesMoved: run.metrics?.decoderKvBytesMoved === 0,
  };
}

export function gateChecks(checks, label) {
  const failed = Object.entries(checks)
    .filter(([, pass]) => !pass)
    .map(([name]) => name);
  if (failed.length > 0) {
    throw new Error(`${label}: failed checks: ${failed.join(", ")}`);
  }
}

export function exactTokens(left, right) {
  return JSON.stringify(left) === JSON.stringify(right);
}

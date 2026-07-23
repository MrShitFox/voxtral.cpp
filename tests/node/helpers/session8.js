import { loadEnvironment } from "../config/environment.js";
import { buildLocal, buildRemoteVulkan } from "./build.js";
import { runProcess } from "./exec.js";
import { checkRemoteConnection, syncSources } from "./remote.js";

export const SESSION8_PRODUCTION_ENV = Object.freeze({
  VOXTRAL_ENC_KV_LOGICAL_BATCH: "4",
  VOXTRAL_ENC_KV_PHYSICAL_ROWS: "4",
  VOXTRAL_ENCODER_FINISH_PHYSICAL: "8",
  VOXTRAL_ENCODER_TELEMETRY: "1",
  VOXTRAL_PROFILE: "1",
});

export const SESSION8_GATES = Object.freeze({
  pipelineRtfHard: 0.95,
  pipelineRtfTarget: 0.80,
  decoderKvBytes: 1_000_000_000,
  encoderKvBytes: 270_000_000,
  peakVramBytes: 4_600_000_000,
  deadlineMissRate: 0.001,
  finishMs: 1_000,
  finishTargetMs: 750,
  firstDecoderStepMs: 700,
  firstTokenMs: 1_100,
  firstPartialMs: 1_200,
  coldFirstTokenMs: 1_600,
  coldFirstPartialMs: 1_700,
  postInputDrainMs: 500,
});

export function gate(condition, message) {
  if (!condition) throw new Error(message);
}

export function exactTokens(a, b) {
  return JSON.stringify(a ?? []) === JSON.stringify(b ?? []);
}

export function stageSummary(stage = {}) {
  return {
    count: stage.count ?? 0,
    totalMs: stage.totalMs ?? 0,
    meanMs: stage.meanMs ?? 0,
    p50Ms: stage.p50Ms ?? 0,
    p95Ms: stage.p95Ms ?? 0,
    p99Ms: stage.p99Ms ?? 0,
    maxMs: stage.maxMs ?? 0,
  };
}

export function summarizeRun(r, { includeTokens = true } = {}) {
  const stages = Object.fromEntries(
    Object.entries(r.profileStages ?? {}).map(([name, stage]) => [name, stageSummary(stage)]),
  );
  return {
    state: r.state,
    finishStatus: r.finishStatus,
    evidence: r.evidence,
    audioDurationMs: r.audioDurationMs,
    pipelineRtf: r.pipelineRtf,
    computeHeadroomRatio: r.computeHeadroomRatio,
    realtimeFactorIncludingPacingAndFinish: r.realtimeFactor,
    timing: {
      totalGpuComputeMs: r.totalGpuComputeMs,
      totalPipelineComputeMs: r.totalPipelineComputeMs,
      gpuBusyMeanMsPerFeed: r.gpuBusyMeanMsPerFeed,
      cpuWallMeanMsPerFeed: r.cpuWallMeanMsPerFeed,
      gpuBusyMeasurement: r.gpuBusyMeasurement,
      gpuTimestampQueries: r.gpuTimestampQueries,
    },
    backlog: {
      p50Ms: r.backlogP50Ms,
      p95Ms: r.backlogP95Ms,
      p99Ms: r.backlogP99Ms,
      maxMs: r.backlogMaxMs,
      finalMs: r.finalBacklogMs,
      slopeMsPerSec: r.backlogGrowthSlopeMsPerSec,
      deadlineMisses: r.deadlineMisses,
      deadlineMissRate: r.deadlineMissRate,
    },
    encoderBacklog: r.encoderBacklog,
    adapterBacklog: r.adapterBacklog,
    decoderBacklog: r.decoderBacklog,
    latency: {
      modelLoadMs: r.modelLoadMs,
      warmupMs: r.warmupMs,
      firstDecoderStepMs: r.firstDecoderStepMs,
      firstTokenMs: r.firstTokenMs,
      firstVisibleTextMs: r.firstVisibleTextMs,
      warmModelFirstDecoderStepMs: r.warmModelFirstDecoderStepMs,
      warmModelFirstTokenMs: r.warmModelFirstTokenMs,
      warmModelFirstVisibleTextMs: r.warmModelFirstVisibleTextMs,
      coldFirstDecoderStepMs: r.coldFirstDecoderStepMs,
      coldFirstTokenMs: r.coldFirstTokenMs,
      coldFirstVisibleTextMs: r.coldFirstVisibleTextMs,
      finishMs: r.finishLatencyMs,
      finishFrontendMs: r.finishFrontendMs,
      finishEncoderMs: r.finishEncoderMs,
      finishDecoderMs: r.finishDecoderMs,
      postInputDrainMs: r.postInputDrainMs ?? 0,
    },
    memory: {
      decoderKvBytes: r.decoderKvAllocatedBytes,
      encoderKvBytes: r.encoderKvAllocatedBytes,
      kvF16Bytes: r.kvF16Bytes,
      temporaryF32KvBytes: r.temporaryF32KvBytes,
      modelLoadedVramBytes: r.modelLoadedVramBytes,
      streamIdleVramBytes: r.streamIdleVramBytes,
      peakVramBytes: r.peakVramBytes,
      afterFinishVramBytes: r.afterFinishVramBytes,
      afterDestroyVramBytes: r.afterDestroyVramBytes,
      childExitVramBytes: r.finalVramBytes,
      modelLoadedRssKiB: r.modelLoadedRssKiB,
      streamIdleRssKiB: r.streamIdleRssKiB,
      afterFinishRssKiB: r.afterFinishRssKiB,
      afterDestroyRssKiB: r.afterDestroyRssKiB,
      peakRssKiB: r.peakRssKiB,
      pcmPeakRetainedSamples: r.pcmPeakRetainedSamples,
      encoderOutputAccumulatedBytes: r.encoderOutputAccumulatedBytes,
    },
    work: {
      encoderFrames: r.encoderFrames,
      adapterGroups: r.adapterGroupsCommitted,
      decoderSteps: r.decoderSteps,
      tokensBeforeFinish: r.tokensBeforeFinish,
      tokensFlushedAtFinish: r.tokensFlushedAtFinish,
      encoderFramesAtFinish: r.encoderFramesComputedDuringFinish,
      encoderGraphBuildCount: r.encoderGraphBuildCount,
      adapterGraphBuildCount: r.adapterGraphBuildCount,
      decoderGraphBuildCount: r.decoderGraphBuildCount,
      encoderAllocations: r.encoderAllocations,
      adapterAllocations: r.adapterAllocations,
      decoderAllocations: r.decoderAllocations,
      steadyEncoderGraphBuildCount: r.steadyEncoderGraphBuildCount,
      steadyAdapterGraphBuildCount: r.steadyAdapterGraphBuildCount,
      steadyDecoderGraphBuildCount: r.steadyDecoderGraphBuildCount,
      steadyEncoderAllocations: r.steadyEncoderAllocations,
      steadyAdapterAllocations: r.steadyAdapterAllocations,
      steadyDecoderAllocations: r.steadyDecoderAllocations,
      allocationCounterUnit: r.allocationCounterUnit,
      individualTensorAllocationsInstrumented: r.individualTensorAllocationsInstrumented,
      backendSyncCount: r.backendSyncCount,
      commandSubmitCount: r.commandSubmitCount,
    },
    decoderKvRing: {
      capacity: r.decoderKvCapacity,
      used: r.decoderKvUsed,
      wraps: r.decoderKvWraps,
      evictions: r.decoderKvEvictions,
      bytesMoved: r.decoderKvBytesMoved,
      fullBufferMoves: r.decoderKvFullBufferMoves,
      oldestAbsolutePosition: r.decoderOldestAbsolutePosition,
      nextAbsolutePosition: r.decoderNextAbsolutePosition,
      firstWrapAbsolutePosition: r.decoderFirstWrapAbsolutePosition,
      firstWrapAudioMs: r.decoderFirstWrapAudioMs,
      preWrapP99Ms: r.decoderPreWrapP99Ms,
      wrapStepMs: r.decoderWrapStepMs,
      postWrapP99Ms: r.decoderPostWrapP99Ms,
    },
    transfer: {
      encoderOutputD2hBytes: r.encoderOutputD2hBytes,
      adapterInputD2hBytes: r.adapterInputD2hBytes,
      adapterOutputD2hBytes: r.adapterOutputD2hBytes,
      logitsD2hBytes: r.logitsD2hBytes,
      tokenIdD2hBytes: r.tokenIdD2hBytes,
    },
    eventsDropped: r.eventsDropped,
    nTokens: r.tokens?.length ?? 0,
    tokens: includeTokens ? r.tokens : undefined,
    transcript: r.text,
    stages,
  };
}

export function gateDeviceResident(r, label = "run") {
  gate(r.state === "completed" && r.finishStatus === "ok",
    `${label}: state=${r.state} finish=${r.finishStatus}`);
  gate(r.evidence?.vulkanEnabled === true && r.evidence?.rx6600Detected === true,
    `${label}: missing Vulkan/RX 6600 runtime evidence`);
  gate(r.evidence?.cpuOnlyFallbackDetected !== true, `${label}: CPU-only fallback detected`);
  gate(r.encoderOutputD2hBytes === 0, `${label}: encoder output D2H=${r.encoderOutputD2hBytes}`);
  gate(r.adapterInputD2hBytes === 0, `${label}: adapter input D2H=${r.adapterInputD2hBytes}`);
  gate(r.adapterOutputD2hBytes === 0, `${label}: adapter output D2H=${r.adapterOutputD2hBytes}`);
  gate(r.logitsD2hBytes === 0, `${label}: logits D2H=${r.logitsD2hBytes}`);
  gate(r.tokenIdD2hBytes === 4 * r.decoderSteps,
    `${label}: token readback=${r.tokenIdD2hBytes}, expected ${4 * r.decoderSteps}`);
  gate(r.encoderOutputAccumulatedBytes === 0,
    `${label}: host encoder accumulation=${r.encoderOutputAccumulatedBytes}`);
  gate(r.pcmPeakRetainedSamples <= 16_000,
    `${label}: retained PCM peak=${r.pcmPeakRetainedSamples} samples`);
  gate(r.eventsDropped === 0, `${label}: events dropped=${r.eventsDropped}`);
}

export function gateKvMemory(r, label = "run") {
  gate(r.decoderKvAllocatedBytes <= SESSION8_GATES.decoderKvBytes,
    `${label}: decoder KV ${r.decoderKvAllocatedBytes} > ${SESSION8_GATES.decoderKvBytes}`);
  gate(r.encoderKvAllocatedBytes <= SESSION8_GATES.encoderKvBytes,
    `${label}: encoder KV ${r.encoderKvAllocatedBytes} > ${SESSION8_GATES.encoderKvBytes}`);
  gate(r.decoderKvElementSize === 2 && r.encoderKvElementSize === 2,
    `${label}: KV is not FP16 (decoder=${r.decoderKvElementSize}, encoder=${r.encoderKvElementSize})`);
  gate(r.temporaryF32KvBytes === 0,
    `${label}: temporary full-size F32 KV=${r.temporaryF32KvBytes}`);
  gate(r.kvF16Bytes === r.decoderKvAllocatedBytes + r.encoderKvAllocatedBytes,
    `${label}: FP16 KV accounting ${r.kvF16Bytes} != decoder+encoder ${r.decoderKvAllocatedBytes + r.encoderKvAllocatedBytes}`);
}

export function gateRing(r, label = "run", { requireWrap = false } = {}) {
  gate(r.decoderKvBytesMoved === 0,
    `${label}: rollover moved ${r.decoderKvBytesMoved} bytes`);
  gate(r.decoderKvFullBufferMoves === 0,
    `${label}: full-buffer moves=${r.decoderKvFullBufferMoves}`);
  if (requireWrap) {
    gate(r.decoderKvWraps > 0, `${label}: decoder KV never wrapped`);
    gate(r.decoderKvEvictions > 0, `${label}: decoder KV never evicted`);
    gate(r.decoderFirstWrapAbsolutePosition >= r.decoderKvCapacity,
      `${label}: invalid first wrap position ${r.decoderFirstWrapAbsolutePosition}`);
    gate(r.decoderPreWrapP99Ms > 0 && r.decoderWrapStepMs > 0 && r.decoderPostWrapP99Ms > 0,
      `${label}: incomplete wrap latency telemetry`);
    gate(r.decoderWrapStepMs <= 2 * r.decoderPreWrapP99Ms,
      `${label}: wrap ${r.decoderWrapStepMs}ms > 2x pre-wrap p99 ${r.decoderPreWrapP99Ms}ms`);
    gate(r.decoderPostWrapP99Ms <= 2 * r.decoderPreWrapP99Ms,
      `${label}: post-wrap p99 ${r.decoderPostWrapP99Ms}ms > 2x pre-wrap p99 ${r.decoderPreWrapP99Ms}ms`);
  }
}

export function gateSustained(r, label = "sustained", { requireWrap = true } = {}) {
  gateDeviceResident(r, label);
  gateKvMemory(r, label);
  gateRing(r, label, { requireWrap });
  gate(r.profileEnabled === true, `${label}: profiler disabled`);
  gate(r.pipelineRtf < SESSION8_GATES.pipelineRtfHard,
    `${label}: pipeline RTF ${r.pipelineRtf} >= ${SESSION8_GATES.pipelineRtfHard}`);
  gate(r.finalBacklogMs === 0, `${label}: final backlog ${r.finalBacklogMs} != 0`);
  gate(r.backlogGrowthSlopeMsPerSec <= 0,
    `${label}: backlog slope ${r.backlogGrowthSlopeMsPerSec} > 0`);
  for (const [stage, metrics] of [["encoder", r.encoderBacklog], ["adapter", r.adapterBacklog], ["decoder", r.decoderBacklog]]) {
    gate(metrics?.finalMs === 0, `${label}: ${stage} final backlog ${metrics?.finalMs} != 0`);
    gate(metrics?.slopeMsPerSec <= 0,
      `${label}: ${stage} backlog slope ${metrics?.slopeMsPerSec} > 0`);
    gate(metrics?.deadlineMissRate < SESSION8_GATES.deadlineMissRate,
      `${label}: ${stage} deadline miss rate ${metrics?.deadlineMissRate} >= ${SESSION8_GATES.deadlineMissRate}`);
  }
  gate(r.deadlineMissRate < SESSION8_GATES.deadlineMissRate,
    `${label}: deadline miss rate ${r.deadlineMissRate} >= ${SESSION8_GATES.deadlineMissRate}`);
  gate(r.finishLatencyMs < SESSION8_GATES.finishMs,
    `${label}: finish ${r.finishLatencyMs}ms >= ${SESSION8_GATES.finishMs}ms`);
  gate((r.postInputDrainMs ?? 0) < SESSION8_GATES.postInputDrainMs,
    `${label}: post-input drain ${r.postInputDrainMs}ms >= ${SESSION8_GATES.postInputDrainMs}ms`);
  gate(r.tokensFlushedAtFinish <= 18,
    `${label}: finish decoded ${r.tokensFlushedAtFinish} tail tokens > 18`);
  gate(r.encoderFramesComputedDuringFinish <= 72,
    `${label}: finish encoded ${r.encoderFramesComputedDuringFinish} frames > 72`);
  gate(r.encoderFramesRecomputed === 0, `${label}: encoder replay=${r.encoderFramesRecomputed}`);
  gate(r.steadyEncoderGraphBuildCount <= 2 && r.steadyEncoderAllocations <= 2,
    `${label}: steady encoder graph growth builds=${r.steadyEncoderGraphBuildCount} allocs=${r.steadyEncoderAllocations}`);
  gate(r.steadyAdapterGraphBuildCount === 0 && r.steadyAdapterAllocations === 0,
    `${label}: steady adapter graph growth builds=${r.steadyAdapterGraphBuildCount} allocs=${r.steadyAdapterAllocations}`);
  // One prefill graph and one bounded masked-to-unmasked transition when the
  // decoder ring first becomes full. The count must remain independent of the
  // number of subsequent decoder steps and wraps.
  gate(r.steadyDecoderGraphBuildCount <= 2 && r.steadyDecoderAllocations <= 2,
    `${label}: steady decoder graph growth builds=${r.steadyDecoderGraphBuildCount} allocs=${r.steadyDecoderAllocations}`);
  // The terminal 72-frame suffix is allowed at most ten P8 encoder passes
  // (alignment can add one partial block), each with one bounded output-ring
  // copy. This is finish work, not growth proportional to utterance duration
  // or decoder-step count.
  gate(r.encoderGraphBuildCount - r.steadyEncoderGraphBuildCount <= 20 &&
       r.encoderAllocations - r.steadyEncoderAllocations <= 20,
    `${label}: unbounded terminal encoder work builds=${r.encoderGraphBuildCount - r.steadyEncoderGraphBuildCount} allocs=${r.encoderAllocations - r.steadyEncoderAllocations}`);
  gate(r.adapterGraphBuildCount === r.steadyAdapterGraphBuildCount &&
       r.adapterAllocations === r.steadyAdapterAllocations,
    `${label}: terminal adapter graph/allocation growth`);
  gate(r.decoderGraphBuildCount === r.steadyDecoderGraphBuildCount &&
       r.decoderAllocations === r.steadyDecoderAllocations,
    `${label}: terminal decoder graph/allocation growth`);
  gate(Number.isFinite(r.peakVramBytes) && r.peakVramBytes < SESSION8_GATES.peakVramBytes,
    `${label}: peak VRAM ${r.peakVramBytes} >= ${SESSION8_GATES.peakVramBytes}`);
  if (r.streamIdleRssKiB > 0 && r.afterFinishRssKiB > 0) {
    gate(r.afterFinishRssKiB <= r.streamIdleRssKiB + 64 * 1024,
      `${label}: RSS grew from ${r.streamIdleRssKiB} to ${r.afterFinishRssKiB} KiB`);
  }
}

export function gateLatency(r, label = "latency") {
  gate(r.firstDecoderStepMs > 0 && r.firstDecoderStepMs < SESSION8_GATES.firstDecoderStepMs,
    `${label}: first decoder ${r.firstDecoderStepMs}ms >= ${SESSION8_GATES.firstDecoderStepMs}ms`);
  gate(r.firstTokenMs > 0 && r.firstTokenMs < SESSION8_GATES.firstTokenMs,
    `${label}: first non-special token ${r.firstTokenMs}ms >= ${SESSION8_GATES.firstTokenMs}ms`);
  gate(r.firstVisibleTextMs > 0 && r.firstVisibleTextMs < SESSION8_GATES.firstPartialMs,
    `${label}: first visible partial ${r.firstVisibleTextMs}ms >= ${SESSION8_GATES.firstPartialMs}ms`);
  gate(r.coldFirstTokenMs > 0 && r.coldFirstTokenMs < SESSION8_GATES.coldFirstTokenMs,
    `${label}: cold first token ${r.coldFirstTokenMs}ms >= ${SESSION8_GATES.coldFirstTokenMs}ms`);
  gate(r.coldFirstVisibleTextMs > 0 && r.coldFirstVisibleTextMs < SESSION8_GATES.coldFirstPartialMs,
    `${label}: cold first partial ${r.coldFirstVisibleTextMs}ms >= ${SESSION8_GATES.coldFirstPartialMs}ms`);
}

export async function prepareSession8(summary, { config = loadEnvironment(), localChecks = true } = {}) {
  summary.steps ??= [];
  const add = (name, result) => summary.steps.push({ name, exitCode: result.exitCode ?? 0, wallMs: result.wallMs ?? 0 });
  if (localChecks) {
    const unit = await runProcess("npm", ["run", "test:unit"], {
      cwd: new URL("..", import.meta.url).pathname,
      timeoutMs: 600_000,
    });
    add("node-unit", unit);
    const build = await buildLocal({ config });
    add("local-relwithdebinfo", build);
    const ctest = await runProcess("ctest", ["--test-dir", config.localBuild, "--output-on-failure"], {
      cwd: config.localRepo,
      timeoutMs: 600_000,
    });
    add("ctest", ctest);
  }
  await checkRemoteConnection({ config });
  summary.steps.push({ name: "gpu-connection", exitCode: 0 });
  const sync = await syncSources({ config });
  add("gpu-sync", sync);
  const build = await buildRemoteVulkan({ config });
  add("gpu-relwithdebinfo", build);
  return config;
}

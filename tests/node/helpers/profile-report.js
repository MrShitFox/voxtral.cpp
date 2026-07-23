// Turn one runStreamSession() result into a stage table + derived diagnostics.
// Pure transformation over the JSON the stream harness already emits; no I/O.

const r2 = (x) => Math.round((Number(x) || 0) * 100) / 100;
const r4 = (x) => Math.round((Number(x) || 0) * 10000) / 10000;

// GPU-facing stages whose execute wall time sums to the GPU-busy estimate.
const GPU_EXECUTE_STAGES = new Set([
  "mel_compute",
  "encoder_graph_execute",
  "adapter_graph_execute",
  "decoder_prefill_graph_execute",
  "decoder_step_graph_execute",
]);

export function aggregateProfile(run) {
  const stages = run.profileStages || {};
  const pipelineMs = run.totalPipelineComputeMs || 0;
  const gpuMs = run.totalGpuComputeMs || 0;

  const rows = Object.entries(stages).map(([stage, p]) => ({
    stage,
    count: p.count,
    totalMs: r2(p.totalMs),
    meanMs: r4(p.meanMs),
    p50Ms: r4(p.p50Ms),
    p95Ms: r4(p.p95Ms),
    p99Ms: r4(p.p99Ms),
    maxMs: r2(p.maxMs),
    pctGpu: gpuMs > 0 && GPU_EXECUTE_STAGES.has(stage) ? r2((100 * p.totalMs) / gpuMs) : 0,
    isGpu: GPU_EXECUTE_STAGES.has(stage),
  }));

  const g = run.gpuTelemetry || null;
  const derived = {
    audioDurationMs: r2(run.audioDurationMs),
    wallDurationMs: r2(run.wallDurationMs),
    paced: !!run.pacedRealtime,
    pipelineRtf: r4(run.pipelineRtf),
    wallRtf: r4(run.realtimeFactor),
    computeHeadroomRatio: r4(run.computeHeadroomRatio),
    totalGpuComputeMs: r2(gpuMs),
    totalPipelineComputeMs: r2(pipelineMs),
    finishLatencyMs: r2(run.finishLatencyMs),
    decoderSteps: run.decoderSteps,
    decoderTokensEmitted: run.decoderTokensEmitted,
    dataFeeds: run.dataFeeds,
    // submission / synchronization structure
    commandSubmitCount: run.commandSubmitCount,
    backendSyncCount: run.backendSyncCount,
    submitToSyncRatio: run.backendSyncCount > 0 ? r4(run.commandSubmitCount / run.backendSyncCount) : null,
    tensorSetCount: run.tensorSetCount,
    tensorGetCount: run.tensorGetCount,
    adapterInputD2hBytes: run.adapterInputD2hBytes,
    // steady-state graph reuse (must stay bounded)
    steadyEncoderGraphBuildCount: run.steadyEncoderGraphBuildCount,
    steadyAdapterGraphBuildCount: run.steadyAdapterGraphBuildCount,
    steadyDecoderGraphBuildCount: run.steadyDecoderGraphBuildCount,
    steadyEncoderAllocations: run.steadyEncoderAllocations,
    steadyAdapterAllocations: run.steadyAdapterAllocations,
    steadyDecoderAllocations: run.steadyDecoderAllocations,
    // memory
    peakVramBytes: run.peakVramBytes,
    peakRssKiB: run.peakRssKiB,
    kvF16Bytes: run.kvF16Bytes,
    temporaryF32KvBytes: run.temporaryF32KvBytes,
    // decoder throughput
    decoderStepMeanMs: r4(stages.decoder_step_graph_execute?.meanMs),
    decoderStepP99Ms: r4(stages.decoder_step_graph_execute?.p99Ms),
    encoderExecMeanMs: r4(stages.encoder_graph_execute?.meanMs),
    encoderExecP99Ms: r4(stages.encoder_graph_execute?.p99Ms),
  };

  if (g) {
    derived.gpu = {
      busyMean: g.busyMean,
      busyMax: g.busyMax,
      idleFraction: r4(1 - (g.busyMean || 0) / 100),
      memBusyMean: g.memBusyMean,
      memBusyMax: g.memBusyMax,
      sclkMean: g.sclkMean,
      sclkMax: g.sclkMax,
      fracTopSclk: g.fracTopSclk,
      clockUtil: g.sclkMax > 0 ? r4((g.sclkMean || 0) / g.sclkMax) : null,
      mclkMean: g.mclkMean,
      mclkMax: g.mclkMax,
      fracTopMclk: g.fracTopMclk,
      powerMeanW: g.powerMeanW,
      powerMaxW: g.powerMaxW,
      tempMaxC: g.tempMaxC,
      samples: g.samples,
    };
  }

  return { rows, derived };
}

/** Compact markdown table of the stage profile, sorted by total time. */
export function renderStageTable(agg) {
  const rows = [...agg.rows].sort((a, b) => b.totalMs - a.totalMs);
  const header = "| stage | count | total ms | mean | p50 | p95 | p99 | max | %gpu |";
  const sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|";
  const body = rows.map((x) =>
    `| ${x.stage} | ${x.count} | ${x.totalMs} | ${x.meanMs} | ${x.p50Ms} | ${x.p95Ms} | ${x.p99Ms} | ${x.maxMs} | ${x.pctGpu || ""} |`,
  );
  return [header, sep, ...body].join("\n");
}

/**
 * Rule-based bottleneck classification from one aggregated run. Requires GPU
 * telemetry (busy%, clocks) to distinguish compute-bound from dispatch-bound;
 * without it, falls back to the submit/sync structure only.
 */
export function classifyBottleneck(derived) {
  const g = derived.gpu;
  const top = [...(derived._rows || [])];
  const evidence = [];
  let klass = "unknown";

  if (!g) {
    evidence.push("no GPU telemetry; classification limited to submit/sync structure");
    klass = derived.submitToSyncRatio === 1 ? "fully-synchronized (telemetry needed to confirm)" : "async-submission";
    return { klass, evidence };
  }

  const busy = g.busyMean ?? 0;
  const clockUtil = g.clockUtil ?? 0;
  const idle = g.idleFraction ?? 1 - busy / 100;

  if (busy >= 85 && clockUtil >= 0.9) {
    klass = "compute-bound";
    evidence.push(`GPU busy ${busy}% with core clock at ${Math.round(clockUtil * 100)}% of max (${g.sclkMean}/${g.sclkMax} MHz)`);
    evidence.push(`GPU idle only ${Math.round(idle * 100)}% despite 1:1 submit:sync — large graphs amortize submission`);
    if ((g.memBusyMean ?? 0) < 50) {
      evidence.push(`memory controller ${g.memBusyMean}% busy — not bandwidth-saturated (single-token decode is latency/occupancy-bound, not BW-bound)`);
    } else {
      evidence.push(`memory controller ${g.memBusyMean}% busy — significant bandwidth pressure`);
    }
  } else if (busy < 70 && (derived.submitToSyncRatio ?? 1) >= 1) {
    klass = "dispatch/synchronization-bound";
    evidence.push(`GPU busy only ${busy}% with 1:1 submit:sync — CPU round-trips leave the GPU idle ${Math.round(idle * 100)}%`);
  } else if (busy >= 70 && clockUtil < 0.9) {
    klass = "clock/DPM-limited";
    evidence.push(`GPU busy ${busy}% but core clock only ${Math.round(clockUtil * 100)}% of max — DPM policy leaves clock headroom`);
  } else {
    klass = "mixed";
    evidence.push(`GPU busy ${busy}%, clock util ${Math.round(clockUtil * 100)}% — no single dominant limiter`);
  }
  return { klass, evidence };
}

export { GPU_EXECUTE_STAGES };

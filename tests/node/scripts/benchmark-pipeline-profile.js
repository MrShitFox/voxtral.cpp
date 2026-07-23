// Stage 3 — full pipeline stage profile of the production incremental pipeline
// (variant C: encoder KV F32 + decoder KV FP16, 4/4).
//
// Runs unpaced/warm on the 2-minute and 4-minute fixtures with the runtime
// profiler and amdgpu telemetry enabled, then emits, per fixture and stage:
// calls, total/mean/p50/p95/p99/max wall time and % of GPU compute, plus the
// submission/synchronization structure, steady-state graph/allocation reuse,
// D2H bytes, KV attribution, GPU busy%/clocks/power/temp, and a rule-based
// bottleneck classification.

import { loadEnvironment } from "../config/environment.js";
import { writeArtifactBundle } from "../helpers/artifacts.js";
import { canonicalWav, profileRun, summarizeRepeats } from "../helpers/benchmark-run.js";
import { renderStageTable, classifyBottleneck } from "../helpers/profile-report.js";

const config = loadEnvironment();
const repeats = Math.max(1, Number(process.env.VOXTRAL_PROFILE_REPEATS || 2));
const expectedTokens = { "2min": 1531, "4min": 3072 };

const fixtures = [
  { name: "2min", audio: canonicalWav(config.remoteFixture2min) },
  { name: "4min", audio: canonicalWav(config.remoteFixture4min) },
];

const summary = {
  startedAt: new Date().toISOString(),
  command: "npm run benchmark:pipeline-profile",
  variant: "production-C (encoder KV F32 + decoder KV FP16, 4/4)",
  repeats,
  fixtures: [],
};
const textArtifacts = {};

try {
  for (const fx of fixtures) {
    const results = [];
    for (let i = 0; i < repeats; i++) {
      const { run, agg } = await profileRun({
        config, label: `profile-${fx.name}-r${i}`, audioPath: fx.audio, paced: false,
      });
      if (run.state !== "completed") throw new Error(`${fx.name} r${i}: state=${run.state}`);
      const want = expectedTokens[fx.name];
      if (want && run.tokens?.length !== want) {
        throw new Error(`${fx.name} r${i}: token count ${run.tokens?.length} != expected ${want} (parity guard)`);
      }
      results.push({ run, agg });
      console.log(`[profile] ${fx.name} r${i}: RTF=${agg.derived.pipelineRtf} enc=${agg.derived.encoderExecMeanMs}ms dec=${agg.derived.decoderStepMeanMs}ms busy=${agg.derived.gpu?.busyMean}% sclk=${agg.derived.gpu?.sclkMean}`);
    }

    // Representative (median-RTF) run for the stage table; medians across repeats
    // for the headline metrics.
    const byRtf = [...results].sort((a, b) => a.agg.derived.pipelineRtf - b.agg.derived.pipelineRtf);
    const rep = byRtf[Math.floor(byRtf.length / 2)];
    const table = renderStageTable(rep.agg);
    const diagnosis = classifyBottleneck(rep.agg.derived);

    textArtifacts[`stage-profile-${fx.name}.txt`] =
      `${fx.name} production-C unpaced warm\n\n${table}\n\nbottleneck: ${diagnosis.klass}\n- ${diagnosis.evidence.join("\n- ")}`;

    summary.fixtures.push({
      name: fx.name,
      audio: fx.audio,
      tokens: rep.run.tokens?.length,
      rtf: summarizeRepeats(results.map((r) => r.agg.derived.pipelineRtf)),
      encoderExecMeanMs: summarizeRepeats(results.map((r) => r.agg.derived.encoderExecMeanMs)),
      decoderStepMeanMs: summarizeRepeats(results.map((r) => r.agg.derived.decoderStepMeanMs)),
      representative: rep.agg.derived,
      stageRows: rep.agg.rows,
      bottleneck: diagnosis,
    });

    console.log(`\n=== ${fx.name} stage profile (representative run) ===\n${table}`);
    console.log(`bottleneck: ${diagnosis.klass}\n  - ${diagnosis.evidence.join("\n  - ")}\n`);
  }
  summary.exitCode = 0;
} catch (error) {
  summary.exitCode = 1;
  summary.error = error.message;
  process.exitCode = 1;
  console.error(`[profile] error: ${error.message}`);
} finally {
  summary.finishedAt = new Date().toISOString();
  const artifact = await writeArtifactBundle({
    config, testName: "pipeline-stage-profile", backend: "Vulkan",
    command: summary.command, result: summary, textArtifacts,
  });
  console.log(`[profile] ${summary.exitCode === 0 ? "PASS" : "FAIL"} summary: ${artifact.directory}`);
}

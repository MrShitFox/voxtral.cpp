import { loadEnvironment } from "../config/environment.js";
import { writeArtifactBundle } from "../helpers/artifacts.js";
import { loadLatestPrecisionMatrix } from "../helpers/precision-cache.js";
import {
  divergenceRegions,
  semanticRisk,
  transcriptMetrics,
} from "../helpers/quality.js";
import { gate } from "../helpers/session8.js";

const config = loadEnvironment();
const summary = {
  startedAt: new Date().toISOString(),
  command: "npm run acceptance:fp16-quality",
};
const textArtifacts = {};

function reviewedFixtureQuality(fixture, variantId, fixtureId) {
  const reference = fixture.variants.A?.representative;
  const candidate = fixture.variants[variantId]?.representative;
  gate(reference && candidate, `${fixtureId}: representative run is missing`);
  gate(Array.isArray(reference.tokenTimeline) &&
       reference.tokenTimeline.length === reference.tokens.length,
  `${fixtureId}: F32 token timeline is incomplete`);
  gate(Array.isArray(candidate.tokenTimeline) &&
       candidate.tokenTimeline.length === candidate.tokens.length,
  `${fixtureId}-${variantId}: token timeline is incomplete`);
  const divergence = divergenceRegions(reference.tokenTimeline, candidate.tokenTimeline);
  const transcript = transcriptMetrics(reference.transcript, candidate.transcript);
  return {
    tokenDistance: divergence.tokenDistance,
    tokenReferenceCount: divergence.tokenReferenceCount,
    tokenCandidateCount: divergence.tokenCandidateCount,
    tokenDivergenceRate: divergence.tokenDivergenceRate,
    firstDivergence: divergence.firstDivergence,
    lastDivergence: divergence.lastDivergence,
    sustainedDesynchronization: divergence.sustainedDesynchronization,
    regions: divergence.regions,
    wer: transcript.wer,
    cer: transcript.cer,
    semanticRisk: semanticRisk(reference.transcript, candidate.transcript, divergence),
  };
}

try {
  const matrix = await loadLatestPrecisionMatrix(config);
  summary.sourceArtifact = matrix.directory;
  summary.fixtures = matrix.result.fixtures;
  summary.variants = {};

  for (const variantId of ["B", "C", "D"]) {
    const aggregate = matrix.result.variants[variantId];
    gate(aggregate, `precision matrix omitted variant ${variantId}`);
    const fixtureQuality = Object.fromEntries(
      Object.entries(matrix.result.fixtureResults).map(([fixtureId, fixture]) => {
        const quality = reviewedFixtureQuality(fixture, variantId, fixtureId);
        gate(Number.isFinite(quality.tokenDivergenceRate) &&
             Number.isFinite(quality.wer.rate) &&
             Number.isFinite(quality.cer.rate),
        `${fixtureId}-${variantId}: non-finite quality metric`);
        gate(quality.regions.every((region) =>
          typeof region.classification === "string" &&
          typeof region.reconverged === "boolean" &&
          Array.isArray(region.referenceIds) &&
          Array.isArray(region.candidateIds)),
        `${fixtureId}-${variantId}: incomplete divergence classification`);
        return [fixtureId, quality];
      }),
    );
    summary.variants[variantId] = {
      aggregate,
      fixtures: fixtureQuality,
      automaticHardGates: {
        noChangedNumber: Object.values(fixtureQuality)
          .every((quality) => !quality.semanticRisk.changedNumbers),
        noChangedNegation: Object.values(fixtureQuality)
          .every((quality) => !quality.semanticRisk.changedNegations),
        noSentenceCountChange: Object.values(fixtureQuality)
          .every((quality) => !quality.semanticRisk.sentenceCountChanged),
        noSustainedDesynchronization: Object.values(fixtureQuality)
          .every((quality) => !quality.semanticRisk.sustainedTokenDesynchronization),
      },
      manualSemanticReviewRequired: Object.values(fixtureQuality)
        .some((quality) => quality.regions.some((region) => region.meaningChanged === null)),
    };
  }

  const passingFp16 = ["B", "C", "D"].filter((id) => {
    const variant = matrix.result.variants[id];
    return variant.numericalQualityPassed &&
      variant.hardQualityPassed &&
      variant.deterministic &&
      variant.chunkPlanIndependent;
  });
  gate(passingFp16.length > 0, "no FP16 variant passed the measured quality gates");
  const selected = matrix.result.productionDecision.selected;
  gate(selected && matrix.result.variants[selected], "precision matrix has no production selection");
  const selectedQuality = matrix.result.variants[selected];
  gate(selectedQuality.numericalQualityPassed && selectedQuality.hardQualityPassed,
    `selected variant ${selected} failed quality gates`);
  gate(selectedQuality.wer <= 0.005, `selected WER ${selectedQuality.wer} > 0.5%`);
  gate(selectedQuality.cer <= 0.0025, `selected CER ${selectedQuality.cer} > 0.25%`);
  gate(selectedQuality.tokenDivergence <= 0.005,
    `selected token divergence ${selectedQuality.tokenDivergence} > 0.5%`);
  summary.passingFp16Variants = passingFp16;
  summary.selected = selected;
  summary.corpusLimitation =
    "The two local fixtures show no material quality regression if all reviewed hard gates pass, but they are not a complete multilingual quality corpus.";
  const divergenceRows = [];
  for (const [variantId, variant] of Object.entries(summary.variants)) {
    for (const [fixtureId, quality] of Object.entries(variant.fixtures)) {
      for (const region of quality.regions) {
        divergenceRows.push(JSON.stringify({
          fixture: fixtureId,
          variant: variantId,
          ...region,
        }));
      }
    }
  }
  textArtifacts["reviewed-divergence-analysis.txt"] =
    divergenceRows.length > 0
      ? `${divergenceRows.join("\n")}\n`
      : "No token divergences.\n";
  summary.exitCode = 0;
} catch (error) {
  summary.exitCode = 1;
  summary.error = error.stack ?? error.message;
  process.exitCode = 1;
} finally {
  summary.finishedAt = new Date().toISOString();
  const artifact = await writeArtifactBundle({
    config,
    testName: "session8.1-fp16-quality",
    backend: "Vulkan/RADV RX 6600",
    command: summary.command,
    result: summary,
    textArtifacts,
  });
  console.log(`[fp16-quality] ${summary.exitCode === 0 ? "PASS" : "FAIL"} summary: ${artifact.directory}`);
  if (summary.error) console.error(`[fp16-quality] error: ${summary.error}`);
}

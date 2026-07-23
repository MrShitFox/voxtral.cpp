import { readFile } from "node:fs/promises";
import path from "node:path";

import { loadEnvironment } from "../config/environment.js";

export function deriveProductionDecision(result) {
  const variants = result.variants ?? {};
  const eligible = (variant) =>
    variant &&
    variant.numericalQualityPassed &&
    variant.hardQualityPassed &&
    variant.deterministic &&
    variant.chunkPlanIndependent &&
    variant.realtimePassed &&
    variant.peakVramBytes < result.rx6600VramTotalBytes;
  const { A, B, C, D } = variants;
  let selected = null;
  if (eligible(B) &&
      (B.tokenDivergence === 0 || B.tokenDivergence < D.tokenDivergence)) selected = "B";
  else if (eligible(C) && C.tokenDivergence <= D.tokenDivergence) selected = "C";
  else if (eligible(D)) selected = "D";
  else if (eligible(A)) selected = "A";
  return {
    selected,
    rule:
      "B only when exact or strictly less divergent than D, then C, D, A; quality, determinism, realtime and RX 6600 fit are mandatory",
    candidates: Object.fromEntries(
      ["A", "B", "C", "D"].map((id) => [id, {
        eligible: eligible(variants[id]),
        result: variants[id]?.result ?? "MISSING",
      }]),
    ),
    derivedFromMeasuredMatrix: true,
  };
}

export async function loadLatestPrecisionMatrix(config = loadEnvironment()) {
  const pointer = path.join(config.artifactDir, "session8.1-latest-precision-matrix.txt");
  let directory;
  try {
    directory = (await readFile(pointer, "utf8")).trim();
  } catch (error) {
    throw new Error(
      `precision matrix artifact is unavailable; run npm run acceptance:precision-matrix first (${error.message})`,
    );
  }
  const resolved = path.resolve(directory);
  const artifactRoot = `${path.resolve(config.artifactDir)}${path.sep}`;
  if (!resolved.startsWith(artifactRoot)) {
    throw new Error(`precision matrix pointer escapes artifact root: ${resolved}`);
  }
  const result = JSON.parse(await readFile(path.join(resolved, "result.json"), "utf8"));
  if (result.exitCode !== 0 || result.schemaVersion !== 1) {
    throw new Error(`latest precision matrix is not a successful schema-v1 artifact: ${resolved}`);
  }
  result.recordedProductionDecision = result.productionDecision;
  result.productionDecision = deriveProductionDecision(result);
  return { directory: resolved, result };
}

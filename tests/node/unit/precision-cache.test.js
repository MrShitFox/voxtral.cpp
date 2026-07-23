import { describe, expect, test } from "vitest";

import { deriveProductionDecision } from "../helpers/precision-cache.js";

function variant(tokenDivergence, overrides = {}) {
  return {
    numericalQualityPassed: true,
    hardQualityPassed: true,
    deterministic: true,
    chunkPlanIndependent: true,
    realtimePassed: true,
    peakVramBytes: 4_000_000_000,
    tokenDivergence,
    result: "PASS",
    ...overrides,
  };
}

function matrix({ a = 0, b, c, d }) {
  return {
    rx6600VramTotalBytes: 8_000_000_000,
    variants: {
      A: variant(a),
      B: variant(b),
      C: variant(c),
      D: variant(d),
    },
  };
}

describe("production precision decision", () => {
  test("prefers exact encoder FP16 candidate B", () => {
    expect(deriveProductionDecision(matrix({ b: 0, c: 0, d: 0 })).selected).toBe("B");
  });

  test("selects exact decoder FP16 C when B only ties divergent D", () => {
    expect(deriveProductionDecision(matrix({
      b: 4 / 1531,
      c: 0,
      d: 4 / 1531,
    })).selected).toBe("C");
  });

  test("selects B when it is strictly less divergent than D", () => {
    expect(deriveProductionDecision(matrix({
      b: 1 / 1000,
      c: 0,
      d: 2 / 1000,
    })).selected).toBe("B");
  });
});

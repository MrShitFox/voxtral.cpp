import { describe, expect, test } from "vitest";

import {
  classifyDivergence,
  divergenceRegions,
  semanticRisk,
  transcriptMetrics,
} from "../helpers/quality.js";

function records(ids, pieces = ids.map(String)) {
  return ids.map((id, index) => ({
    id,
    index,
    piece: pieces[index],
    audioEndSample: index * 1_280,
    timestampMs: index * 80,
  }));
}

describe("quality comparison helpers", () => {
  test("aligns a local substitution and proves reconvergence", () => {
    const result = divergenceRegions(
      records([1, 2, 3, 4], ["a", " b", " c", " d"]),
      records([1, 9, 3, 4], ["a", " B", " c", " d"]),
    );
    expect(result.tokenDistance).toBe(1);
    expect(result.regions).toHaveLength(1);
    expect(result.regions[0].reconverged).toBe(true);
    expect(result.sustainedDesynchronization).toBe(false);
  });

  test("does not call one final local edit sustained desynchronization", () => {
    const result = divergenceRegions(records([1, 2, 3]), records([1, 2, 9]));
    expect(result.regions[0].reconverged).toBe(false);
    expect(result.sustainedDesynchronization).toBe(false);
  });

  test("detects insertion, number and negation risks", () => {
    expect(classifyDivergence("", " added", ["insertion"])).toBe("word insertion");
    const divergence = divergenceRegions(records([1]), records([2]));
    const risk = semanticRisk("do not use 12", "do use 13", divergence);
    expect(risk.changedNumbers).toBe(true);
    expect(risk.changedNegations).toBe(true);
  });

  test("classifies a subpiece spelling drift inside a capitalized name as proper noun", () => {
    const result = divergenceRegions(
      records([1, 2, 3, 4, 5], [" Алама", "га", "ру", "да", "."]),
      records([1, 9, 8, 7, 5], [" Алама", "гор", "о", "да", "."]),
    );
    expect(result.regions).toHaveLength(1);
    expect(result.regions[0].classification).toBe("proper noun");
    expect(result.regions[0].meaningChanged).toBeNull();
  });

  test("calculates exact WER and CER denominators", () => {
    const metrics = transcriptMetrics("one two", "one too");
    expect(metrics.wer.edits).toBe(1);
    expect(metrics.wer.referenceUnits).toBe(2);
    expect(metrics.wer.rate).toBe(0.5);
    expect(metrics.cer.edits).toBeGreaterThan(0);
  });
});

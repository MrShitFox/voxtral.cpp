import { describe, expect, test } from "vitest";

import { CHUNK_SAMPLES, createChunkPlan } from "../helpers/chunks.js";

function pcm(samples) {
  const buffer = Buffer.alloc(samples * 2);
  for (let index = 0; index < samples; index += 1) buffer.writeUInt16LE(index & 0xffff, index * 2);
  return buffer;
}

function dataEvents(plan) {
  return plan.events.filter((event) => event.sampleCount > 0);
}

function proveExact(input, plan) {
  expect(dataEvents(plan).reduce((sum, event) => sum + event.sampleCount, 0)).toBe(input.length / 2);
  expect(Buffer.concat(dataEvents(plan).map((event) => event.pcm))).toEqual(input);
}

describe("deterministic PCM chunk plans", () => {
  test.each([
    ["80ms", 1_280],
    ["160ms", 2_560],
    ["320ms", 5_120],
    ["480ms", 7_680],
    ["1000ms", 16_000],
  ])("uses exact %s boundaries", (strategy, expected) => {
    expect(CHUNK_SAMPLES[strategy]).toBe(expected);
    const input = pcm(expected * 2 + 7);
    const plan = createChunkPlan(input, { strategy });
    expect(dataEvents(plan).map((event) => event.sampleCount)).toEqual([expected, expected, 7]);
    proveExact(input, plan);
  });

  test("seeded random is reproducible and usually differs across seeds", () => {
    const input = pcm(50_000);
    const sizes = (seed) => dataEvents(createChunkPlan(input, { strategy: "seeded-random", seed, maxSamples: 2_000 })).map((event) => event.sampleCount);
    expect(sizes(42)).toEqual(sizes(42));
    expect(sizes(42)).not.toEqual(sizes(43));
    proveExact(input, createChunkPlan(input, { strategy: "seeded-random", seed: 42 }));
  });

  test("handles empty, one sample and an odd sample count", () => {
    proveExact(pcm(0), createChunkPlan(pcm(0)));
    proveExact(pcm(1), createChunkPlan(pcm(1), { strategy: "single-sample" }));
    const odd = pcm(7);
    const plan = createChunkPlan(odd, { strategy: "single-sample" });
    expect(dataEvents(plan)).toHaveLength(7);
    proveExact(odd, plan);
  });

  test("zero-length feed events do not change PCM", () => {
    const input = pcm(10);
    const plan = createChunkPlan(input, { strategy: "custom", sampleSizes: [4, 3], zeroLengthAt: [0, 2, 10] });
    expect(plan.events.filter((event) => event.sampleCount === 0)).toHaveLength(3);
    expect(plan.events.findIndex((event) => event.sampleOffset === 2 && event.sampleCount === 0)).toBeGreaterThan(0);
    proveExact(input, plan);
  });

  test("custom plans cycle validated sample counts", () => {
    const plan = createChunkPlan(pcm(12), { strategy: "custom", sampleSizes: [3, 5] });
    expect(dataEvents(plan).map((event) => event.sampleCount)).toEqual([3, 5, 3, 1]);
    expect(() => createChunkPlan(pcm(1), { strategy: "custom", sampleSizes: [0] })).toThrow(/positive integers/u);
    expect(() => createChunkPlan(pcm(1), { strategy: "custom", sampleSizes: [1.5] })).toThrow(/positive integers/u);
  });

  test("rejects byte-truncated PCM16", () => {
    expect(() => createChunkPlan(Buffer.alloc(3))).toThrow(/whole number/u);
  });
});

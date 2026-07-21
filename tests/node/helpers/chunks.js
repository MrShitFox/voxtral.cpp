export const SAMPLE_RATE = 16_000;
export const PCM16_BYTES_PER_SAMPLE = 2;
export const CHUNK_SAMPLES = Object.freeze({
  "80ms": 1_280,
  "160ms": 2_560,
  "320ms": 5_120,
  "480ms": 7_680,
  "1000ms": 16_000,
});

function seededGenerator(seed) {
  if (!Number.isInteger(seed)) throw new Error("seed must be an integer");
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4_294_967_296;
  };
}

function sizeProvider(strategy, options, totalSamples) {
  if (strategy === "full") return () => Math.max(1, totalSamples);
  if (strategy === "single-sample") return () => 1;
  if (CHUNK_SAMPLES[strategy]) return () => CHUNK_SAMPLES[strategy];
  if (strategy === "custom") {
    const sizes = options.sampleSizes;
    if (!Array.isArray(sizes) || sizes.length === 0 || sizes.some((size) => !Number.isInteger(size) || size <= 0)) {
      throw new Error("custom sampleSizes must be a non-empty array of positive integers");
    }
    let index = 0;
    return () => sizes[index++ % sizes.length];
  }
  if (strategy === "seeded-random") {
    const minimum = options.minSamples ?? 1;
    const maximum = options.maxSamples ?? SAMPLE_RATE;
    if (!Number.isInteger(minimum) || !Number.isInteger(maximum) || minimum <= 0 || maximum < minimum) {
      throw new Error("random minSamples/maxSamples must be positive integers with min <= max");
    }
    const random = seededGenerator(options.seed ?? 0);
    return () => minimum + Math.floor(random() * (maximum - minimum + 1));
  }
  throw new Error(`Unknown chunk strategy: ${strategy}`);
}

/** Create deterministic feed events. PCM data is represented by subarray views, never copies. */
export function createChunkPlan(pcm, options = {}) {
  if (!Buffer.isBuffer(pcm)) throw new TypeError("PCM must be a Buffer");
  if (pcm.length % PCM16_BYTES_PER_SAMPLE !== 0) throw new Error("PCM16 buffer must contain a whole number of samples");
  const strategy = options.strategy ?? "full";
  const totalSamples = pcm.length / PCM16_BYTES_PER_SAMPLE;
  const nextSize = sizeProvider(strategy, options, totalSamples);
  const zeroLengthAt = options.zeroLengthAt ?? [];
  if (!Array.isArray(zeroLengthAt) || zeroLengthAt.some((offset) => !Number.isInteger(offset) || offset < 0 || offset > totalSamples)) {
    throw new Error("zeroLengthAt must contain valid sample offsets");
  }
  const zeroOffsets = new Map();
  for (const offset of zeroLengthAt) zeroOffsets.set(offset, (zeroOffsets.get(offset) ?? 0) + 1);

  const events = [];
  let sampleOffset = 0;
  const addZeroEvents = (offset) => {
    const count = zeroOffsets.get(offset) ?? 0;
    for (let index = 0; index < count; index += 1) {
      events.push({ type: "feed", sampleOffset: offset, sampleCount: 0, pcm: pcm.subarray(offset * 2, offset * 2) });
    }
  };

  addZeroEvents(0);
  while (sampleOffset < totalSamples) {
    const nextZeroOffset = [...zeroOffsets.keys()]
      .filter((offset) => offset > sampleOffset)
      .reduce((minimum, offset) => Math.min(minimum, offset), totalSamples);
    const sampleCount = Math.min(nextSize(), totalSamples - sampleOffset, nextZeroOffset - sampleOffset);
    const byteStart = sampleOffset * PCM16_BYTES_PER_SAMPLE;
    const byteEnd = byteStart + sampleCount * PCM16_BYTES_PER_SAMPLE;
    events.push({ type: "feed", sampleOffset, sampleCount, pcm: pcm.subarray(byteStart, byteEnd) });
    sampleOffset += sampleCount;
    addZeroEvents(sampleOffset);
    if (events.length > (options.maxEvents ?? 1_000_000)) {
      throw new Error("Chunk plan exceeds maxEvents; use a coarser strategy for long audio");
    }
  }

  return { strategy, totalSamples, events };
}

import { describe, expect, test } from "vitest";

import { assertStreamingWav, extractPcm16, parseWav } from "../helpers/wav.js";

function chunk(id, data) {
  const padding = data.length & 1;
  const result = Buffer.alloc(8 + data.length + padding);
  result.write(id, 0, 4, "ascii");
  result.writeUInt32LE(data.length, 4);
  data.copy(result, 8);
  return result;
}

function wavFixture({ channels = 1, sampleRate = 16_000, bits = 16, format = 1, samples = [1, -2, 32767], oddChunk = false } = {}) {
  const fmt = Buffer.alloc(16);
  const bytes = bits / 8;
  fmt.writeUInt16LE(format, 0);
  fmt.writeUInt16LE(channels, 2);
  fmt.writeUInt32LE(sampleRate, 4);
  fmt.writeUInt32LE(sampleRate * channels * bytes, 8);
  fmt.writeUInt16LE(channels * bytes, 12);
  fmt.writeUInt16LE(bits, 14);
  const pcm = Buffer.alloc(samples.length * 2);
  samples.forEach((value, index) => pcm.writeInt16LE(value, index * 2));
  const chunks = [chunk("fmt ", fmt)];
  if (oddChunk) chunks.push(chunk("JUNK", Buffer.from([1, 2, 3])));
  chunks.push(chunk("data", pcm));
  const body = Buffer.concat(chunks);
  const header = Buffer.alloc(12);
  header.write("RIFF", 0, 4, "ascii");
  header.writeUInt32LE(body.length + 4, 4);
  header.write("WAVE", 8, 4, "ascii");
  return Buffer.concat([header, body]);
}

describe("RIFF/WAVE parser", () => {
  test("parses PCM16 metadata and returns a payload view", () => {
    const fixture = wavFixture();
    const wav = parseWav(fixture);
    expect(wav).toMatchObject({ sampleRate: 16_000, channels: 1, frameCount: 3, sampleCount: 3 });
    expect(wav.durationSeconds).toBeCloseTo(3 / 16_000);
    expect([...extractPcm16(fixture)]).toEqual([...wav.pcm]);
  });

  test("walks arbitrary chunks and honors odd-size padding", () => {
    expect(parseWav(wavFixture({ oddChunk: true })).frameCount).toBe(3);
  });

  test("diagnoses unsupported and malformed files", () => {
    expect(() => parseWav(Buffer.from("short"))).toThrow(/Truncated WAV/u);
    const notRiff = Buffer.alloc(12);
    expect(() => parseWav(notRiff)).toThrow(/RIFF signature/u);
    expect(() => parseWav(wavFixture({ bits: 8 }))).toThrow(/bit depth/u);
    expect(() => parseWav(wavFixture({ format: 3 }))).toThrow(/encoding/u);
    const truncated = wavFixture().subarray(0, -1);
    expect(() => parseWav(truncated)).toThrow(/Truncated/u);
  });

  test("streaming validation requires 16 kHz mono PCM16", () => {
    expect(assertStreamingWav(parseWav(wavFixture()))).toBeTruthy();
    expect(() => assertStreamingWav(parseWav(wavFixture({ sampleRate: 8_000 })))).toThrow(/16000/u);
    expect(() => assertStreamingWav(parseWav(wavFixture({ channels: 2, samples: [1, 2] })))).toThrow(/mono/u);
  });
});

import { mkdtemp, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, test } from "vitest";

import { inspectWav, prepareStreamingAudio, sha256 } from "../helpers/audio.js";

function wavFixture() {
  const fmt = Buffer.alloc(16);
  fmt.writeUInt16LE(1, 0);
  fmt.writeUInt16LE(1, 2);
  fmt.writeUInt32LE(16_000, 4);
  fmt.writeUInt32LE(32_000, 8);
  fmt.writeUInt16LE(2, 12);
  fmt.writeUInt16LE(16, 14);
  const pcm = Buffer.alloc(6);
  const body = Buffer.alloc(8 + fmt.length + 8 + pcm.length);
  body.write("fmt ", 0, 4, "ascii");
  body.writeUInt32LE(fmt.length, 4);
  fmt.copy(body, 8);
  body.write("data", 24, 4, "ascii");
  body.writeUInt32LE(pcm.length, 28);
  pcm.copy(body, 32);
  const header = Buffer.alloc(12);
  header.write("RIFF", 0, 4, "ascii");
  header.writeUInt32LE(body.length + 4, 4);
  header.write("WAVE", 8, 4, "ascii");
  return Buffer.concat([header, body]);
}

const directories = [];
afterEach(async () => Promise.all(directories.splice(0).map((directory) => rm(directory, { recursive: true, force: true }))));

describe("audio preparation", () => {
  test("keeps a ready source immutable and extracts PCM", async () => {
    const directory = await mkdtemp(path.join(os.tmpdir(), "voxtral-audio-test-"));
    directories.push(directory);
    const filePath = path.join(directory, "ready.wav");
    const fixture = wavFixture();
    await writeFile(filePath, fixture);
    const prepared = await prepareStreamingAudio(filePath);
    expect(prepared.converted).toBe(false);
    expect(prepared.pcm.length).toBe(6);
    expect(prepared.metadata.sourceSha256).toBe(sha256(fixture));
    expect((await inspectWav(filePath)).frameCount).toBe(3);
    await prepared.cleanup();
  });
});

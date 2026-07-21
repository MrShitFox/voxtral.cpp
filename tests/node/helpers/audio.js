import { createHash, randomUUID } from "node:crypto";
import { copyFile, mkdtemp, readFile, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import { runProcess } from "./exec.js";
import { assertStreamingWav, parseWav } from "./wav.js";

export function sha256(buffer) {
  return createHash("sha256").update(buffer).digest("hex");
}

/** Parse a WAV file and return serializable metadata without its PCM payload. */
export async function inspectWav(filePath) {
  const buffer = await readFile(filePath);
  const wav = parseWav(buffer);
  const { pcm: _pcm, ...metadata } = wav;
  return {
    filePath: path.resolve(filePath),
    sha256: sha256(buffer),
    ...metadata,
  };
}

/** Prepare immutable source audio as mono/16 kHz/PCM16, converting in a temporary directory when needed. */
export async function prepareStreamingAudio(inputPath, options = {}) {
  const originalBuffer = await readFile(inputPath);
  const original = parseWav(originalBuffer);
  let temporaryDir = null;
  let preparedPath = path.resolve(inputPath);
  let preparedBuffer = originalBuffer;
  let converted = false;

  try {
    assertStreamingWav(original);
  } catch {
    temporaryDir = await mkdtemp(path.join(os.tmpdir(), "voxtral-audio-"));
    preparedPath = path.join(temporaryDir, `${randomUUID()}.wav`);
    try {
      await runProcess(
        options.ffmpegPath ?? "ffmpeg",
        ["-y", "-i", path.resolve(inputPath), "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", preparedPath],
        { timeoutMs: options.timeoutMs ?? 120_000 },
      );
      preparedBuffer = await readFile(preparedPath);
      converted = true;
    } catch (error) {
      await rm(temporaryDir, { recursive: true, force: true });
      throw error;
    }
  }

  const prepared = assertStreamingWav(parseWav(preparedBuffer));
  let cleaned = false;
  return {
    inputPath: path.resolve(inputPath),
    preparedPath,
    converted,
    pcm: prepared.pcm,
    metadata: {
      sourceSha256: sha256(originalBuffer),
      preparedSha256: sha256(preparedBuffer),
      sampleRate: prepared.sampleRate,
      channels: prepared.channels,
      bitsPerSample: prepared.bitsPerSample,
      frameCount: prepared.frameCount,
      durationSeconds: prepared.durationSeconds,
    },
    async cleanup({ preserveTo } = {}) {
      if (cleaned || !temporaryDir) return;
      if (preserveTo) await copyFile(preparedPath, preserveTo);
      await rm(temporaryDir, { recursive: true, force: true });
      cleaned = true;
    },
  };
}

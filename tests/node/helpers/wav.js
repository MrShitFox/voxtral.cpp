const RIFF_HEADER_BYTES = 12;

function fourCc(buffer, offset) {
  return buffer.toString("ascii", offset, offset + 4);
}

/** Parse a RIFF/WAVE buffer and expose its PCM payload as a zero-copy Buffer view. */
export function parseWav(buffer) {
  if (!Buffer.isBuffer(buffer)) throw new TypeError("WAV input must be a Buffer");
  if (buffer.length < RIFF_HEADER_BYTES) throw new Error("Truncated WAV: missing RIFF header");
  if (fourCc(buffer, 0) !== "RIFF") throw new Error("Unsupported WAV: RIFF signature is missing");
  if (fourCc(buffer, 8) !== "WAVE") throw new Error("Unsupported RIFF file: WAVE signature is missing");

  const riffSize = buffer.readUInt32LE(4);
  const riffEnd = riffSize + 8;
  if (riffEnd > buffer.length) {
    throw new Error(`Truncated WAV: RIFF declares ${riffEnd} bytes, file has ${buffer.length}`);
  }

  let format = null;
  let data = null;
  let offset = RIFF_HEADER_BYTES;
  while (offset + 8 <= riffEnd) {
    const id = fourCc(buffer, offset);
    const size = buffer.readUInt32LE(offset + 4);
    const start = offset + 8;
    const end = start + size;
    if (end > riffEnd) throw new Error(`Truncated WAV chunk ${JSON.stringify(id)} at byte ${offset}`);

    if (id === "fmt " && format === null) {
      if (size < 16) throw new Error("Invalid WAV fmt chunk: expected at least 16 bytes");
      format = {
        audioFormat: buffer.readUInt16LE(start),
        channels: buffer.readUInt16LE(start + 2),
        sampleRate: buffer.readUInt32LE(start + 4),
        byteRate: buffer.readUInt32LE(start + 8),
        blockAlign: buffer.readUInt16LE(start + 12),
        bitsPerSample: buffer.readUInt16LE(start + 14),
      };
    } else if (id === "data" && data === null) {
      data = buffer.subarray(start, end);
    }
    offset = end + (size & 1);
  }

  if (!format) throw new Error("Invalid WAV: fmt chunk is missing");
  if (!data) throw new Error("Invalid WAV: data chunk is missing");
  if (format.audioFormat !== 1) {
    throw new Error(`Unsupported WAV encoding ${format.audioFormat}: only PCM is supported`);
  }
  if (format.bitsPerSample !== 16) {
    throw new Error(`Unsupported WAV bit depth ${format.bitsPerSample}: only signed PCM16 LE is supported`);
  }
  if (format.channels < 1 || format.sampleRate < 1) throw new Error("Invalid WAV channel count or sample rate");
  const expectedBlockAlign = format.channels * 2;
  if (format.blockAlign !== expectedBlockAlign) {
    throw new Error(`Invalid WAV block alignment ${format.blockAlign}; expected ${expectedBlockAlign}`);
  }
  if (data.length % format.blockAlign !== 0) {
    throw new Error(`Truncated PCM payload: ${data.length} bytes is not frame-aligned`);
  }

  const frameCount = data.length / format.blockAlign;
  return {
    ...format,
    frameCount,
    sampleCount: frameCount * format.channels,
    durationSeconds: frameCount / format.sampleRate,
    pcm: data,
  };
}

/** Reject any WAV that is not the streaming contract: mono 16 kHz PCM16 LE. */
export function assertStreamingWav(wav) {
  if (wav.audioFormat !== 1 || wav.bitsPerSample !== 16 || wav.channels !== 1 || wav.sampleRate !== 16_000) {
    throw new Error(
      `Streaming input must be mono 16000 Hz PCM16 LE; got ${wav.channels} channel(s), ${wav.sampleRate} Hz, ${wav.bitsPerSample}-bit format ${wav.audioFormat}`,
    );
  }
  return wav;
}

/** Extract interleaved signed PCM16 bytes, optionally enforcing the streaming contract. */
export function extractPcm16(buffer, { requireStreamingFormat = false } = {}) {
  const wav = parseWav(buffer);
  if (requireStreamingFormat) assertStreamingWav(wav);
  return wav.pcm;
}

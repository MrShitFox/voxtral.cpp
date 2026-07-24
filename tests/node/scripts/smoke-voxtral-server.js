import crypto from "node:crypto";
import fs from "node:fs/promises";
import net from "node:net";
import { spawn } from "node:child_process";

import { assertStreamingWav, parseWav } from "../helpers/wav.js";

function gate(condition, message) {
  if (!condition) throw new Error(message);
}

function delay(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

function parseArguments(argv) {
  const options = {
    baseUrl: "http://127.0.0.1:8080",
    wav: "samples/8297-275156-0000.wav",
    expectedText: "samples/8297-275156-0000.txt",
    websocat: "websocat",
    skipBatch: false,
    skipRealtime: false,
  };
  for (let index = 0; index < argv.length; index += 2) {
    const name = argv[index];
    const value = argv[index + 1];
    if (!value) throw new Error(`${name} requires a value`);
    if (name === "--base-url") options.baseUrl = value.replace(/\/+$/u, "");
    else if (name === "--wav") options.wav = value;
    else if (name === "--expected-text") options.expectedText = value;
    else if (name === "--websocat") options.websocat = value;
    else if (name === "--skip-batch") options.skipBatch = value === "true";
    else if (name === "--skip-realtime") options.skipRealtime = value === "true";
    else throw new Error(`unknown option: ${name}`);
  }
  const apiKey = process.env.VOXTRAL_SERVER_SMOKE_KEY;
  if (!apiKey) {
    throw new Error("VOXTRAL_SERVER_SMOKE_KEY must be set");
  }
  return { ...options, apiKey };
}

function canonicalTranscript(text) {
  return text
    .trim()
    .toLocaleLowerCase("en-US")
    .replace(/[^\p{L}\p{N}]+/gu, " ")
    .trim();
}

class WebsocatClient {
  constructor({ executable, url, apiKey }) {
    this.events = [];
    this.waiters = [];
    this.stderr = "";
    this.stdoutBuffer = "";
    this.exited = false;
    this.exitPromise = new Promise((resolve, reject) => {
      this.resolveExit = resolve;
      this.rejectExit = reject;
    });
    this.process = spawn(executable, [
      "--text",
      "--binary-prefix=B",
      "--text-prefix=T",
      "--base64",
      `-H=Authorization: Bearer ${apiKey}`,
      url,
    ], { stdio: ["pipe", "pipe", "pipe"] });
    this.process.once("error", (error) => {
      this.rejectAll(error);
      this.rejectExit(error);
    });
    this.process.stdout.on("data", (chunk) => this.consumeStdout(chunk));
    this.process.stderr.on("data", (chunk) => {
      if (this.stderr.length < 16_384) this.stderr += chunk.toString("utf8");
    });
    this.process.once("exit", (code, signal) => {
      this.exited = true;
      const result = { code, signal };
      this.resolveExit(result);
      this.rejectAll(new Error(
        `websocat exited before the expected event: code=${code} signal=${signal}` +
        (this.stderr ? ` stderr=${this.stderr.trim()}` : ""),
      ));
    });
  }

  consumeStdout(chunk) {
    this.stdoutBuffer += chunk.toString("utf8");
    for (;;) {
      const newline = this.stdoutBuffer.indexOf("\n");
      if (newline < 0) return;
      const line = this.stdoutBuffer.slice(0, newline).replace(/\r$/u, "");
      this.stdoutBuffer = this.stdoutBuffer.slice(newline + 1);
      if (!line.startsWith("T")) continue;
      let event;
      try {
        event = JSON.parse(line.slice(1));
      } catch (error) {
        this.rejectAll(new Error(`invalid server event JSON: ${error.message}`));
        continue;
      }
      this.events.push(event);
      for (let index = 0; index < this.waiters.length; index += 1) {
        const waiter = this.waiters[index];
        if (waiter.predicate(event)) {
          this.waiters.splice(index, 1);
          clearTimeout(waiter.timer);
          waiter.resolve(event);
          break;
        }
      }
    }
  }

  rejectAll(error) {
    for (const waiter of this.waiters.splice(0)) {
      clearTimeout(waiter.timer);
      waiter.reject(error);
    }
  }

  waitFor(predicate, timeoutMs = 30_000) {
    const existing = this.events.find(predicate);
    if (existing) return Promise.resolve(existing);
    if (this.exited) {
      return Promise.reject(new Error("websocat already exited"));
    }
    return new Promise((resolve, reject) => {
      const waiter = { predicate, resolve, reject, timer: null };
      waiter.timer = setTimeout(() => {
        const index = this.waiters.indexOf(waiter);
        if (index >= 0) this.waiters.splice(index, 1);
        reject(new Error("timed out waiting for a WebSocket event"));
      }, timeoutMs);
      this.waiters.push(waiter);
    });
  }

  waitForType(type, timeoutMs) {
    return this.waitFor((event) => event.type === type, timeoutMs);
  }

  sendText(value) {
    this.process.stdin.write(`T${JSON.stringify(value)}\n`);
  }

  sendBinary(buffer) {
    this.process.stdin.write(`B${buffer.toString("base64")}\n`);
  }

  async waitForExit(timeoutMs = 10_000) {
    let timer;
    try {
      return await Promise.race([
        this.exitPromise,
        new Promise((_, reject) => {
          timer = setTimeout(
            () => reject(new Error("websocat did not exit")), timeoutMs);
        }),
      ]);
    } finally {
      clearTimeout(timer);
    }
  }

  disconnect() {
    this.process.kill("SIGKILL");
  }

  endInput() {
    this.process.stdin.end();
  }
}

async function unauthenticatedUpgradeStatus(websocketUrl) {
  const url = new URL(websocketUrl);
  gate(url.protocol === "ws:", "smoke harness expects ws://");
  const port = Number(url.port || 80);
  const key = crypto.randomBytes(16).toString("base64");
  return await new Promise((resolve, reject) => {
    const socket = net.createConnection({ host: url.hostname, port });
    let response = "";
    const timer = setTimeout(() => {
      socket.destroy();
      reject(new Error("unauthenticated upgrade timed out"));
    }, 10_000);
    socket.once("error", reject);
    socket.on("data", (chunk) => {
      response += chunk.toString("latin1");
      if (!response.includes("\r\n\r\n")) return;
      clearTimeout(timer);
      socket.destroy();
      const match = response.match(/^HTTP\/1\.1 (\d{3})/u);
      if (!match) reject(new Error(`invalid HTTP upgrade response: ${response}`));
      else resolve(Number(match[1]));
    });
    socket.once("connect", () => {
      socket.write(
        `GET ${url.pathname} HTTP/1.1\r\n` +
        `Host: ${url.host}\r\n` +
        "Upgrade: websocket\r\n" +
        "Connection: Upgrade\r\n" +
        `Sec-WebSocket-Key: ${key}\r\n` +
        "Sec-WebSocket-Version: 13\r\n\r\n",
      );
    });
  });
}

async function connectRealtime(options, websocketUrl) {
  const client = new WebsocatClient({
    executable: options.websocat,
    url: websocketUrl,
    apiKey: options.apiKey,
  });
  client.sendText({
    type: "session.configure",
    audio: { format: "pcm_s16le", sample_rate: 16000, channels: 1 },
    events: { token: true, partial: true },
  });
  await client.waitForType("session.created");
  return client;
}

async function main() {
  const options = parseArguments(process.argv.slice(2));
  const wavBytes = await fs.readFile(options.wav);
  const wav = assertStreamingWav(parseWav(wavBytes));
  const expectedText = (await fs.readFile(options.expectedText, "utf8")).trim();
  const batchUrl = `${options.baseUrl}/v1/audio/transcriptions`;
  const websocketUrl = options.baseUrl
    .replace(/^http:/u, "ws:")
    .replace(/^https:/u, "wss:") + "/v1/realtime/transcription";

  const healthResponse = await fetch(`${options.baseUrl}/health`);
  gate(healthResponse.status === 200, "health must return 200");
  const health = await healthResponse.json();
  gate(health.ready === true && health.busy === false, "health readiness");
  gate(health.capabilities?.sample_rate === 16000, "health capabilities");

  const unauthorizedBatch = await fetch(batchUrl, {
    method: "POST",
    headers: { "Content-Type": "audio/wav" },
    body: Buffer.alloc(0),
  });
  gate(unauthorizedBatch.status === 401, "batch without key must return 401");
  gate(
    await unauthenticatedUpgradeStatus(websocketUrl) === 401,
    "WebSocket upgrade without key must return 401",
  );

  let batch = null;
  let batchWallMs = null;
  if (!options.skipBatch) {
    const batchStarted = performance.now();
    const batchResponse = await fetch(batchUrl, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${options.apiKey}`,
        "Content-Type": "audio/wav",
      },
      body: wavBytes,
    });
    gate(batchResponse.status === 200, `batch returned ${batchResponse.status}`);
    batch = await batchResponse.json();
    batchWallMs = performance.now() - batchStarted;
    gate(batch.object === "audio.transcription", "batch object");
    gate(
      canonicalTranscript(batch.text) === canonicalTranscript(expectedText),
      "batch transcript mismatch",
    );
    gate(batch.duration_ms === 3580, `batch duration ${batch.duration_ms}`);
    gate(batch.processing_ms > 0, "batch processing_ms");
    gate(batch.realtime_factor > 0, "batch realtime_factor");
  }

  let final = null;
  let completed = null;
  let tokenCount = null;
  let partialCount = null;
  let busy = null;
  if (!options.skipRealtime) {
    const realtime = await connectRealtime(options, websocketUrl);
    realtime.sendText({ type: "ping", id: "42" });
    const pong = await realtime.waitForType("pong");
    gate(pong.id === "42", "application pong ID");

    const busyResponse = await fetch(batchUrl, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${options.apiKey}`,
        "Content-Type": "application/octet-stream",
        "X-Audio-Format": "pcm_s16le",
        "X-Sample-Rate": "16000",
        "X-Audio-Channels": "1",
      },
      body: wav.pcm,
    });
    gate(busyResponse.status === 503, "busy batch must return 503");
    gate(busyResponse.headers.get("retry-after") === "1", "Retry-After");
    busy = await busyResponse.json();
    gate(busy.error?.code === "server_busy", "server_busy error code");

    for (let offset = 0; offset < wav.pcm.length; offset += 2560) {
      realtime.sendBinary(wav.pcm.subarray(offset, offset + 2560));
      await delay(80);
    }
    realtime.sendText({ type: "input_audio.end" });
    final = await realtime.waitForType("transcript.final", 60_000);
    completed = await realtime.waitForType("session.completed", 60_000);
    realtime.endInput();
    await realtime.waitForExit();
    tokenCount = realtime.events.filter(
      (event) => event.type === "transcript.token").length;
    partialCount = realtime.events.filter(
      (event) => event.type === "transcript.partial").length;
    gate(tokenCount === 56, `expected 56 realtime tokens, got ${tokenCount}`);
    gate(partialCount > 0, "realtime partial events");
    gate(
      canonicalTranscript(final.text) === canonicalTranscript(expectedText),
      "realtime transcript mismatch",
    );
    gate(completed.audio_duration_ms === 3580, "realtime duration");
    gate(
      Number.isFinite(completed.processing?.realtime_factor),
      "realtime processing metrics",
    );
  }

  const disconnect = await connectRealtime(options, websocketUrl);
  disconnect.sendBinary(wav.pcm.subarray(0, 2560));
  disconnect.disconnect();
  await disconnect.waitForExit();

  // One bounded grace interval lets the worker observe EOF at its next public
  // API boundary. This is not polling and does not start another inference.
  await delay(500);
  const recovered = await connectRealtime(options, websocketUrl);
  recovered.sendText({ type: "session.cancel" });
  await recovered.waitForType("session.cancelled");
  recovered.endInput();
  await recovered.waitForExit();

  const finalHealthResponse = await fetch(`${options.baseUrl}/health`);
  const finalHealth = await finalHealthResponse.json();
  gate(
    finalHealthResponse.status === 200 && finalHealth.busy === false,
    "lease released after disconnect/cancel",
  );

  const summary = {
    ok: true,
    health: { ready: health.ready, apiVersion: health.voxtral_api_version },
    auth: { batchUnauthorized: 401, websocketUnauthorized: 401 },
    batch: batch ? {
      durationMs: batch.duration_ms,
      processingMs: batch.processing_ms,
      wallMs: batchWallMs,
      transcriptSha256: crypto.createHash("sha256").update(batch.text).digest("hex"),
    } : { skipped: true },
    realtime: completed ? {
      tokenCount,
      partialCount,
      audioDurationMs: completed.audio_duration_ms,
      realtimeFactor: completed.processing.realtime_factor,
      transcriptSha256: crypto.createHash("sha256").update(final.text).digest("hex"),
    } : { skipped: true },
    busy: busy ? { status: 503, code: busy.error.code } : { skipped: true },
    disconnectLeaseRecovered: true,
  };
  console.log(JSON.stringify(summary));
}

main().catch((error) => {
  console.error(`[voxtral-server-smoke] ${error.message}`);
  process.exitCode = 1;
});

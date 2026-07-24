"use strict";

const TARGET_SAMPLE_RATE = 16000;
const CHUNK_DURATION_MS = 80;
const CHUNK_SAMPLES = 1280;
const CHUNK_BYTES = 2560;

const BUFFER_WARNING_BYTES = 256 * 1024;
const BUFFER_PAUSE_BYTES = 1024 * 1024;
const LOCAL_QUEUE_HARD_BYTES = 5 * TARGET_SAMPLE_RATE * 2;
const QUEUE_FLUSH_INTERVAL_MS = 20;
const AUDIO_LOG_INTERVAL_MS = 1000;
const CANCEL_CLOSE_TIMEOUT_MS = 1500;

const elements = {
  mainStatus: document.querySelector("#main-status"),
  capabilityError: document.querySelector("#capability-error"),
  localError: document.querySelector("#local-error"),
  websocketUrl: document.querySelector("#websocket-url"),
  receiveTokens: document.querySelector("#receive-tokens"),
  startButton: document.querySelector("#start-button"),
  finishButton: document.querySelector("#finish-button"),
  cancelButton: document.querySelector("#cancel-button"),
  pingButton: document.querySelector("#ping-button"),
  clearButton: document.querySelector("#clear-button"),
  wsState: document.querySelector("#ws-state"),
  sessionState: document.querySelector("#session-state"),
  micState: document.querySelector("#mic-state"),
  terminalStatus: document.querySelector("#terminal-status"),
  sessionId: document.querySelector("#session-id"),
  protocolVersion: document.querySelector("#protocol-version"),
  inputRate: document.querySelector("#input-rate"),
  outputRate: document.querySelector("#output-rate"),
  chunksSent: document.querySelector("#chunks-sent"),
  samplesSent: document.querySelector("#samples-sent"),
  bytesSent: document.querySelector("#bytes-sent"),
  audioDuration: document.querySelector("#audio-duration"),
  wallDuration: document.querySelector("#wall-duration"),
  bufferedAmount: document.querySelector("#buffered-amount"),
  localQueue: document.querySelector("#local-queue"),
  partialCount: document.querySelector("#partial-count"),
  tokenCount: document.querySelector("#token-count"),
  warningCount: document.querySelector("#warning-count"),
  errorCount: document.querySelector("#error-count"),
  lastSequence: document.querySelector("#last-sequence"),
  pingRtt: document.querySelector("#ping-rtt"),
  approxLag: document.querySelector("#approx-lag"),
  partialTranscript: document.querySelector("#partial-transcript"),
  finalTranscript: document.querySelector("#final-transcript"),
  eventTimeline: document.querySelector("#event-timeline"),
  rawLog: document.querySelector("#raw-log"),
};

class StreamingLinearResampler {
  constructor(inputRate, outputRate) {
    if (!(inputRate > 0) || !(outputRate > 0)) {
      throw new Error("Invalid resampler sample rate");
    }
    this.inputRate = inputRate;
    this.outputRate = outputRate;
    this.buffer = new Float32Array(0);
    // Input position in units of 1/outputRate samples. With normal integer
    // AudioContext rates this keeps the fractional phase exact over long runs.
    this.phaseUnits = 0;
  }

  push(input) {
    if (!(input instanceof Float32Array) || input.length === 0) {
      return new Float32Array(0);
    }

    const combined = new Float32Array(this.buffer.length + input.length);
    combined.set(this.buffer);
    combined.set(input, this.buffer.length);
    const output = [];

    // phaseUnits is intentionally carried between callbacks. Resetting it for
    // each AudioWorklet block would add discontinuities and duration drift.
    while (
      this.phaseUnits <
      (combined.length - 1) * this.outputRate
    ) {
      const index = Math.floor(this.phaseUnits / this.outputRate);
      const fraction =
        (this.phaseUnits % this.outputRate) / this.outputRate;
      output.push(
        combined[index] +
        (combined[index + 1] - combined[index]) * fraction,
      );
      this.phaseUnits += this.inputRate;
    }

    const consumed = Math.min(
      Math.floor(this.phaseUnits / this.outputRate),
      combined.length,
    );
    this.buffer = combined.slice(consumed);
    this.phaseUnits -= consumed * this.outputRate;
    return Float32Array.from(output);
  }

  flush() {
    const output = [];
    while (this.phaseUnits < this.buffer.length * this.outputRate) {
      const index = Math.floor(this.phaseUnits / this.outputRate);
      const next = Math.min(index + 1, this.buffer.length - 1);
      const fraction =
        (this.phaseUnits % this.outputRate) / this.outputRate;
      output.push(
        this.buffer[index] +
        (this.buffer[next] - this.buffer[index]) * fraction,
      );
      this.phaseUnits += this.inputRate;
    }
    this.buffer = new Float32Array(0);
    this.phaseUnits = 0;
    return Float32Array.from(output);
  }
}

const state = {
  supported: true,
  socket: null,
  wsState: "DISCONNECTED",
  sessionState: "DISCONNECTED",
  micState: "IDLE",
  terminalStatus: "—",
  sessionCreated: false,
  completed: false,
  cancelRequested: false,
  endSent: false,
  finishRequested: false,
  finishCaptureFlushed: false,
  audioAccepting: false,
  mediaStream: null,
  audioContext: null,
  sourceNode: null,
  workletNode: null,
  silentGain: null,
  workletStopResolve: null,
  resampler: null,
  pendingSamples: new Float32Array(0),
  pcmQueue: [],
  queuedBytes: 0,
  chunksSent: 0,
  samplesSent: 0,
  bytesSent: 0,
  aggregateChunks: 0,
  aggregateBytes: 0,
  aggregateSamples: 0,
  partialCount: 0,
  tokenCount: 0,
  warningCount: 0,
  errorCount: 0,
  lastSequence: null,
  lastServerAudioEndMs: null,
  wallStartedAt: null,
  wallEndedAt: null,
  pingSentAt: new Map(),
  pingCounter: 0,
  lastBufferWarningAt: 0,
  cancelTimer: null,
};

function formatBytes(bytes) {
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KiB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(2)} MiB`;
}

function formatSeconds(milliseconds) {
  return `${(milliseconds / 1000).toFixed(3)} s`;
}

function nowClock() {
  return new Date().toLocaleTimeString(undefined, {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    fractionalSecondDigits: 3,
  });
}

function relativeTime() {
  if (state.wallStartedAt === null) {
    return "—";
  }
  return `+${((performance.now() - state.wallStartedAt) / 1000).toFixed(3)} s`;
}

function rawPayload(payload) {
  if (typeof payload === "string") {
    return payload;
  }
  try {
    return JSON.stringify(payload, null, 2);
  } catch {
    return String(payload);
  }
}

function addRawLog(direction, payload) {
  const entry = document.createElement("div");
  entry.className = `log-entry ${direction.toLowerCase()}`;

  const time = document.createElement("time");
  time.dateTime = new Date().toISOString();
  time.textContent = nowClock();

  const directionLabel = document.createElement("span");
  directionLabel.className = "direction";
  directionLabel.textContent = direction;

  const body = document.createElement("pre");
  body.textContent = rawPayload(payload);

  entry.append(time, directionLabel, body);
  elements.rawLog.append(entry);
  elements.rawLog.scrollTop = elements.rawLog.scrollHeight;
}

function timelineClass(type) {
  if (type === "error" || type.includes("error") || type.includes("overflow")) {
    return "error";
  }
  if (type === "session.warning" || type.includes("warning")) {
    return "warning";
  }
  if (
    type === "session.completed" ||
    type === "session.cancelled" ||
    type === "WebSocket open"
  ) {
    return "success";
  }
  return "";
}

function addTimeline(type, event = {}, summary = "") {
  const row = document.createElement("tr");
  row.className = timelineClass(type);
  const values = [
    nowClock(),
    relativeTime(),
    Number.isFinite(event.sequence) ? String(event.sequence) : "—",
    type,
    Number.isFinite(event.audio_end_ms) ? String(event.audio_end_ms) : "—",
    summary,
  ];
  for (const value of values) {
    const cell = document.createElement("td");
    cell.textContent = value;
    row.append(cell);
  }
  elements.eventTimeline.append(row);
  row.scrollIntoView({ block: "nearest" });
}

function setLocalError(message) {
  elements.localError.textContent = message;
  elements.localError.classList.remove("hidden");
}

function clearLocalError() {
  elements.localError.textContent = "";
  elements.localError.classList.add("hidden");
}

function setSessionState(value) {
  state.sessionState = value;
  updateUi();
}

function updateButtons() {
  const active = [
    "CONNECTING",
    "CONFIGURING",
    "READY",
    "STREAMING",
    "FINISHING",
  ].includes(state.sessionState);
  const socketStillActive =
    state.socket !== null && state.socket.readyState !== WebSocket.CLOSED;
  elements.startButton.disabled =
    !state.supported || active || socketStillActive;
  elements.finishButton.disabled = state.sessionState !== "STREAMING";
  elements.cancelButton.disabled = !active || state.cancelRequested;
  elements.pingButton.disabled =
    !state.sessionCreated ||
    state.sessionState === "FINISHING" ||
    state.socket?.readyState !== WebSocket.OPEN;
  elements.websocketUrl.disabled = active;
  elements.receiveTokens.disabled = active;
}

function updateUi() {
  elements.mainStatus.textContent = state.sessionState;
  elements.mainStatus.className =
    `badge ${state.sessionState.toLowerCase()}`;
  elements.wsState.textContent = state.wsState;
  elements.sessionState.textContent = state.sessionState;
  elements.micState.textContent = state.micState;
  elements.terminalStatus.textContent = state.terminalStatus;
  elements.chunksSent.textContent = String(state.chunksSent);
  elements.samplesSent.textContent = String(state.samplesSent);
  elements.bytesSent.textContent = formatBytes(state.bytesSent);
  elements.audioDuration.textContent =
    formatSeconds((state.samplesSent / TARGET_SAMPLE_RATE) * 1000);
  elements.bufferedAmount.textContent =
    formatBytes(state.socket?.bufferedAmount ?? 0);
  elements.localQueue.textContent =
    `${state.pcmQueue.length} chunks / ${formatBytes(state.queuedBytes)}`;
  elements.partialCount.textContent = String(state.partialCount);
  elements.tokenCount.textContent = String(state.tokenCount);
  elements.warningCount.textContent = String(state.warningCount);
  elements.errorCount.textContent = String(state.errorCount);
  elements.lastSequence.textContent =
    state.lastSequence === null ? "—" : String(state.lastSequence);

  if (state.lastServerAudioEndMs === null) {
    elements.approxLag.textContent = "—";
  } else {
    const sentMs = (state.samplesSent / TARGET_SAMPLE_RATE) * 1000;
    elements.approxLag.textContent =
      `${Math.max(0, sentMs - state.lastServerAudioEndMs).toFixed(0)} ms`;
  }

  if (state.wallStartedAt === null) {
    elements.wallDuration.textContent = "0.000 s";
  } else {
    const end = state.wallEndedAt ?? performance.now();
    elements.wallDuration.textContent =
      formatSeconds(end - state.wallStartedAt);
  }
  updateButtons();
}

function resetSessionData() {
  state.socket = null;
  state.wsState = "DISCONNECTED";
  state.sessionState = "DISCONNECTED";
  state.micState = "IDLE";
  state.terminalStatus = "—";
  state.sessionCreated = false;
  state.completed = false;
  state.cancelRequested = false;
  state.endSent = false;
  state.finishRequested = false;
  state.finishCaptureFlushed = false;
  state.audioAccepting = false;
  state.mediaStream = null;
  state.audioContext = null;
  state.sourceNode = null;
  state.workletNode = null;
  state.silentGain = null;
  state.workletStopResolve = null;
  state.resampler = null;
  state.pendingSamples = new Float32Array(0);
  state.pcmQueue = [];
  state.queuedBytes = 0;
  state.chunksSent = 0;
  state.samplesSent = 0;
  state.bytesSent = 0;
  state.aggregateChunks = 0;
  state.aggregateBytes = 0;
  state.aggregateSamples = 0;
  state.partialCount = 0;
  state.tokenCount = 0;
  state.warningCount = 0;
  state.errorCount = 0;
  state.lastSequence = null;
  state.lastServerAudioEndMs = null;
  state.wallStartedAt = performance.now();
  state.wallEndedAt = null;
  state.pingSentAt.clear();
  state.lastBufferWarningAt = 0;
  elements.sessionId.textContent = "—";
  elements.protocolVersion.textContent = "—";
  elements.inputRate.textContent = "—";
  elements.pingRtt.textContent = "—";
  elements.partialTranscript.textContent = "";
  elements.finalTranscript.textContent = "";
  clearLocalError();
  if (state.cancelTimer !== null) {
    clearTimeout(state.cancelTimer);
    state.cancelTimer = null;
  }
  updateUi();
}

function summarizeEvent(event) {
  switch (event.type) {
    case "session.created":
      return `session=${event.session_id ?? "—"}, protocol=${event.protocol_version ?? "—"}`;
    case "transcript.token":
      return `token_id=${event.token_id ?? "—"} text=${event.text ?? ""}`;
    case "transcript.partial":
    case "transcript.final":
      return event.text ?? "";
    case "session.completed":
      return (
        `audio=${event.audio_duration_ms ?? "—"} ms, ` +
        `RTF=${event.processing?.realtime_factor ?? "—"}, ` +
        `backlog=${event.processing?.backlog_final_ms ?? "—"} ms`
      );
    case "session.warning":
      return `${event.code ?? "warning"}: ${event.message ?? ""}`;
    case "session.cancelled":
      return "server acknowledged cancellation";
    case "error":
      return `${event.code ?? "error"}: ${event.message ?? ""}; fatal=${Boolean(event.fatal)}`;
    case "pong":
    case "ping":
      return `id=${event.id ?? "—"}`;
    case "input_audio.end":
    case "session.cancel":
      return event.type;
    default:
      return rawPayload(event);
  }
}

function sendControl(payload) {
  if (!state.socket || state.socket.readyState !== WebSocket.OPEN) {
    throw new Error("WebSocket is not open for control message");
  }
  state.socket.send(JSON.stringify(payload));
  addRawLog("OUT", payload);
  addTimeline(`OUT ${payload.type}`, payload, summarizeEvent(payload));
}

function floatToPcm16Le(samples) {
  const buffer = new ArrayBuffer(samples.length * 2);
  const view = new DataView(buffer);
  for (let index = 0; index < samples.length; index += 1) {
    const sample = Math.max(-1, Math.min(1, samples[index]));
    const pcm = sample < 0 ? sample * 32768 : sample * 32767;
    view.setInt16(index * 2, pcm, true);
  }
  return buffer;
}

function enqueuePcm(buffer, sampleCount) {
  if (state.queuedBytes + buffer.byteLength > LOCAL_QUEUE_HARD_BYTES) {
    void localFatal(
      `Local audio queue overflow: refusing to silently drop ` +
      `${formatBytes(buffer.byteLength)} after the bounded 5-second PCM queue filled.`,
    );
    return false;
  }
  state.pcmQueue.push({ buffer, sampleCount });
  state.queuedBytes += buffer.byteLength;
  updateUi();
  return true;
}

function recordSentChunk(buffer, sampleCount) {
  state.chunksSent += 1;
  state.samplesSent += sampleCount;
  state.bytesSent += buffer.byteLength;
  state.aggregateChunks += 1;
  state.aggregateSamples += sampleCount;
  state.aggregateBytes += buffer.byteLength;
}

function sendOrQueueChunk(buffer, sampleCount) {
  if (
    !state.socket ||
    state.socket.readyState !== WebSocket.OPEN ||
    state.endSent
  ) {
    void localFatal("WebSocket connection failed while audio was pending.");
    return;
  }

  if (
    state.pcmQueue.length > 0 ||
    state.socket.bufferedAmount >= BUFFER_PAUSE_BYTES
  ) {
    enqueuePcm(buffer, sampleCount);
    return;
  }

  state.socket.send(buffer);
  recordSentChunk(buffer, sampleCount);
  updateUi();
}

function appendResampledSamples(samples) {
  if (samples.length === 0) {
    return;
  }
  const combined = new Float32Array(
    state.pendingSamples.length + samples.length,
  );
  combined.set(state.pendingSamples);
  combined.set(samples, state.pendingSamples.length);

  let offset = 0;
  while (combined.length - offset >= CHUNK_SAMPLES) {
    const chunk = combined.slice(offset, offset + CHUNK_SAMPLES);
    sendOrQueueChunk(floatToPcm16Le(chunk), chunk.length);
    offset += CHUNK_SAMPLES;
    if (state.sessionState === "ERROR") {
      break;
    }
  }
  state.pendingSamples = combined.slice(offset);
}

function handleWorkletMessage(event) {
  if (event.data?.type === "stopped") {
    state.workletStopResolve?.();
    state.workletStopResolve = null;
    return;
  }
  if (event.data?.type !== "samples" || !state.audioAccepting) {
    return;
  }
  try {
    appendResampledSamples(state.resampler.push(event.data.samples));
  } catch (error) {
    void localFatal(`Microphone audio processing failed: ${error.message}`);
  }
}

async function startMicrophone() {
  state.micState = "REQUESTING";
  updateUi();
  addRawLog("LOCAL", "Requesting microphone permission");
  addTimeline("local microphone permission", {}, "permission requested");

  let mediaStream;
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
    });
  } catch (error) {
    const denied =
      error?.name === "NotAllowedError" ||
      error?.name === "PermissionDeniedError";
    throw new Error(
      denied
        ? "Microphone permission denied"
        : `Microphone capture failed: ${error.message}`,
    );
  }

  if (state.cancelRequested || state.sessionState !== "READY") {
    mediaStream.getTracks().forEach((track) => track.stop());
    state.micState = "STOPPED";
    addRawLog("LOCAL", "Microphone start abandoned because the session ended");
    updateUi();
    return;
  }

  const AudioContextClass =
    window.AudioContext || window.webkitAudioContext;
  const context = new AudioContextClass();
  state.mediaStream = mediaStream;
  state.audioContext = context;

  try {
    await context.audioWorklet.addModule("./pcm-worklet.js");
    if (context.state === "suspended") {
      await context.resume();
    }
    if (state.cancelRequested || state.sessionState !== "READY") {
      mediaStream.getTracks().forEach((track) => track.stop());
      await context.close().catch(() => {});
      state.mediaStream = null;
      state.audioContext = null;
      state.micState = "STOPPED";
      addRawLog(
        "LOCAL",
        "AudioWorklet start abandoned because the session ended",
      );
      updateUi();
      return;
    }
    state.resampler = new StreamingLinearResampler(
      context.sampleRate,
      TARGET_SAMPLE_RATE,
    );
    state.sourceNode = context.createMediaStreamSource(mediaStream);
    state.workletNode = new AudioWorkletNode(
      context,
      "voxtral-pcm-processor",
      {
        numberOfInputs: 1,
        numberOfOutputs: 1,
        outputChannelCount: [1],
      },
    );
    state.silentGain = context.createGain();
    state.silentGain.gain.value = 0;
    state.workletNode.port.onmessage = handleWorkletMessage;
    state.sourceNode.connect(state.workletNode);
    state.workletNode.connect(state.silentGain);
    state.silentGain.connect(context.destination);
    state.audioAccepting = true;
  } catch (error) {
    mediaStream.getTracks().forEach((track) => track.stop());
    await context.close().catch(() => {});
    throw new Error(`AudioWorklet failed to load: ${error.message}`);
  }

  state.micState = "CAPTURING";
  const deviceRate = mediaStream.getAudioTracks()[0]?.getSettings()
    .sampleRate;
  elements.inputRate.textContent = Number.isFinite(deviceRate)
    ? `${deviceRate} Hz device / ${context.sampleRate} Hz AudioContext`
    : `${context.sampleRate} Hz AudioContext`;
  setSessionState("STREAMING");
  addRawLog(
    "LOCAL",
    `Microphone started at ${context.sampleRate} Hz; streaming resample to ` +
    `${TARGET_SAMPLE_RATE} Hz mono PCM16LE`,
  );
  addTimeline(
    "local microphone started",
    {},
    `${context.sampleRate} Hz input → ${TARGET_SAMPLE_RATE} Hz output`,
  );
}

async function stopMicrophone(flushAudio) {
  if (!state.mediaStream && !state.audioContext) {
    state.audioAccepting = false;
    state.micState = "STOPPED";
    updateUi();
    return;
  }

  state.micState = "STOPPING";
  updateUi();
  state.sourceNode?.disconnect();
  state.mediaStream?.getTracks().forEach((track) => track.stop());

  if (state.workletNode) {
    const stopped = new Promise((resolve) => {
      let settled = false;
      const finish = () => {
        if (!settled) {
          settled = true;
          resolve();
        }
      };
      state.workletStopResolve = finish;
      setTimeout(finish, 500);
    });
    state.workletNode.port.postMessage({ type: "stop" });
    await stopped;
    state.workletNode.disconnect();
    state.workletNode.port.close();
  }
  state.silentGain?.disconnect();
  state.audioAccepting = false;

  if (flushAudio && state.resampler) {
    appendResampledSamples(state.resampler.flush());
    if (state.pendingSamples.length > 0) {
      const finalSamples = state.pendingSamples;
      state.pendingSamples = new Float32Array(0);
      sendOrQueueChunk(floatToPcm16Le(finalSamples), finalSamples.length);
    }
  } else {
    state.pendingSamples = new Float32Array(0);
  }

  if (state.audioContext && state.audioContext.state !== "closed") {
    await state.audioContext.close().catch(() => {});
  }
  state.mediaStream = null;
  state.audioContext = null;
  state.sourceNode = null;
  state.workletNode = null;
  state.silentGain = null;
  state.resampler = null;
  state.micState = "STOPPED";
  addRawLog("LOCAL", `Microphone stopped; flushAudio=${flushAudio}`);
  addTimeline(
    "local microphone stopped",
    {},
    flushAudio
      ? "resampler and final partial PCM flushed"
      : "audio discarded by cancellation",
  );
  updateUi();
}

function flushPcmQueue() {
  if (!state.socket || state.socket.readyState !== WebSocket.OPEN) {
    return;
  }

  while (
    state.pcmQueue.length > 0 &&
    state.socket.bufferedAmount < BUFFER_PAUSE_BYTES
  ) {
    const chunk = state.pcmQueue.shift();
    state.queuedBytes -= chunk.buffer.byteLength;
    state.socket.send(chunk.buffer);
    recordSentChunk(chunk.buffer, chunk.sampleCount);
  }

  if (
    state.finishRequested &&
    state.finishCaptureFlushed &&
    !state.endSent &&
    state.pcmQueue.length === 0
  ) {
    sendControl({ type: "input_audio.end" });
    state.endSent = true;
    addRawLog(
      "LOCAL",
      "All local PCM was handed to WebSocket before input_audio.end",
    );
  }
}

function monitorBackpressure() {
  flushPcmQueue();
  const buffered = state.socket?.bufferedAmount ?? 0;
  if (
    buffered >= BUFFER_WARNING_BYTES &&
    performance.now() - state.lastBufferWarningAt >= 1000
  ) {
    state.lastBufferWarningAt = performance.now();
    state.warningCount += 1;
    const summary =
      `WebSocket bufferedAmount is ${formatBytes(buffered)}; ` +
      `local queue is ${formatBytes(state.queuedBytes)}`;
    addRawLog("LOCAL", summary);
    addTimeline("local queue warning", {}, summary);
  }
  updateUi();
}

function flushAudioAggregateLog() {
  if (state.aggregateChunks === 0) {
    return;
  }
  const durationMs =
    (state.aggregateSamples / TARGET_SAMPLE_RATE) * 1000;
  addRawLog(
    "OUT",
    `sent ${state.aggregateChunks} audio chunks, ` +
    `${state.aggregateBytes.toLocaleString()} bytes, ` +
    `${durationMs.toFixed(0)} ms audio`,
  );
  state.aggregateChunks = 0;
  state.aggregateBytes = 0;
  state.aggregateSamples = 0;
}

function handleServerEvent(event) {
  const type =
    typeof event.type === "string" ? event.type : "unknown server event";
  addTimeline(type, event, summarizeEvent(event));

  if (Number.isFinite(event.sequence)) {
    state.lastSequence = event.sequence;
  }
  if (Number.isFinite(event.audio_end_ms)) {
    state.lastServerAudioEndMs = event.audio_end_ms;
  }

  switch (type) {
    case "session.created":
      if (state.sessionState !== "CONFIGURING") {
        addRawLog(
          "LOCAL",
          `Unexpected session.created while state=${state.sessionState}`,
        );
      }
      state.sessionCreated = true;
      elements.sessionId.textContent = event.session_id ?? "—";
      elements.protocolVersion.textContent =
        event.protocol_version === undefined
          ? "—"
          : String(event.protocol_version);
      setSessionState("READY");
      void startMicrophone().catch((error) => {
        void localFatal(error.message);
      });
      break;
    case "transcript.token":
      state.tokenCount += 1;
      break;
    case "transcript.partial":
      state.partialCount += 1;
      elements.partialTranscript.textContent =
        typeof event.text === "string" ? event.text : "";
      break;
    case "transcript.final":
      elements.finalTranscript.textContent =
        typeof event.text === "string" ? event.text : "";
      break;
    case "session.completed":
      state.completed = true;
      state.wallEndedAt = performance.now();
      state.terminalStatus = "Clean completion";
      setSessionState("COMPLETED");
      flushAudioAggregateLog();
      state.socket?.close(1000, "session completed");
      break;
    case "session.warning":
      state.warningCount += 1;
      break;
    case "session.cancelled":
      state.wallEndedAt = performance.now();
      state.terminalStatus = "Clean cancellation";
      setSessionState("CANCELLED");
      if (state.cancelTimer !== null) {
        clearTimeout(state.cancelTimer);
        state.cancelTimer = null;
      }
      state.socket?.close(1000, "session cancelled");
      break;
    case "error":
      state.errorCount += 1;
      setLocalError(
        `Server returned ${event.fatal ? "fatal " : ""}error ` +
        `${event.code ?? "unknown"}: ${event.message ?? "no message"}`,
      );
      if (event.fatal) {
        state.wallEndedAt = performance.now();
        state.terminalStatus = "Server fatal error";
        setSessionState("ERROR");
        state.pcmQueue = [];
        state.queuedBytes = 0;
        void stopMicrophone(false);
      }
      break;
    case "pong": {
      const sentAt = state.pingSentAt.get(event.id);
      if (sentAt !== undefined) {
        const rtt = performance.now() - sentAt;
        state.pingSentAt.delete(event.id);
        elements.pingRtt.textContent = `${rtt.toFixed(1)} ms`;
      }
      break;
    }
    default:
      addRawLog(
        "LOCAL",
        `Unknown server event type "${type}" was preserved and ignored safely`,
      );
      break;
  }
  updateUi();
}

function handleSocketMessage(message) {
  if (typeof message.data !== "string") {
    void localFatal("Server sent an unexpected binary message.");
    return;
  }
  addRawLog("IN", message.data);
  let event;
  try {
    event = JSON.parse(message.data);
  } catch (error) {
    void localFatal(`Invalid JSON from server: ${error.message}`);
    return;
  }
  if (!event || typeof event !== "object" || Array.isArray(event)) {
    void localFatal("Invalid JSON from server: expected an event object.");
    return;
  }
  handleServerEvent(event);
}

function handleSocketClose(event) {
  flushAudioAggregateLog();
  state.wsState = "DISCONNECTED";
  addRawLog(
    "LOCAL",
    `WebSocket close: code=${event.code}, reason=${event.reason || "(none)"}, ` +
    `wasClean=${event.wasClean}`,
  );
  addTimeline(
    "WebSocket close",
    {},
    `code=${event.code}, reason=${event.reason || "(none)"}, wasClean=${event.wasClean}`,
  );

  if (state.completed) {
    state.terminalStatus = "Clean completion";
    setSessionState("COMPLETED");
  } else if (state.cancelRequested) {
    state.wallEndedAt ??= performance.now();
    if (state.terminalStatus !== "Clean cancellation") {
      state.terminalStatus = event.wasClean
        ? "Cancelled; clean WebSocket close"
        : "Cancelled; connection closed";
    }
    setSessionState("CANCELLED");
  } else if (state.sessionState !== "ERROR") {
    state.wallEndedAt = performance.now();
    state.terminalStatus = "Connection closed before completion";
    setLocalError(
      `Connection closed before completion: code=${event.code}, ` +
      `reason=${event.reason || "(none)"}, wasClean=${event.wasClean}`,
    );
    setSessionState("ERROR");
  }
  if (state.micState !== "STOPPING") {
    void stopMicrophone(false);
  }
  state.socket = null;
  updateUi();
}

function openSocket(url) {
  return new Promise((resolve, reject) => {
    let settled = false;
    let socket;
    try {
      socket = new WebSocket(url);
    } catch (error) {
      reject(new Error(`WebSocket connection failed: ${error.message}`));
      return;
    }

    state.socket = socket;
    state.wsState = "CONNECTING";
    setSessionState("CONNECTING");
    addRawLog("LOCAL", `WebSocket connecting to ${url}`);
    addTimeline("WebSocket connecting", {}, url);

    socket.binaryType = "arraybuffer";
    socket.addEventListener("open", () => {
      settled = true;
      state.wsState = "OPEN";
      addRawLog("LOCAL", "WebSocket open");
      addTimeline("WebSocket open", {}, url);
      updateUi();
      resolve();
    });
    socket.addEventListener("message", handleSocketMessage);
    socket.addEventListener("error", () => {
      addRawLog("LOCAL", "WebSocket error event");
      addTimeline("WebSocket error", {}, "browser did not expose extra details");
      if (!settled) {
        settled = true;
        reject(new Error("WebSocket connection failed"));
      }
    });
    socket.addEventListener("close", (event) => {
      if (!settled) {
        settled = true;
        reject(
          new Error(
            `WebSocket connection failed: close code=${event.code}, ` +
            `reason=${event.reason || "(none)"}`,
          ),
        );
      }
      handleSocketClose(event);
    });
  });
}

async function startSession() {
  resetSessionData();
  const url = elements.websocketUrl.value.trim();
  if (!/^wss?:\/\//u.test(url)) {
    setLocalError("WebSocket URL must start with ws:// or wss://");
    state.wallEndedAt = performance.now();
    state.terminalStatus = "Invalid WebSocket URL";
    setSessionState("ERROR");
    return;
  }

  try {
    await openSocket(url);
    setSessionState("CONFIGURING");
    sendControl({
      type: "session.configure",
      audio: {
        format: "pcm_s16le",
        sample_rate: TARGET_SAMPLE_RATE,
        channels: 1,
      },
      events: {
        token: elements.receiveTokens.checked,
        partial: true,
      },
    });
  } catch (error) {
    state.wallEndedAt = performance.now();
    state.terminalStatus = "Connection failed";
    setLocalError(error.message);
    setSessionState("ERROR");
  }
}

async function finishSession() {
  if (state.sessionState !== "STREAMING" || state.finishRequested) {
    return;
  }
  state.finishRequested = true;
  state.finishCaptureFlushed = false;
  setSessionState("FINISHING");
  try {
    await stopMicrophone(true);
    state.finishCaptureFlushed = true;
    flushPcmQueue();
  } catch (error) {
    await localFatal(`Failed to finish microphone capture: ${error.message}`);
  }
}

async function cancelSession() {
  if (state.cancelRequested) {
    return;
  }
  state.cancelRequested = true;
  state.finishRequested = false;
  state.finishCaptureFlushed = false;
  state.pcmQueue = [];
  state.queuedBytes = 0;
  state.pendingSamples = new Float32Array(0);
  state.terminalStatus = "Cancellation requested";
  setSessionState("CANCELLED");

  try {
    if (state.socket?.readyState === WebSocket.OPEN && state.sessionCreated) {
      sendControl({ type: "session.cancel" });
    } else {
      state.socket?.close(1000, "cancelled before configuration");
    }
  } catch (error) {
    addRawLog("LOCAL", `Could not send session.cancel: ${error.message}`);
  }
  await stopMicrophone(false);

  state.cancelTimer = setTimeout(() => {
    if (state.socket?.readyState === WebSocket.OPEN) {
      addRawLog(
        "LOCAL",
        "Cancellation close timeout elapsed; closing WebSocket locally",
      );
      state.socket.close(1000, "client cancel timeout");
    }
  }, CANCEL_CLOSE_TIMEOUT_MS);
  updateUi();
}

async function localFatal(message) {
  if (state.sessionState === "ERROR") {
    return;
  }
  setLocalError(message);
  addRawLog("LOCAL", message);
  addTimeline("local fatal error", {}, message);
  state.errorCount += 1;
  state.wallEndedAt = performance.now();
  state.terminalStatus = "Local fatal error";
  setSessionState("ERROR");
  state.audioAccepting = false;
  state.pcmQueue = [];
  state.queuedBytes = 0;
  state.pendingSamples = new Float32Array(0);

  try {
    if (state.socket?.readyState === WebSocket.OPEN && state.sessionCreated) {
      sendControl({ type: "session.cancel" });
    }
  } catch (error) {
    addRawLog("LOCAL", `Could not send session.cancel: ${error.message}`);
  }
  await stopMicrophone(false);
  if (state.socket?.readyState === WebSocket.OPEN) {
    state.socket.close(1011, "local fatal error");
  }
  updateUi();
}

function sendPing() {
  if (
    state.socket?.readyState !== WebSocket.OPEN ||
    !state.sessionCreated ||
    state.sessionState === "FINISHING"
  ) {
    return;
  }
  state.pingCounter += 1;
  const random =
    typeof crypto.randomUUID === "function"
      ? crypto.randomUUID()
      : `${Date.now()}-${state.pingCounter}`;
  const id = `browser-${random}`;
  state.pingSentAt.set(id, performance.now());
  sendControl({ type: "ping", id });
}

function clearLogs() {
  elements.eventTimeline.textContent = "";
  elements.rawLog.textContent = "";
}

function emergencyShutdown() {
  state.audioAccepting = false;
  state.mediaStream?.getTracks().forEach((track) => track.stop());
  state.sourceNode?.disconnect();
  state.workletNode?.disconnect();
  state.silentGain?.disconnect();
  if (state.socket && state.socket.readyState < WebSocket.CLOSING) {
    state.socket.close(1000, "page unloading");
  }
}

function checkCapabilities() {
  const missing = [];
  if (!navigator.mediaDevices?.getUserMedia) {
    missing.push("navigator.mediaDevices.getUserMedia");
  }
  if (!(window.AudioContext || window.webkitAudioContext)) {
    missing.push("AudioContext");
  }
  if (!window.AudioWorkletNode) {
    missing.push("AudioWorkletNode");
  }
  if (!window.WebSocket) {
    missing.push("WebSocket");
  }
  if (!window.isSecureContext) {
    missing.push("secure context (serve from http://127.0.0.1, not file://)");
  }

  if (missing.length > 0) {
    state.supported = false;
    elements.capabilityError.textContent =
      `Unsupported browser/context. Missing: ${missing.join(", ")}.`;
    elements.capabilityError.classList.remove("hidden");
    state.terminalStatus = "Unsupported browser";
    setSessionState("ERROR");
  } else {
    addRawLog("LOCAL", "Browser capability check passed");
  }
}

elements.startButton.addEventListener("click", () => {
  void startSession();
});
elements.finishButton.addEventListener("click", () => {
  void finishSession();
});
elements.cancelButton.addEventListener("click", () => {
  void cancelSession();
});
elements.pingButton.addEventListener("click", sendPing);
elements.clearButton.addEventListener("click", clearLogs);
window.addEventListener("beforeunload", emergencyShutdown);
window.addEventListener("pagehide", emergencyShutdown);

setInterval(monitorBackpressure, QUEUE_FLUSH_INTERVAL_MS);
setInterval(flushAudioAggregateLog, AUDIO_LOG_INTERVAL_MS);
setInterval(updateUi, 100);

elements.outputRate.textContent =
  `${TARGET_SAMPLE_RATE} Hz / mono / PCM16LE / ${CHUNK_DURATION_MS} ms chunks`;
checkCapabilities();
updateUi();

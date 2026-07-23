// DIAGNOSTIC (untracked, not a gate): empirically decompose the warm first-token
// latency on the real 2-minute fixture. Prints the first lexical (non-special)
// token's absolute decoder position and its audio-arrival floor (the earliest
// wall-clock instant its audio can exist under 80 ms real-time pacing), so the
// structural floor of firstTokenMs can be read directly rather than asserted.
import path from "node:path";

import { loadEnvironment } from "../config/environment.js";
import { normalizeFixtureOnGpu, syncFixture } from "../helpers/remote.js";
import { runStreamSession } from "../helpers/stream.js";
import { SESSION8_PRODUCTION_ENV, SESSION8_GATES } from "../helpers/session8.js";

const config = loadEnvironment();
const localFixture = path.join(config.localRepo, "voxTest2min.m4a");
await syncFixture(localFixture, { config });
const fixture = await normalizeFixtureOnGpu({ config });

const run = await runStreamSession({
  config,
  planName: "probe-first-token",
  audioPath: fixture.wavPath,
  realtimeMs: 80,
  warmup: true,
  skipParity: true,
  env: SESSION8_PRODUCTION_ENV,
  timeoutMs: 420_000,
});

const sr = run.sampleRate ?? 16000;
const arrMs = (s) => (s * 1000) / sr; // audioEndSample -> real-time arrival ms
const tokenEvents = (run.events ?? []).filter((e) => e.type === "token");
const firstLexical = tokenEvents.find((e) =>
  !e.special && /[\p{L}\p{N}]/u.test(e.piece ?? ""));

const out = {
  gates: {
    firstDecoderStepOverheadMs: SESSION8_GATES.firstDecoderStepOverheadMs,
    firstTokenOverheadMs: SESSION8_GATES.firstTokenOverheadMs,
    firstPartialOverheadMs: SESSION8_GATES.firstPartialOverheadMs,
  },
  measured: {
    modelLoadMs: run.modelLoadMs,
    contextCreationMs: run.contextCreationMs,
    warmupMs: run.warmupMs,
    streamStartMs: run.streamStartMs,
    firstDecoderStepMs: run.firstDecoderStepMs,
    firstDecoderStepEligibilityMs: run.firstDecoderStepEligibilityMs,
    firstDecoderStepOverheadMs: run.firstDecoderStepOverheadMs,
    firstTokenMs: run.firstTokenMs,
    firstTokenEligibilityMs: run.firstTokenEligibilityMs,
    firstTokenOverheadMs: run.firstTokenOverheadMs,
    firstVisibleTextMs: run.firstVisibleTextMs,
    firstPartialEligibilityMs: run.firstPartialEligibilityMs,
    firstPartialOverheadMs: run.firstPartialOverheadMs,
  },
  firstLexical: firstLexical
    ? {
        decoderPosition: firstLexical.decoderPosition,
        token: firstLexical.token,
        audioEndSample: firstLexical.audioEndSample,
        audioArrivalMs: Math.round(arrMs(firstLexical.audioEndSample)),
        pipelineLatencyMs: Math.round(run.firstTokenMs - arrMs(firstLexical.audioEndSample)),
      }
    : null,
  firstDecoderStepFloor: {
    // position 38 = first real step; its audio floor = (38-31)*80 = 560ms
    audioArrivalMs: tokenEvents.length ? Math.round(arrMs(tokenEvents[0].audioEndSample)) : null,
    position: tokenEvents.length ? tokenEvents[0].decoderPosition : null,
  },
  leadingTokens: tokenEvents.slice(0, 24).map((e) => ({
    pos: e.decoderPosition,
    tok: e.token,
    special: e.special,
    arrivalMs: Math.round(arrMs(e.audioEndSample)),
  })),
  totalTokenEvents: tokenEvents.length,
  transcriptHead: (run.text ?? "").slice(0, 80),
  pipelineRtf: run.pipelineRtf,
  finalBacklogMs: run.finalBacklogMs,
};
console.log("PROBE_RESULT " + JSON.stringify(out));

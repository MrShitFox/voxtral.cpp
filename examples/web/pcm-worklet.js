"use strict";

class VoxtralPcmProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.active = true;
    this.port.onmessage = (event) => {
      if (event.data?.type === "stop") {
        this.active = false;
        this.port.postMessage({ type: "stopped" });
      }
    };
  }

  process(inputs) {
    if (!this.active) {
      return false;
    }
    const input = inputs[0];
    if (!input || input.length === 0 || input[0].length === 0) {
      return true;
    }

    // getUserMedia requests mono. If a browser supplies more channels anyway,
    // the first channel is the explicit mono capture source for this demo.
    const samples = new Float32Array(input[0]);
    this.port.postMessage({ type: "samples", samples }, [samples.buffer]);
    return true;
  }
}

registerProcessor("voxtral-pcm-processor", VoxtralPcmProcessor);

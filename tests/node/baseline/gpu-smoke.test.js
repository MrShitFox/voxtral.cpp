import { describe, expect, test } from "vitest";

import { gpuConfig, runRemote, shellQuote } from "../helpers/remote.js";

const binary = `${gpuConfig.repository}/build-vulkan/voxtral`;
const audio = `${gpuConfig.repository}/samples/8297-275156-0000.wav`;

describe.sequential("RX 6600 Vulkan baseline", () => {
  test("GPU server is reachable and has the model and Vulkan build", async () => {
    const command = [
      "set -e",
      `test -s ${shellQuote(gpuConfig.model)}`,
      `test -x ${shellQuote(binary)}`,
      "vulkaninfo --summary",
    ].join("; ");
    const result = await runRemote(command);

    expect(result.timedOut, result.diagnostics).toBe(false);
    expect(result.status, result.diagnostics).toBe(0);
    expect(result.stdout, result.diagnostics).toMatch(/AMD Radeon RX 6600|RADV NAVI23/i);
  });

  test("short inference returns a transcript through the Vulkan backend", async () => {
    const command = [
      "set -e",
      `cd ${shellQuote(gpuConfig.repository)}`,
      [
        shellQuote(binary),
        "--model",
        shellQuote(gpuConfig.model),
        "--audio",
        shellQuote(audio),
        "--gpu vulkan",
        "--log-level info",
      ].join(" "),
    ].join("; ");
    const result = await runRemote(command);
    const transcript = result.stdout
      .split(/\r?\n/u)
      .map((line) => line.trim())
      .find((line) => line && !line.startsWith("[tokens]"));

    expect(result.timedOut, result.diagnostics).toBe(false);
    expect(result.status, result.diagnostics).toBe(0);
    expect(transcript, result.diagnostics).toBeTruthy();
    expect(transcript, result.diagnostics).not.toBe("[no-transcript]");
    expect(result.stderr, result.diagnostics).toMatch(
      /AMD Radeon RX 6600[\s\S]*backend: VULKAN[\s\S]*runtime_backends: Vulkan\(1 dev\)/i,
    );
    expect(result.stderr, result.diagnostics).not.toMatch(/no GPU backend available, using CPU/i);
  });
});

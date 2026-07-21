import { describe, expect, test } from "vitest";

import { parseInferenceOutput } from "../helpers/inference.js";

describe("batch inference output parser", () => {
  test("extracts transcript and Vulkan evidence without rejecting CPU fallback helper", () => {
    const parsed = parseInferenceOutput({
      exitCode: 0,
      wallMs: 1234,
      stdout: "What are you doing here? He asked.\n[tokens] 1 2\n",
      stderr: [
        "ggml_vulkan: 0 = AMD Radeon RX 6600 (RADV NAVI23)",
        "voxtral_I: backend: VULKAN (CPU fallback 10 threads)",
        "[summary] runtime_backends: Vulkan(1 dev), CPU(1 dev)",
        "[summary] processing_time_ms=1200.25",
      ].join("\n"),
    });
    expect(parsed.transcript).toBe("What are you doing here? He asked.");
    expect(parsed.tokens).toEqual([1, 2]);
    expect(parsed.backend).toBe("Vulkan");
    expect(parsed.evidence).toEqual({ vulkanEnabled: true, rx6600Detected: true, cpuOnlyFallbackDetected: false });
  });

  test("detects complete CPU-only fallback", () => {
    const parsed = parseInferenceOutput({ exitCode: 0, wallMs: 1, stdout: "text\n", stderr: "no GPU backend available, using CPU\n" });
    expect(parsed.backend).toBe("CPU");
    expect(parsed.evidence.cpuOnlyFallbackDetected).toBe(true);
  });
});

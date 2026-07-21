import { describe, expect, test } from "vitest";

import { loadEnvironment } from "../config/environment.js";
import { buildRemoteVulkan } from "../helpers/build.js";
import { runBatchInference } from "../helpers/inference.js";
import { checkRemoteConnection, checkRemoteModel, checkRemoteVulkan } from "../helpers/remote.js";

const enabled = process.env.VOXTRAL_TEST_GPU === "1";

describe.skipIf(!enabled).sequential("RX 6600 Vulkan baseline", () => {
  const config = loadEnvironment();

  test("GPU server, model, Vulkan device and build are ready", async () => {
    const connection = await checkRemoteConnection({ config });
    const model = await checkRemoteModel({ config });
    const vulkan = await checkRemoteVulkan({ config });
    const build = await buildRemoteVulkan({ config });

    expect(connection.stdout).toContain("voxtral-ssh-ok");
    expect(Number(model.stdout.trim())).toBeGreaterThan(1_000_000_000);
    expect(vulkan.stdout).toMatch(/AMD Radeon RX 6600|RADV NAVI23/iu);
    expect(build.verification.stdout).toContain("GGML_VULKAN:BOOL=ON");
    expect(build.binaryPath).toBe(config.remoteBinary);
  }, 1_200_000);

  test("short inference returns transcript and positive Vulkan/RX 6600 evidence", async () => {
    const result = await runBatchInference({ config, testName: "baseline-gpu-smoke", timeoutMs: 240_000 });
    expect(result.exitCode).toBe(0);
    expect(result.transcript).toBeTruthy();
    expect(result.transcript).not.toBe("[no-transcript]");
    expect(result.backend).toBe("Vulkan");
    expect(result.device).toBe("AMD Radeon RX 6600");
    expect(result.evidence.vulkanEnabled).toBe(true);
    expect(result.evidence.rx6600Detected).toBe(true);
    expect(result.evidence.cpuOnlyFallbackDetected).toBe(false);
    expect(result.artifact.directory).toContain(config.artifactDir);
  }, 300_000);
});

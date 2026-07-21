import fs from "node:fs";
import { describe, expect, test } from "vitest";

import { describeEnvironment, loadEnvironment } from "../config/environment.js";
import { runProcess } from "../helpers/exec.js";
import { checkRemoteConnection, checkRemoteModel, checkRemoteVulkan } from "../helpers/remote.js";

const enabled = process.env.VOXTRAL_TEST_REMOTE === "1";

describe.skipIf(!enabled).sequential("acceptance environment", () => {
  const config = loadEnvironment();

  test("configuration and local prerequisites are valid", async () => {
    expect(describeEnvironment(config).gpuPassword).toBe("<redacted>");
    expect(fs.statSync(config.localRepo).isDirectory()).toBe(true);
    expect(fs.statSync(config.localSmokeAudio).isFile()).toBe(true);
    for (const executable of ["node", "npm", "cmake", "ninja", "ccache", "ffmpeg", "ffprobe", "sshpass", "rsync"]) {
      const result = await runProcess("sh", ["-c", `command -v "$1"`, "sh", executable], { timeoutMs: 10_000 });
      expect(result.stdout.trim(), `${executable} must be on PATH`).toBeTruthy();
    }
  });

  test("remote connection, model and RX 6600 Vulkan runtime are available", async () => {
    const connection = await checkRemoteConnection({ config });
    const model = await checkRemoteModel({ config });
    const vulkan = await checkRemoteVulkan({ config });
    expect(connection.exitCode).toBe(0);
    expect(Number(model.stdout.trim())).toBeGreaterThan(1_000_000_000);
    expect(vulkan.stdout).toMatch(/AMD Radeon RX 6600|RADV NAVI23/iu);
  }, 60_000);
});

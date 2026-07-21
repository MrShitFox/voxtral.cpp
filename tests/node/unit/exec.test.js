import process from "node:process";
import { describe, expect, test } from "vitest";

import { ProcessExecutionError, runProcess } from "../helpers/exec.js";

describe("process runner", () => {
  test("captures a nonzero exit with diagnostics", async () => {
    await expect(runProcess(process.execPath, ["-e", "console.log('out'); console.error('err'); process.exit(7)"]))
      .rejects.toMatchObject({ name: "ProcessExecutionError", result: { exitCode: 7, stdout: "out\n", stderr: "err\n" } });
  });

  test("can return nonzero exit status instead of throwing", async () => {
    const result = await runProcess(process.execPath, ["-e", "process.exit(3)"], { rejectOnNonZero: false });
    expect(result.exitCode).toBe(3);
  });

  test("terminates a process after timeout", async () => {
    try {
      await runProcess(process.execPath, ["-e", "setInterval(() => {}, 1000)"], { timeoutMs: 50, killGraceMs: 50 });
      throw new Error("expected timeout");
    } catch (error) {
      expect(error).toBeInstanceOf(ProcessExecutionError);
      expect(error.result.timedOut).toBe(true);
      expect(error.result.signal).toMatch(/SIGTERM|SIGKILL/u);
    }
  });

  test("drains large stdout and stderr without deadlock", async () => {
    const size = 1_200_000;
    const result = await runProcess(process.execPath, ["-e", `process.stdout.write('o'.repeat(${size})); process.stderr.write('e'.repeat(${size}))`]);
    expect(result.stdout).toHaveLength(size);
    expect(result.stderr).toHaveLength(size);
  });

  test("reports signal termination", async () => {
    const result = await runProcess(process.execPath, ["-e", "process.kill(process.pid, 'SIGTERM')"], { rejectOnNonZero: false });
    expect(result.exitCode).toBeNull();
    expect(result.signal).toBe("SIGTERM");
  });

  test("supports streaming callbacks", async () => {
    const chunks = [];
    await runProcess(process.execPath, ["-e", "process.stdout.write('live')"], { onStdout: (chunk) => chunks.push(chunk.toString()) });
    expect(chunks.join("")).toBe("live");
  });
});

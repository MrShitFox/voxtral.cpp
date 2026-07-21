import { mkdtemp, readFile, readdir, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, test } from "vitest";

import { loadEnvironment } from "../config/environment.js";
import { writeArtifactBundle } from "../helpers/artifacts.js";

const directories = [];
afterEach(async () => Promise.all(directories.splice(0).map((directory) => rm(directory, { recursive: true, force: true }))));

describe("artifact collection", () => {
  test("writes the standard bundle and removes password fields", async () => {
    const artifactDir = await mkdtemp(path.join(os.tmpdir(), "voxtral-artifacts-"));
    directories.push(artifactDir);
    const config = loadEnvironment({}, { VOXTRAL_ARTIFACT_DIR: artifactDir, VOXTRAL_GPU_PASSWORD: "artifact-secret" });
    const bundle = await writeArtifactBundle({
      config,
      runId: "unit-run",
      commitSha: "abc123",
      testName: "artifact unit",
      command: "sshpass -e ssh root@gpu true",
      result: { exitCode: 0, wallMs: 12, password: "artifact-secret", stdout: "ok", stderr: "" },
    });
    expect(await readdir(bundle.directory)).toEqual([
      "command.txt", "metadata.json", "result.json", "stderr.log", "stdout.log",
    ]);
    const all = (await Promise.all((await readdir(bundle.directory)).map((name) => readFile(path.join(bundle.directory, name), "utf8")))).join("\n");
    expect(all).not.toContain("artifact-secret");
    expect(all).toContain("<redacted>");
  });
});

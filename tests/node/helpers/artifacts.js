import { randomUUID } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

import { loadEnvironment, redactSecrets } from "../config/environment.js";
import { runProcess } from "./exec.js";

function sanitize(value, secrets) {
  if (typeof value === "string") return redactSecrets(value, secrets);
  if (Array.isArray(value)) return value.map((item) => sanitize(item, secrets));
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value).map(([key, item]) => [
        key,
        key.toLowerCase().includes("password") ? "<redacted>" : sanitize(item, secrets),
      ]),
    );
  }
  return value;
}

async function commitSha(localRepo) {
  try {
    const result = await runProcess("git", ["rev-parse", "HEAD"], { cwd: localRepo, timeoutMs: 10_000 });
    return result.stdout.trim();
  } catch {
    return "unknown";
  }
}

/** Create the standard five-file artifact bundle for one integration run. */
export async function writeArtifactBundle(options) {
  const config = options.config ?? loadEnvironment();
  const timestamp = options.timestamp ?? new Date().toISOString();
  const runId = options.runId ?? `${timestamp.replaceAll(":", "-")}-${randomUUID()}`;
  const directory = path.join(options.artifactDir ?? config.artifactDir, runId);
  await mkdir(directory, { recursive: true });

  // A one-character sshpass value is never present in argv (sshpass -e). Avoid corrupting
  // every numeric diagnostic while still structurally redacting all password fields.
  const secrets = [config.gpuPassword, ...(options.secrets ?? [])].filter((secret) => secret.length > 1);
  const result = sanitize(options.result ?? {}, secrets);
  const metadata = sanitize({
    runId,
    timestamp,
    commitSha: options.commitSha ?? await commitSha(config.localRepo),
    testName: options.testName,
    host: options.host ?? config.gpuHost,
    backend: options.backend ?? result.backend ?? "unknown",
    binaryPath: options.binaryPath ?? config.remoteBinary,
    modelPath: options.modelPath ?? config.remoteModel,
    audio: options.audioMetadata ?? null,
    exitStatus: result.exitCode ?? null,
    wallMs: result.wallMs ?? null,
  }, secrets);
  const command = sanitize(options.command ?? result.commandLine ?? "", secrets);
  const stdout = sanitize(options.stdout ?? result.stdout ?? "", secrets);
  const stderr = sanitize(options.stderr ?? result.stderr ?? "", secrets);

  await Promise.all([
    writeFile(path.join(directory, "metadata.json"), `${JSON.stringify(metadata, null, 2)}\n`),
    writeFile(path.join(directory, "command.txt"), `${command}\n`),
    writeFile(path.join(directory, "stdout.log"), stdout),
    writeFile(path.join(directory, "stderr.log"), stderr),
    writeFile(path.join(directory, "result.json"), `${JSON.stringify(result, null, 2)}\n`),
  ]);
  return { runId, directory, metadata };
}

import fs from "node:fs";
import crypto from "node:crypto";
import path from "node:path";

import { loadEnvironment } from "../config/environment.js";
import { runProcess } from "./exec.js";

const SAFE_REMOTE_REPO = "/root/voxtral.cpp";

async function sha256File(filePath) {
  const hash = crypto.createHash("sha256");
  await new Promise((resolve, reject) => {
    const input = fs.createReadStream(filePath);
    input.on("data", (chunk) => hash.update(chunk));
    input.on("error", reject);
    input.on("end", resolve);
  });
  return hash.digest("hex");
}

export function shellQuote(value) {
  return `'${String(value).replaceAll("'", `'\\''`)}'`;
}

/** Reject broad or unexpected local/remote rsync targets before --delete is used. */
export function assertSafeSyncPaths(config) {
  const source = path.resolve(config.localRepo);
  if (!path.isAbsolute(source) || source === path.parse(source).root || !fs.statSync(source).isDirectory()) {
    throw new Error(`Unsafe or missing rsync source: ${source}`);
  }
  if (path.posix.normalize(config.remoteRepo) !== SAFE_REMOTE_REPO) {
    throw new Error(`Refusing rsync --delete outside ${SAFE_REMOTE_REPO}: ${config.remoteRepo}`);
  }
  if (path.posix.dirname(config.remoteBuild) !== SAFE_REMOTE_REPO) {
    throw new Error(`Remote build must remain below ${SAFE_REMOTE_REPO}`);
  }
}

/** Build SSH arguments without embedding the password in argv. */
export function buildSshInvocation(command, config = loadEnvironment()) {
  if (typeof command !== "string" || command.trim() === "") {
    throw new Error("Remote command must be a non-empty string");
  }
  return {
    command: "sshpass",
    args: [
      "-e",
      "ssh",
      "-F",
      "/dev/null",
      "-o",
      "StrictHostKeyChecking=accept-new",
      "-o",
      "ConnectTimeout=10",
      `${config.gpuUser}@${config.gpuHost}`,
      command,
    ],
    env: { SSHPASS: config.gpuPassword },
  };
}

/** Build guarded rsync arguments. The only destructive destination is /root/voxtral.cpp. */
export function buildRsyncInvocation(config = loadEnvironment()) {
  assertSafeSyncPaths(config);
  return {
    command: "sshpass",
    args: [
      "-e",
      "rsync",
      "-az",
      "--delete",
      "--exclude=.git/",
      "--exclude=build*/",
      "--exclude=node_modules/",
      "--exclude=.artifacts/",
      "--exclude=.cache/",
      "--exclude=*.gguf",
      // The user's private long-form fixture is intentionally outside the
      // repository contract. `.git/info/exclude` is not consulted by rsync,
      // so keep an explicit source-sync guard here as well.
      "--exclude=/voxTest2min.m4a",
      "--exclude=/voxTest2min-16k-mono-pcm16.*",
      `${config.localRepo.replace(/\/+$/u, "")}/`,
      `${config.gpuUser}@${config.gpuHost}:${config.remoteRepo}/`,
    ],
    env: { SSHPASS: config.gpuPassword },
  };
}

/**
 * Build the non-destructive rsync invocation for the private long-form fixture.
 * This deliberately targets the external test-data directory, never the source
 * checkout, and never uses --delete. The fixture is therefore not part of the
 * normal source synchronization contract.
 */
export function buildFixtureRsyncInvocation(localFixture, config = loadEnvironment()) {
  const source = path.resolve(localFixture);
  if (!path.isAbsolute(source) || !fs.statSync(source).isFile()) {
    throw new Error(`Fixture is not a regular local file: ${source}`);
  }
  const basename = path.basename(source);
  if (basename !== "voxTest2min.m4a") {
    throw new Error(`Refusing unexpected fixture name: ${basename}`);
  }
  const remoteDir = "/root/voxtral-test-data";
  const remotePath = `${remoteDir}/${basename}`;
  return {
    command: "sshpass",
    args: [
      "-e", "rsync", "-az",
      "-e", "ssh -F /dev/null -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10",
      source,
      `${config.gpuUser}@${config.gpuHost}:${remotePath}`,
    ],
    env: { SSHPASS: config.gpuPassword },
    remoteDir,
    remotePath,
  };
}

/** Transfer the private fixture outside the repository source tree. */
export async function syncFixture(localFixture, options = {}) {
  const config = options.config ?? loadEnvironment();
  const invocation = buildFixtureRsyncInvocation(localFixture, config);
  const localSha256 = await sha256File(path.resolve(localFixture));
  await runRemote(`mkdir -p ${shellQuote(invocation.remoteDir)}`, {
    config,
    timeoutMs: 30_000,
    secrets: options.secrets,
  });
  const transfer = await runProcess(invocation.command, invocation.args, {
    ...options,
    config: undefined,
    env: { ...invocation.env, ...options.env },
    timeoutMs: options.timeoutMs ?? 120_000,
    secrets: [config.gpuPassword, ...(options.secrets ?? [])].filter((secret) => secret.length > 1),
  });
  const remote = await runRemote(`sha256sum ${shellQuote(invocation.remotePath)} | awk '{print $1}'`, {
    config,
    timeoutMs: 30_000,
  });
  const remoteSha256 = remote.stdout.trim();
  if (!/^[a-f0-9]{64}$/u.test(remoteSha256) || remoteSha256 !== localSha256) {
    throw new Error(`Fixture SHA-256 mismatch after transfer: local=${localSha256} remote=${remoteSha256}`);
  }
  return { ...transfer, localSha256, remoteSha256 };
}

/**
 * Normalize the transferred M4A once on the GPU host and return metadata for
 * the exact canonical PCM consumed by every scheduler run. No audio payload is
 * copied back into the repository or into an artifact bundle.
 */
export async function normalizeFixtureOnGpu(options = {}) {
  const config = options.config ?? loadEnvironment();
  const dir = "/root/voxtral-test-data";
  const input = `${dir}/voxTest2min.m4a`;
  const wav = `${dir}/voxTest2min-16k-mono-pcm16.wav`;
  const pcm = `${dir}/voxTest2min-16k-mono-pcm16.pcm`;
  const command = [
    `test -s ${shellQuote(input)}`,
    `ffmpeg -v error -y -i ${shellQuote(input)} -ar 16000 -ac 1 -c:a pcm_s16le ${shellQuote(wav)}`,
    `ffmpeg -v error -y -i ${shellQuote(wav)} -f s16le -acodec pcm_s16le ${shellQuote(pcm)}`,
    `printf 'fixture_sha256='; sha256sum ${shellQuote(input)} | awk '{print $1}'; printf '\n'`,
    `printf 'wav_sha256='; sha256sum ${shellQuote(wav)} | awk '{print $1}'; printf '\n'`,
    `printf 'pcm_sha256='; sha256sum ${shellQuote(pcm)} | awk '{print $1}'; printf '\n'`,
    `printf 'wav_bytes='; stat -c '%s' ${shellQuote(wav)}; printf '\n'`,
    `printf 'pcm_bytes='; stat -c '%s' ${shellQuote(pcm)}; printf '\n'`,
    `printf 'duration_seconds='; ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 ${shellQuote(wav)}; printf '\n'`,
    `printf 'sample_rate='; ffprobe -v error -select_streams a:0 -show_entries stream=sample_rate -of default=noprint_wrappers=1:nokey=1 ${shellQuote(wav)}; printf '\n'`,
    `printf 'channels='; ffprobe -v error -select_streams a:0 -show_entries stream=channels -of default=noprint_wrappers=1:nokey=1 ${shellQuote(wav)}; printf '\n'`,
    `printf 'sample_count='; stat -c '%s' ${shellQuote(pcm)} | awk '{print int($1/2)}'; printf '\n'`,
  ].join(" && ");
  const result = await runRemote(command, { config, timeoutMs: options.timeoutMs ?? 180_000 });
  const fields = {};
  for (const line of result.stdout.split(/\r?\n/u)) {
    const match = line.match(/^([a-z0-9_]+)=(.+)$/u);
    if (match) fields[match[1]] = match[2].trim();
  }
  const required = ["fixture_sha256", "wav_sha256", "pcm_sha256", "duration_seconds", "sample_rate", "channels", "sample_count"];
  for (const key of required) {
    if (!fields[key]) throw new Error(`GPU fixture normalization omitted ${key}: ${result.stdout}`);
  }
  return {
    ...fields,
    wavPath: wav,
    pcmPath: pcm,
    durationMs: Number(fields.duration_seconds) * 1000,
    sampleRate: Number(fields.sample_rate),
    channels: Number(fields.channels),
    sampleCount: Number(fields.sample_count),
    sourceSha256: fields.fixture_sha256,
    canonicalWavSha256: fields.wav_sha256,
    canonicalPcmSha256: fields.pcm_sha256,
  };
}

export async function runRemote(command, options = {}) {
  const config = options.config ?? loadEnvironment();
  const invocation = buildSshInvocation(command, config);
  try {
    return await runProcess(invocation.command, invocation.args, {
      ...options,
      config: undefined,
      env: { ...invocation.env, ...options.env },
      secrets: [config.gpuPassword, ...(options.secrets ?? [])].filter((secret) => secret.length > 1),
    });
  } catch (error) {
    error.message = `Remote command failed for ${config.gpuUser}@${config.gpuHost}: ${error.message}`;
    throw error;
  }
}

export function checkRemoteConnection(options = {}) {
  return runRemote("printf 'voxtral-ssh-ok\\n'", { timeoutMs: 20_000, ...options });
}

/** Verify that the configured remote GGUF is a non-empty regular payload. */
export function checkRemoteModel(options = {}) {
  const config = options.config ?? loadEnvironment();
  return runRemote(`test -s ${shellQuote(config.remoteModel)} && stat -c '%s' ${shellQuote(config.remoteModel)}`, {
    ...options,
    config,
  });
}

/** Return vulkaninfo summary from the configured GPU host. */
export function checkRemoteVulkan(options = {}) {
  return runRemote("vulkaninfo --summary", { timeoutMs: 30_000, ...options });
}

/** Synchronize source files to the one allowlisted remote repository. */
export async function syncSources(options = {}) {
  const config = options.config ?? loadEnvironment();
  const invocation = buildRsyncInvocation(config);
  return runProcess(invocation.command, invocation.args, {
    ...options,
    config: undefined,
    env: { ...invocation.env, ...options.env },
    timeoutMs: options.timeoutMs ?? 180_000,
    secrets: [config.gpuPassword, ...(options.secrets ?? [])].filter((secret) => secret.length > 1),
  });
}

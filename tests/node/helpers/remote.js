import fs from "node:fs";
import path from "node:path";

import { loadEnvironment } from "../config/environment.js";
import { runProcess } from "./exec.js";

const SAFE_REMOTE_REPO = "/root/voxtral.cpp";

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
      `${config.localRepo.replace(/\/+$/u, "")}/`,
      `${config.gpuUser}@${config.gpuHost}:${config.remoteRepo}/`,
    ],
    env: { SSHPASS: config.gpuPassword },
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

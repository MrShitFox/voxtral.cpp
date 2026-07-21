import { spawn } from "node:child_process";

import { redactSecrets } from "../config/environment.js";

export class ProcessExecutionError extends Error {
  constructor(message, result) {
    super(message);
    this.name = "ProcessExecutionError";
    this.result = result;
  }
}

function safeMetadata(command, args, cwd, secrets) {
  return {
    command: redactSecrets(command, secrets),
    args: args.map((arg) => redactSecrets(arg, secrets)),
    cwd: cwd ? redactSecrets(cwd, secrets) : undefined,
  };
}

/**
 * Spawn a process, continuously drain both output pipes and return structured diagnostics.
 * No shell is used. Set rejectOnNonZero=false when an exit status is itself test data.
 */
export function runProcess(command, args = [], options = {}) {
  if (typeof command !== "string" || command.length === 0) {
    throw new TypeError("command must be a non-empty string");
  }
  if (!Array.isArray(args) || args.some((arg) => typeof arg !== "string")) {
    throw new TypeError("args must be an array of strings");
  }

  const {
    cwd,
    env = {},
    timeoutMs = 120_000,
    killGraceMs = 1_000,
    rejectOnNonZero = true,
    input,
    onStdout,
    onStderr,
    signal: abortSignal,
    secrets = [],
  } = options;
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) {
    throw new TypeError("timeoutMs must be a positive finite number");
  }

  const metadata = safeMetadata(command, args, cwd, secrets);
  const startedAt = new Date();
  const startedNs = process.hrtime.bigint();

  return new Promise((resolve, reject) => {
    let settled = false;
    let timedOut = false;
    let aborted = false;
    let forceKillTimer;
    const stdoutChunks = [];
    const stderrChunks = [];

    const child = spawn(command, args, {
      cwd,
      env: { ...process.env, ...env },
      stdio: [input === undefined ? "ignore" : "pipe", "pipe", "pipe"],
      windowsHide: true,
    });

    const terminate = (reason) => {
      if (child.exitCode !== null || child.signalCode !== null) return;
      if (reason === "timeout") timedOut = true;
      if (reason === "abort") aborted = true;
      child.kill("SIGTERM");
      forceKillTimer = setTimeout(() => {
        if (child.exitCode === null && child.signalCode === null) child.kill("SIGKILL");
      }, killGraceMs);
      forceKillTimer.unref?.();
    };

    const timeout = setTimeout(() => terminate("timeout"), timeoutMs);
    const abort = () => terminate("abort");
    if (abortSignal?.aborted) abort();
    else abortSignal?.addEventListener("abort", abort, { once: true });

    child.stdout.on("data", (chunk) => {
      stdoutChunks.push(chunk);
      onStdout?.(chunk);
    });
    child.stderr.on("data", (chunk) => {
      stderrChunks.push(chunk);
      onStderr?.(chunk);
    });

    child.on("error", (error) => {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      clearTimeout(forceKillTimer);
      abortSignal?.removeEventListener("abort", abort);
      reject(new ProcessExecutionError(
        `Failed to start ${metadata.command}: ${redactSecrets(error.message, secrets)}`,
        { ...metadata, spawnError: error, timedOut, aborted },
      ));
    });

    child.on("close", (exitCode, processSignal) => {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      clearTimeout(forceKillTimer);
      abortSignal?.removeEventListener("abort", abort);
      const stdout = redactSecrets(Buffer.concat(stdoutChunks).toString("utf8"), secrets);
      const stderr = redactSecrets(Buffer.concat(stderrChunks).toString("utf8"), secrets);
      const wallMs = Number(process.hrtime.bigint() - startedNs) / 1e6;
      const result = {
        ...metadata,
        startedAt: startedAt.toISOString(),
        exitCode,
        signal: processSignal,
        timedOut,
        aborted,
        wallMs,
        stdout,
        stderr,
      };
      result.diagnostics = [
        `command: ${[result.command, ...result.args].join(" ")}`,
        `cwd: ${result.cwd ?? process.cwd()}`,
        `exitCode: ${exitCode ?? "null"}`,
        `signal: ${processSignal ?? "none"}`,
        `timedOut: ${timedOut}`,
        `wallMs: ${wallMs.toFixed(1)}`,
        "--- stdout ---",
        stdout,
        "--- stderr ---",
        stderr,
      ].join("\n");

      if (rejectOnNonZero && (exitCode !== 0 || timedOut || aborted)) {
        const reason = timedOut ? "timed out" : aborted ? "was aborted" : `exited with code ${exitCode}`;
        reject(new ProcessExecutionError(
          `${result.command} ${reason}\n${result.diagnostics}`,
          result,
        ));
      } else {
        resolve(result);
      }
    });

    if (input !== undefined) child.stdin.end(input);
  });
}

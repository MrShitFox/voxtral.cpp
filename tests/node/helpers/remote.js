import { spawn } from "node:child_process";

export const gpuConfig = Object.freeze({
  host: process.env.VOXTRAL_GPU_HOST ?? "192.168.2.136",
  user: process.env.VOXTRAL_GPU_USER ?? "root",
  password: process.env.VOXTRAL_GPU_PASSWORD ?? "",
  repository: process.env.VOXTRAL_GPU_REPOSITORY ?? "/root/voxtral.cpp",
  model:
    process.env.VOXTRAL_GPU_MODEL ??
    "/root/models/Voxtral-Mini-4B-Realtime-2602/Voxtral-Mini-4B-Realtime-2602-Q4_K_M.gguf",
});

export function shellQuote(value) {
  return `'${String(value).replaceAll("'", `'\\''`)}'`;
}

export function runRemote(command, { timeoutMs = 120_000 } = {}) {
  if (!gpuConfig.password) {
    throw new Error("VOXTRAL_GPU_PASSWORD is required for the remote baseline test");
  }

  return new Promise((resolve, reject) => {
    const child = spawn(
      "sshpass",
      [
        "-e",
        "ssh",
        "-F",
        "/dev/null",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=10",
        `${gpuConfig.user}@${gpuConfig.host}`,
        command,
      ],
      {
        env: { ...process.env, SSHPASS: gpuConfig.password },
        stdio: ["ignore", "pipe", "pipe"],
      },
    );

    let stdout = "";
    let stderr = "";
    let timedOut = false;

    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk) => {
      stdout += chunk;
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk;
    });

    const timer = setTimeout(() => {
      timedOut = true;
      child.kill("SIGTERM");
    }, timeoutMs);

    child.on("error", (error) => {
      clearTimeout(timer);
      reject(error);
    });
    child.on("close", (status, signal) => {
      clearTimeout(timer);
      resolve({
        status,
        signal,
        timedOut,
        stdout,
        stderr,
        diagnostics: [
          `command: ${command}`,
          `status: ${status}`,
          `signal: ${signal ?? "none"}`,
          `timedOut: ${timedOut}`,
          "--- stdout ---",
          stdout,
          "--- stderr ---",
          stderr,
        ].join("\n"),
      });
    });
  });
}

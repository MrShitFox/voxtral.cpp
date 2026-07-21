import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { loadEnvironment } from "../config/environment.js";
import { runProcess } from "./exec.js";
import { runRemote, shellQuote, syncSources } from "./remote.js";

function localConfigureArgs(config) {
  return [
    "-S", ".", "-B", config.localBuild, "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
    "-DVOXTRAL_NATIVE_OPT=ON",
    "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
    "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
  ];
}

/** Configure when needed, then build the native local CLI. */
export async function buildLocal({ config = loadEnvironment(), forceConfigure = false } = {}) {
  const started = process.hrtime.bigint();
  const cache = path.join(config.localBuild, "CMakeCache.txt");
  let configure = null;
  if (forceConfigure || !fs.existsSync(cache)) {
    configure = await runProcess("cmake", localConfigureArgs(config), {
      cwd: config.localRepo,
      timeoutMs: 600_000,
      env: { CCACHE_DIR: process.env.CCACHE_DIR ?? "/tmp/voxtral-ccache-local" },
    });
  }
  const build = await runProcess(
    "cmake",
    ["--build", config.localBuild, `-j${Math.max(1, os.availableParallelism?.() ?? os.cpus().length)}`],
    { cwd: config.localRepo, timeoutMs: 900_000, env: { CCACHE_DIR: process.env.CCACHE_DIR ?? "/tmp/voxtral-ccache-local" } },
  );
  if (!fs.existsSync(config.localBinary)) {
    throw new Error(`Local build completed but binary is missing: ${config.localBinary}`);
  }
  return {
    configured: configure !== null,
    configure,
    build,
    binaryPath: config.localBinary,
    wallMs: Number(process.hrtime.bigint() - started) / 1e6,
  };
}

export function syncToGpu(options = {}) {
  return syncSources(options);
}

/** Configure the remote tree only when its cache is absent, then verify Vulkan in CMakeCache. */
export async function configureRemoteVulkan({ config = loadEnvironment(), forceConfigure = false } = {}) {
  const cache = path.posix.join(config.remoteBuild, "CMakeCache.txt");
  const probe = await runRemote(`test -f ${shellQuote(cache)}`, { config, rejectOnNonZero: false });
  let configure = null;
  if (forceConfigure || probe.exitCode !== 0) {
    const configureArgs = [
      "cmake -S .",
      `-B ${shellQuote(config.remoteBuild)}`,
      "-G Ninja",
      "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
      "-DGGML_VULKAN=ON",
      "-DVOXTRAL_NATIVE_OPT=ON",
      "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
      "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
    ].join(" ");
    const command = `cd ${shellQuote(config.remoteRepo)} && ${configureArgs}`;
    configure = await runRemote(command, { config, timeoutMs: 900_000 });
  }
  const verification = await runRemote(
    `grep -Fx 'GGML_VULKAN:BOOL=ON' ${shellQuote(cache)}`,
    { config, timeoutMs: 30_000 },
  );
  return { configured: configure !== null, configure, verification, cache };
}

/** Build and locate the Vulkan executable on the GPU host. */
export async function buildRemoteVulkan({ config = loadEnvironment(), forceConfigure = false } = {}) {
  const started = process.hrtime.bigint();
  const configure = await configureRemoteVulkan({ config, forceConfigure });
  const build = await runRemote(
    `cd ${shellQuote(config.remoteRepo)} && cmake --build ${shellQuote(config.remoteBuild)} -j8`,
    { config, timeoutMs: 1_200_000 },
  );
  await runRemote(`test -x ${shellQuote(config.remoteBinary)}`, { config, timeoutMs: 30_000 });
  return {
    ...configure,
    build,
    binaryPath: config.remoteBinary,
    wallMs: Number(process.hrtime.bigint() - started) / 1e6,
  };
}

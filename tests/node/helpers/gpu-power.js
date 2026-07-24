import { loadEnvironment } from "../config/environment.js";
import { runRemote, shellQuote } from "./remote.js";

// amdgpu DPM control for the benchmark host. The only file this module writes is
// power_dpm_force_performance_level, and only to a small allowlisted set of
// documented levels. The original value is always captured and restored, so a
// benchmark can pin clocks for a measurement window without leaving the host in
// a forced state.
//
// The target DRM device is resolved dynamically instead of assuming card1:
//   1. VOXTRAL_GPU_SYSFS_DEVICE (explicit override, e.g. /sys/class/drm/card1/device);
//   2. otherwise the single AMD (vendor 0x1002) DRM device exposing a
//      power_dpm_force_performance_level control is auto-detected;
//   3. zero or more-than-one candidate is a clear error asking for the override.
const DEFAULT_DEVICE = "/sys/class/drm/card1/device";
const AMD_VENDOR_ID = "0x1002";
const VALID_LEVELS = new Set([
  "auto",
  "low",
  "high",
  "profile_standard",
  "profile_min_sclk",
  "profile_min_mclk",
  "profile_peak",
]);

// Resolved device path memoised per environment config (env is constant per run).
const deviceCache = new WeakMap();

/** Resolve the amdgpu DRM device directory on the remote host (see rules above). */
export async function resolveDevice(options = {}) {
  const config = options.config ?? loadEnvironment();
  if (deviceCache.has(config)) return deviceCache.get(config);

  const override = (process.env.VOXTRAL_GPU_SYSFS_DEVICE ?? "").trim();
  let device;
  if (override) {
    const probe = await runRemote(
      `test -e ${shellQuote(`${override}/power_dpm_force_performance_level`)}`,
      { config, timeoutMs: 20_000, rejectOnNonZero: false },
    );
    if (probe.exitCode !== 0) {
      throw new Error(
        `VOXTRAL_GPU_SYSFS_DEVICE=${override} has no power_dpm_force_performance_level control`,
      );
    }
    device = override;
  } else {
    // Enumerate AMD DRM devices that expose a DPM performance-level control.
    const script =
      'for d in /sys/class/drm/card*/device; do ' +
      '[ -e "$d/power_dpm_force_performance_level" ] || continue; ' +
      `[ "$(cat "$d/vendor" 2>/dev/null)" = "${AMD_VENDOR_ID}" ] || continue; ` +
      'echo "$d"; done';
    const result = await runRemote(script, { config, timeoutMs: 20_000 });
    const devices = result.stdout.split("\n").map((s) => s.trim()).filter(Boolean);
    if (devices.length === 0) {
      throw new Error(
        "no AMD DRM device with DPM control found; set VOXTRAL_GPU_SYSFS_DEVICE explicitly",
      );
    }
    if (devices.length > 1) {
      throw new Error(
        `ambiguous AMD DRM devices [${devices.join(", ")}]; set VOXTRAL_GPU_SYSFS_DEVICE to choose one`,
      );
    }
    device = devices[0];
  }
  deviceCache.set(config, device);
  return device;
}

async function levelPath(config) {
  return `${await resolveDevice({ config })}/power_dpm_force_performance_level`;
}

export async function readPowerLevel(options = {}) {
  const config = options.config ?? loadEnvironment();
  const result = await runRemote(`cat ${shellQuote(await levelPath(config))}`, {
    config,
    timeoutMs: 20_000,
  });
  return result.stdout.trim();
}

export async function setPowerLevel(level, options = {}) {
  if (!VALID_LEVELS.has(level)) {
    throw new Error(`Refusing to write unknown power level: ${level}`);
  }
  const config = options.config ?? loadEnvironment();
  const path = await levelPath(config);
  // sysfs write must be root; the box is root over SSH. echo appends the newline
  // the attribute handler tolerates on this driver.
  await runRemote(`echo ${shellQuote(level)} > ${shellQuote(path)}`, { config, timeoutMs: 20_000 });
  const applied = await readPowerLevel({ config });
  if (applied !== level) {
    throw new Error(`Power level did not apply: requested ${level}, read back ${applied}`);
  }
  return applied;
}

/** Return the raw pp_dpm_sclk / pp_dpm_mclk tables (the '*' marks the active state). */
export async function readClockTables(options = {}) {
  const config = options.config ?? loadEnvironment();
  const device = await resolveDevice({ config });
  const result = await runRemote(
    `printf 'sclk\\n'; cat ${shellQuote(`${device}/pp_dpm_sclk`)}; printf 'mclk\\n'; cat ${shellQuote(`${device}/pp_dpm_mclk`)}`,
    { config, timeoutMs: 20_000 },
  );
  return result.stdout;
}

/**
 * Pin the requested power level for the duration of `fn`, then restore the
 * original level even if `fn` throws. Returns whatever `fn` resolves to.
 */
export async function withPowerLevel(level, fn, options = {}) {
  const config = options.config ?? loadEnvironment();
  const original = await readPowerLevel({ config });
  await setPowerLevel(level, { config });
  try {
    return await fn({ original, applied: level });
  } finally {
    try {
      await setPowerLevel(original, { config });
    } catch (error) {
      // A failed restore must be loud: the host would be left pinned.
      console.error(`[gpu-power] FAILED to restore power level to '${original}': ${error.message}`);
    }
  }
}

export { VALID_LEVELS, DEFAULT_DEVICE };

import { loadEnvironment } from "../config/environment.js";
import { runRemote, shellQuote } from "./remote.js";

// amdgpu DPM control for the RX 6600 (card1). The only file this module writes
// is power_dpm_force_performance_level, and only to a small allowlisted set of
// documented levels. The original value is always captured and restored, so a
// benchmark can pin clocks for a measurement window without leaving the host in
// a forced state.
const DEVICE = "/sys/class/drm/card1/device";
const LEVEL_PATH = `${DEVICE}/power_dpm_force_performance_level`;
const VALID_LEVELS = new Set([
  "auto",
  "low",
  "high",
  "profile_standard",
  "profile_min_sclk",
  "profile_min_mclk",
  "profile_peak",
]);

export async function readPowerLevel(options = {}) {
  const config = options.config ?? loadEnvironment();
  const result = await runRemote(`cat ${shellQuote(LEVEL_PATH)}`, { config, timeoutMs: 20_000 });
  return result.stdout.trim();
}

export async function setPowerLevel(level, options = {}) {
  if (!VALID_LEVELS.has(level)) {
    throw new Error(`Refusing to write unknown power level: ${level}`);
  }
  const config = options.config ?? loadEnvironment();
  // sysfs write must be root; the box is root over SSH. echo appends the newline
  // the attribute handler tolerates on this driver.
  await runRemote(`echo ${shellQuote(level)} > ${shellQuote(LEVEL_PATH)}`, { config, timeoutMs: 20_000 });
  const applied = await readPowerLevel({ config });
  if (applied !== level) {
    throw new Error(`Power level did not apply: requested ${level}, read back ${applied}`);
  }
  return applied;
}

/** Return the raw pp_dpm_sclk / pp_dpm_mclk tables (the '*' marks the active state). */
export async function readClockTables(options = {}) {
  const config = options.config ?? loadEnvironment();
  const result = await runRemote(
    `printf 'sclk\\n'; cat ${shellQuote(`${DEVICE}/pp_dpm_sclk`)}; printf 'mclk\\n'; cat ${shellQuote(`${DEVICE}/pp_dpm_mclk`)}`,
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

export { VALID_LEVELS, LEVEL_PATH, DEVICE };

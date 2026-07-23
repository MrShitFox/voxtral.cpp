import path from "node:path";

import { loadEnvironment } from "../config/environment.js";
import { runRemote, shellQuote } from "./remote.js";

/** Absolute path of the streaming-skeleton harness on the GPU host. */
export function remoteStreamBinary(config = loadEnvironment()) {
  return path.posix.join(config.remoteBuild, "voxtral-stream-test");
}

/** Turn a createChunkPlan() result into the plain sample-count list the harness reads. */
export function planCounts(plan) {
  return plan.events.map((event) => event.sampleCount);
}

/** Write a feed plan (one sample count per line) to the remote repo over SSH stdin. */
export async function writeRemotePlan(remotePath, counts, { config = loadEnvironment() } = {}) {
  const text = `${counts.join("\n")}\n`;
  await runRemote(`cat > ${shellQuote(remotePath)}`, { config, input: text, timeoutMs: 30_000 });
  return remotePath;
}

function detectEvidence(stdout, stderr) {
  const combined = `${stdout}\n${stderr}`;
  return {
    vulkanEnabled: /backend:\s*VULKAN|runtime_backends:\s*[^\n]*Vulkan\([1-9]/iu.test(combined),
    rx6600Detected: /AMD Radeon RX 6600|RADV NAVI23/iu.test(combined),
    cpuOnlyFallbackDetected: /no GPU backend available, using CPU/iu.test(combined),
  };
}

/**
 * Drive the incremental stream runtime once with a given feed plan and return its parsed
 * JSON result plus backend evidence. `counts` is authoritative; when omitted the
 * harness feeds the whole clip in one call (built-in "full" mode).
 */
export async function runStreamSession({
  config = loadEnvironment(),
  planName = "plan",
  counts = null,
  mode = null,
  audioPath = null,
  modelPath = null,
  maxTokens = 0,
  ab = false,
  kvParity = false,
  warmup = false,
  env = null,
  realtimeMs = 0,
  skipParity = false,
  manualOracle = false,
  monitorMemory = false,
  gpuTelemetry = false,
  telemetryIntervalMs = 100,
  maxEvents = 0,
  backpressure = false,
  captureRolloverMemory = false,
  mallocTrimAfter = false,
  discardEventHistory = false,
  syntheticSeconds = 0,
  maxTotalSamples = 0,
  sequentialStreams = 0,
  sequentialSamples = 0,
  timeoutMs = 300_000,
} = {}) {
  const binary = remoteStreamBinary(config);
  const audio = audioPath ?? config.remoteSmokeAudio;
  const model = modelPath ?? config.remoteModel;

  const args = [
    shellQuote(binary),
    "--model", shellQuote(model),
    "--wav", shellQuote(audio),
    "--gpu", "vulkan",
    "--max-tokens", String(maxTokens),
  ];

  // --ab: load the model once, run two streams (A then B) each owning its own
  // context, and emit both results plus a distinctContexts flag.
  if (ab) {
    args.push("--ab");
  }
  if (kvParity) {
    args.push("--kv-parity");
  }
  if (warmup) {
    args.push("--warmup");
  }
  if (realtimeMs > 0) {
    args.push("--realtime-ms", String(realtimeMs));
  }
  if (skipParity) {
    args.push("--skip-parity");
  }
  if (manualOracle) {
    args.push("--manual-oracle");
  }
  if (maxEvents > 0) {
    args.push("--max-events", String(maxEvents));
  }
  if (backpressure) {
    args.push("--backpressure");
  }
  if (captureRolloverMemory) {
    args.push("--capture-rollover-memory");
  }
  if (mallocTrimAfter) {
    args.push("--malloc-trim-after");
  }
  if (discardEventHistory) {
    args.push("--discard-event-history");
  }
  if (syntheticSeconds > 0) {
    args.push("--synthetic-seconds", String(syntheticSeconds));
  }
  if (maxTotalSamples > 0) {
    args.push("--max-total-samples", String(maxTotalSamples));
  }
  if (sequentialStreams > 0) {
    args.push("--sequential-streams", String(sequentialStreams));
  }
  if (sequentialSamples > 0) {
    args.push("--sequential-samples", String(sequentialSamples));
  }

  let planFile = null;
  if (counts) {
    planFile = path.posix.join(config.remoteRepo, `.stream-plan-${planName}.txt`);
    await writeRemotePlan(planFile, counts, { config });
    args.push("--plan-file", shellQuote(planFile));
  } else if (mode) {
    args.push("--mode", mode);
  }

  // Optional environment (e.g. VOXTRAL_ENCODER_STRATEGY=reference, VOXTRAL_ENC_KV_GRID).
  const envPrefix = env
    ? Object.entries(env).map(([k, v]) => `${k}=${shellQuote(String(v))}`).join(" ") + " "
    : "";
  const streamCommand = `${envPrefix}${args.join(" ")}`;
  let command = `cd ${shellQuote(config.remoteRepo)} && ${streamCommand}`;
  // A single remote sampler covers exactly the workload lifetime. monitorMemory
  // keeps its original 50 ms VRAM/RSS peak fidelity; gpuTelemetry additionally
  // samples amdgpu sysfs (busy%, clocks, power, temp) at telemetryIntervalMs and
  // emits a summary line plus a retained per-tick CSV for ceiling analysis.
  if (monitorMemory || gpuTelemetry) {
    const safePlan = String(planName).replaceAll(/[^a-zA-Z0-9_-]/gu, "-");
    const stem = `/tmp/voxtral-memory-${process.pid}-${Date.now()}-${safePlan}`;
    const stdoutPath = `${stem}.stdout`;
    const stderrPath = `${stem}.stderr`;
    const gpuCsvPath = `${stem}.gpu.csv`;
    const vramPath = "/sys/class/drm/card1/device/mem_info_vram_used";
    const sleepSec = gpuTelemetry ? Math.max(0.02, telemetryIntervalMs / 1000).toFixed(3) : "0.05";
    const lines = [`cd ${shellQuote(config.remoteRepo)} && (`];
    if (gpuTelemetry) {
      lines.push(
        "DEV=/sys/class/drm/card1/device",
        "HW=$(ls -d $DEV/hwmon/hwmon* 2>/dev/null | head -1)",
        "topsclk=$(awk '{for(i=1;i<=NF;i++) if($i ~ /Mhz/){v=$i;sub(/Mhz/,\"\",v);if(v+0>m)m=v+0}} END{print m+0}' $DEV/pp_dpm_sclk 2>/dev/null)",
        "topmclk=$(awk '{for(i=1;i<=NF;i++) if($i ~ /Mhz/){v=$i;sub(/Mhz/,\"\",v);if(v+0>m)m=v+0}} END{print m+0}' $DEV/pp_dpm_mclk 2>/dev/null)",
        `printf 't,busy,membusy,sclk,mclk,power_uw,temp_mc\\n' > ${shellQuote(gpuCsvPath)}`,
        "pcpu=0; clktck=$(getconf CLK_TCK 2>/dev/null); clktck=${clktck:-100}",
      );
    }
    lines.push(
      `baseline_vram=$(cat ${shellQuote(vramPath)} 2>/dev/null || printf '0')`,
      `${streamCommand} >${shellQuote(stdoutPath)} 2>${shellQuote(stderrPath)} & stream_pid=$!`,
      "peak_rss=0; peak_vram=$baseline_vram",
      "while kill -0 $stream_pid 2>/dev/null; do",
      "rss=$(awk '/VmHWM:/ {print $2}' /proc/$stream_pid/status 2>/dev/null); rss=${rss:-0}",
      `vram=$(cat ${shellQuote(vramPath)} 2>/dev/null || printf '0'); vram=\${vram:-0}`,
      "if [ $rss -gt $peak_rss ]; then peak_rss=$rss; fi",
      "if [ $vram -gt $peak_vram ]; then peak_vram=$vram; fi",
    );
    if (gpuTelemetry) {
      lines.push(
        "gb=$(cat $DEV/gpu_busy_percent 2>/dev/null); gb=${gb:-0}",
        "gmb=$(cat $DEV/mem_busy_percent 2>/dev/null); gmb=${gmb:-0}",
        "gs=$(awk '/\\*/{for(i=1;i<=NF;i++) if($i ~ /Mhz/){v=$i;sub(/Mhz/,\"\",v);print v;exit}}' $DEV/pp_dpm_sclk 2>/dev/null); gs=${gs:-0}",
        "gm=$(awk '/\\*/{for(i=1;i<=NF;i++) if($i ~ /Mhz/){v=$i;sub(/Mhz/,\"\",v);print v;exit}}' $DEV/pp_dpm_mclk 2>/dev/null); gm=${gm:-0}",
        "gp=$(cat $HW/power1_average 2>/dev/null); gp=${gp:-0}",
        "gt=$(cat $HW/temp1_input 2>/dev/null); gt=${gt:-0}",
        // Process CPU time (utime+stime, clock ticks) of the stream child; the
        // last sample before exit gives total CPU-seconds vs wall = cores used.
        "pc=$(awk '{print $14+$15}' /proc/$stream_pid/stat 2>/dev/null); pcpu=${pc:-$pcpu}",
        `printf '%s,%s,%s,%s,%s,%s,%s\\n' "$(date +%s.%N)" "$gb" "$gmb" "$gs" "$gm" "$gp" "$gt" >> ${shellQuote(gpuCsvPath)}`,
      );
    }
    lines.push(
      `sleep ${sleepSec}`,
      "done",
      "wait $stream_pid; stream_status=$?",
      `final_vram=$(cat ${shellQuote(vramPath)} 2>/dev/null || printf '0'); final_vram=\${final_vram:-0}`,
      `cat ${shellQuote(stdoutPath)}`,
      `cat ${shellQuote(stderrPath)} >&2`,
      "printf '[VOXTRAL_MONITOR] baselineVramBytes=%s peakVramBytes=%s finalVramBytes=%s peakRssKiB=%s\\n' \"$baseline_vram\" \"$peak_vram\" \"$final_vram\" \"$peak_rss\" >&2",
    );
    if (gpuTelemetry) {
      lines.push(
        `awk -F, 'NR>1{n++; b+=$2; if($2+0>bm)bm=$2+0; mb+=$3; if($3+0>mbm)mbm=$3+0; s+=$4; if($4+0>sm)sm=$4+0; if($4+0>=TOPS)ts++; m+=$5; if($5+0>mm)mm=$5+0; if($5+0>=TOPM)tm++; p+=$6; if($6+0>pm)pm=$6+0; if($7+0>tp)tp=$7+0} END{if(n>0) printf "[VOXTRAL_GPU] samples=%d busyMean=%.1f busyMax=%d memBusyMean=%.1f memBusyMax=%d sclkMean=%.0f sclkMax=%d fracTopSclk=%.3f mclkMean=%.0f mclkMax=%d fracTopMclk=%.3f powerMeanW=%.2f powerMaxW=%.2f tempMaxC=%.1f csv=%s\\n", n,b/n,bm,mb/n,mbm,s/n,sm,ts/n,m/n,mm,tm/n,p/n/1e6,pm/1e6,tp/1000,CSV}' TOPS="$topsclk" TOPM="$topmclk" CSV=${shellQuote(gpuCsvPath)} ${shellQuote(gpuCsvPath)} >&2`,
        `printf '[VOXTRAL_CPU] procCpuTicks=%s clkTck=%s\\n' "$pcpu" "$clktck" >&2`,
      );
    }
    lines.push(
      `rm -f ${shellQuote(stdoutPath)} ${shellQuote(stderrPath)}`,
      "exit $stream_status",
      ")",
    );
    command = lines.join("\n");
  }
  const proc = await runRemote(command, { config, timeoutMs });

  const jsonLine = proc.stdout
    .split(/\r?\n/u)
    .map((line) => line.trim())
    .reverse()
    .find((line) => line.startsWith("{"));
  if (!jsonLine) {
    throw new Error(`stream harness produced no JSON\n--- stdout ---\n${proc.stdout}\n--- stderr ---\n${proc.stderr}`);
  }

  const result = JSON.parse(jsonLine);
  result.planName = planName;
  result.planFile = planFile;
  result.commandLine = command;
  result.wallMs = proc.wallMs;
  result.childExited = true;
  result.childExitRssKiB = 0;
  result.evidence = detectEvidence(proc.stdout, proc.stderr);
  const memory = proc.stderr.match(
    /\[VOXTRAL_MONITOR\] baselineVramBytes=(\d+) peakVramBytes=(\d+) finalVramBytes=(\d+) peakRssKiB=(\d+)/u,
  );
  if (memory) {
    result.baselineVramBytes = Number(memory[1]);
    result.peakVramBytes = Number(memory[2]);
    result.finalVramBytes = Number(memory[3]);
    result.peakRssKiB = Number(memory[4]);
  }
  const gpu = proc.stderr.match(/\[VOXTRAL_GPU\] (.+)/u);
  if (gpu) {
    const fields = {};
    for (const pair of gpu[1].trim().split(/\s+/u)) {
      const eq = pair.indexOf("=");
      if (eq <= 0) continue;
      const key = pair.slice(0, eq);
      const raw = pair.slice(eq + 1);
      fields[key] = /^-?\d+(?:\.\d+)?$/u.test(raw) ? Number(raw) : raw;
    }
    result.gpuTelemetry = fields;
  }
  const cpu = proc.stderr.match(/\[VOXTRAL_CPU\] procCpuTicks=(\d+) clkTck=(\d+)/u);
  if (cpu) {
    const ticks = Number(cpu[1]);
    const clkTck = Number(cpu[2]) || 100;
    const cpuSeconds = ticks / clkTck;
    const wallSec = (result.wallDurationMs || 0) / 1000;
    result.cpuTelemetry = {
      procCpuSeconds: Number(cpuSeconds.toFixed(2)),
      wallSeconds: Number(wallSec.toFixed(2)),
      // Average CPU cores busy over the workload (child process only).
      cpuCoresUsed: wallSec > 0 ? Number((cpuSeconds / wallSec).toFixed(3)) : 0,
    };
  }
  result.stdout = proc.stdout;
  result.stderr = proc.stderr;
  return result;
}

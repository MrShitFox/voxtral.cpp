import { loadEnvironment } from "../config/environment.js";
import { writeArtifactBundle } from "../helpers/artifacts.js";
import { runProcess } from "../helpers/exec.js";

const config = loadEnvironment();
const nodeCwd = new URL("..", import.meta.url).pathname;

// Prepare local artifacts, run the model-free unit + C++ tests, sync + build the
// Vulkan target (which includes voxtral-stream-test), then run the incremental
// causal encoder GPU acceptance vitest (batch-parity tensor, chunk invariance,
// during-feed emission, bounded work).
const steps = [
  { name: "environment", command: "npm", args: ["run", "test:environment"] },
  { name: "unit", command: "npm", args: ["run", "test:unit"] },
  { name: "local-build", command: "npm", args: ["run", "build:local"] },
  { name: "cpp-unit", command: "ctest", args: ["--test-dir", config.localBuild, "--output-on-failure"], cwd: config.localRepo },
  { name: "source-sync", command: "npm", args: ["run", "sync:gpu"] },
  { name: "vulkan-build", command: "npm", args: ["run", "build:gpu"] },
  { name: "encoder-kv", command: "npm", args: ["run", "test:encoder-kv:gpu"] },
];

const summary = { startedAt: new Date().toISOString(), steps: [] };

try {
  for (const step of steps) {
    console.log(`[encoder-kv-acceptance] ${step.name}`);
    const result = await runProcess(step.command, step.args, {
      cwd: step.cwd ?? nodeCwd,
      timeoutMs: step.name.includes("build") || step.name === "encoder-kv" ? 1_800_000 : 600_000,
      onStdout: (chunk) => process.stdout.write(chunk),
      onStderr: (chunk) => process.stderr.write(chunk),
    });
    summary.steps.push({ name: step.name, exitCode: result.exitCode, wallMs: result.wallMs });
  }
  summary.exitCode = 0;
} catch (error) {
  summary.exitCode = error.result?.exitCode ?? 1;
  summary.error = error.message;
  summary.steps.push({ name: "failed", exitCode: summary.exitCode, wallMs: error.result?.wallMs });
  process.exitCode = 1;
} finally {
  summary.finishedAt = new Date().toISOString();
  const artifact = await writeArtifactBundle({
    config,
    testName: "encoder-kv-acceptance-summary",
    backend: "Vulkan",
    command: "npm run acceptance:encoder-kv",
    result: summary,
  });
  console.log(`[encoder-kv-acceptance] summary: ${artifact.directory}`);
}

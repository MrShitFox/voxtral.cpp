import { loadEnvironment } from "../config/environment.js";
import { writeArtifactBundle } from "../helpers/artifacts.js";
import { runProcess } from "../helpers/exec.js";

const config = loadEnvironment();
const nodeCwd = new URL("..", import.meta.url).pathname;

// Each step is either an npm script or a direct command. The stream-skeleton
// vitest itself rebuilds the remote target and runs the batch baseline for
// reference, so this script only needs to prepare local artifacts and sync.
const steps = [
  { name: "environment", command: "npm", args: ["run", "test:environment"] },
  { name: "unit", command: "npm", args: ["run", "test:unit"] },
  { name: "local-build", command: "npm", args: ["run", "build:local"] },
  { name: "cpp-unit", command: "ctest", args: ["--test-dir", config.localBuild, "--output-on-failure"], cwd: config.localRepo },
  { name: "source-sync", command: "npm", args: ["run", "sync:gpu"] },
  { name: "vulkan-build", command: "npm", args: ["run", "build:gpu"] },
  { name: "stream-skeleton", command: "npm", args: ["run", "test:stream-skeleton:gpu"] },
];

const summary = { startedAt: new Date().toISOString(), steps: [] };

try {
  for (const step of steps) {
    console.log(`[stream-acceptance] ${step.name}`);
    const result = await runProcess(step.command, step.args, {
      cwd: step.cwd ?? nodeCwd,
      timeoutMs: step.name.includes("build") || step.name === "stream-skeleton" ? 1_200_000 : 600_000,
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
    testName: "stream-skeleton-acceptance-summary",
    backend: "Vulkan",
    command: "npm run acceptance:stream-skeleton",
    result: summary,
  });
  console.log(`[stream-acceptance] summary: ${artifact.directory}`);
}

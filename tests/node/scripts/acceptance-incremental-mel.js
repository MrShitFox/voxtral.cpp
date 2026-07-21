import { loadEnvironment } from "../config/environment.js";
import { writeArtifactBundle } from "../helpers/artifacts.js";
import { runProcess } from "../helpers/exec.js";

const config = loadEnvironment();
const nodeCwd = new URL("..", import.meta.url).pathname;

// Prepare local artifacts, sync + build the Vulkan target (which includes
// voxtral-stream-test), then run the incremental-Mel GPU acceptance vitest.
const steps = [
  { name: "environment", command: "npm", args: ["run", "test:environment"] },
  { name: "unit", command: "npm", args: ["run", "test:unit"] },
  { name: "local-build", command: "npm", args: ["run", "build:local"] },
  { name: "cpp-unit", command: "ctest", args: ["--test-dir", config.localBuild, "--output-on-failure"], cwd: config.localRepo },
  { name: "source-sync", command: "npm", args: ["run", "sync:gpu"] },
  { name: "vulkan-build", command: "npm", args: ["run", "build:gpu"] },
  { name: "incremental-mel", command: "npm", args: ["run", "test:incremental-mel:gpu"] },
];

const summary = { startedAt: new Date().toISOString(), steps: [] };

try {
  for (const step of steps) {
    console.log(`[incremental-mel-acceptance] ${step.name}`);
    const result = await runProcess(step.command, step.args, {
      cwd: step.cwd ?? nodeCwd,
      timeoutMs: step.name.includes("build") || step.name === "incremental-mel" ? 1_200_000 : 600_000,
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
    testName: "incremental-mel-acceptance-summary",
    backend: "Vulkan",
    command: "npm run acceptance:incremental-mel",
    result: summary,
  });
  console.log(`[incremental-mel-acceptance] summary: ${artifact.directory}`);
}

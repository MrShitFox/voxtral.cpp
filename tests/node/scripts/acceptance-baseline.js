import { loadEnvironment } from "../config/environment.js";
import { writeArtifactBundle } from "../helpers/artifacts.js";
import { runProcess } from "../helpers/exec.js";

const config = loadEnvironment();
const steps = [
  ["environment", ["run", "test:environment"]],
  ["unit", ["run", "test:unit"]],
  ["local-build", ["run", "build:local"]],
  ["source-sync", ["run", "sync:gpu"]],
  ["vulkan-build", ["run", "build:gpu"]],
  ["gpu-smoke", ["run", "test:gpu"]],
];
const summary = { startedAt: new Date().toISOString(), steps: [] };

try {
  for (const [name, args] of steps) {
    console.log(`[acceptance] ${name}`);
    const result = await runProcess("npm", args, {
      cwd: new URL("..", import.meta.url).pathname,
      timeoutMs: name.includes("build") ? 1_200_000 : 600_000,
      onStdout: (chunk) => process.stdout.write(chunk),
      onStderr: (chunk) => process.stderr.write(chunk),
    });
    summary.steps.push({ name, exitCode: result.exitCode, wallMs: result.wallMs });
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
    testName: "acceptance-baseline-summary",
    backend: "Vulkan",
    command: "npm run acceptance:baseline",
    result: summary,
  });
  console.log(`[acceptance] summary: ${artifact.directory}`);
}

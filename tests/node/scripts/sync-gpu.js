import { syncToGpu } from "../helpers/build.js";

try {
  const result = await syncToGpu();
  console.log(JSON.stringify({ exitCode: result.exitCode, wallMs: result.wallMs }, null, 2));
} catch (error) {
  console.error(error.message);
  process.exitCode = 1;
}

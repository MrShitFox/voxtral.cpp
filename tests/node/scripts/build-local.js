import { buildLocal } from "../helpers/build.js";

try {
  const result = await buildLocal({ forceConfigure: process.argv.includes("--force-configure") });
  console.log(JSON.stringify({ binaryPath: result.binaryPath, configured: result.configured, wallMs: result.wallMs }, null, 2));
} catch (error) {
  console.error(error.message);
  process.exitCode = 1;
}

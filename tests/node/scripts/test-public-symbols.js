import { readFile, realpath, stat } from "node:fs/promises";
import path from "node:path";

import { loadEnvironment } from "../config/environment.js";
import { writeArtifactBundle } from "../helpers/artifacts.js";
import { buildLocal } from "../helpers/build.js";
import { runProcess } from "../helpers/exec.js";

const config = loadEnvironment();
const summary = {
  startedAt: new Date().toISOString(),
  command: "npm run test:public-symbols",
};

try {
  await buildLocal({ config });
  const library = await realpath(
    path.join(config.localBuild, "libvoxtral.so"),
  );
  const allowlistText = await readFile(
    path.join(config.localRepo, "tests/public-symbols.txt"),
    "utf8",
  );
  const expected = allowlistText
    .split(/\r?\n/u)
    .map((line) => line.trim())
    .filter(Boolean)
    .sort();
  const nm = await runProcess(
    "nm",
    ["-D", "--defined-only", library],
    { cwd: config.localRepo, timeoutMs: 30_000 },
  );
  const readelfSymbols = await runProcess(
    "readelf",
    ["-Ws", library],
    { cwd: config.localRepo, timeoutMs: 30_000 },
  );
  const readelfDynamic = await runProcess(
    "readelf",
    ["-d", library],
    { cwd: config.localRepo, timeoutMs: 30_000 },
  );
  const allDefined = nm.stdout
    .split(/\r?\n/u)
    .map((line) => line.trim().split(/\s+/u).at(-1) ?? "")
    .map((name) => name.replace(/@@.*$/u, ""))
    .filter(Boolean);
  const actual = allDefined
    .filter((name) => name.startsWith("voxtral_"))
    .sort();
  const unexpectedDefined = allDefined.filter(
    (name) => name !== "VOXTRAL_1.0" && !expected.includes(name),
  );
  const unexpected = actual.filter((name) => !expected.includes(name));
  const missing = expected.filter((name) => !actual.includes(name));
  const mangled = nm.stdout
    .split(/\r?\n/u)
    .filter((line) => /\s_Z/u.test(line));
  const soname =
    readelfDynamic.stdout.match(/\(SONAME\).*\[([^\]]+)\]/u)?.[1] ?? null;
  const readelfPublic = expected.every((name) =>
    readelfSymbols.stdout.includes(name));
  const info = await stat(library);

  const checks = {
    expectedSymbolsPresent: missing.length === 0,
    noUnexpectedPublicSymbols: unexpected.length === 0,
    noMangledCppExports: mangled.length === 0,
    onlyDocumentedDynamicSymbols: unexpectedDefined.length === 0,
    symbolCountExact: actual.length === expected.length,
    readelfFindsPublicSymbols: readelfPublic,
    versionedSoname: soname === "libvoxtral.so.1",
  };
  if (!Object.values(checks).every(Boolean)) {
    throw new Error(
      `symbol gate failed: ${JSON.stringify({
        checks,
        missing,
        unexpected,
        unexpectedDefined,
      })}`,
    );
  }

  summary.library = library;
  summary.libraryBytes = info.size;
  summary.soname = soname;
  summary.expected = expected;
  summary.actual = actual;
  summary.missing = missing;
  summary.unexpected = unexpected;
  summary.unexpectedDefined = unexpectedDefined;
  summary.checks = checks;
  summary.exitCode = 0;
  console.log(
    `[public-symbols] PASS ${actual.length} C symbols, ` +
    `SONAME=${soname}, size=${info.size}B`,
  );
} catch (error) {
  summary.exitCode = 1;
  summary.error = error.message;
  process.exitCode = 1;
  console.error(`[public-symbols] error: ${error.message}`);
} finally {
  summary.finishedAt = new Date().toISOString();
  const artifact = await writeArtifactBundle({
    config,
    testName: "public-symbols",
    backend: "local",
    command: summary.command,
    result: summary,
  });
  console.log(
    `[public-symbols] ${summary.exitCode === 0 ? "PASS" : "FAIL"} ` +
    `summary: ${artifact.directory}`,
  );
}

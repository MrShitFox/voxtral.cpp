import path from "node:path";

const DEFAULT_LOCAL_REPO = "/home/glebus/Desktop/Code/cppShit/voxtral.cpp";

export const ENVIRONMENT_KEYS = Object.freeze([
  "VOXTRAL_LOCAL_REPO",
  "VOXTRAL_GPU_HOST",
  "VOXTRAL_GPU_USER",
  "VOXTRAL_GPU_PASSWORD",
  "VOXTRAL_REMOTE_REPO",
  "VOXTRAL_REMOTE_MODEL",
  "VOXTRAL_REMOTE_BUILD",
  "VOXTRAL_FIXTURE_2MIN",
  "VOXTRAL_FIXTURE_4MIN",
  "VOXTRAL_ARTIFACT_DIR",
]);

export const DEFAULT_ENVIRONMENT = Object.freeze({
  VOXTRAL_LOCAL_REPO: DEFAULT_LOCAL_REPO,
  VOXTRAL_GPU_HOST: "192.168.2.136",
  VOXTRAL_GPU_USER: "root",
  VOXTRAL_GPU_PASSWORD: "1",
  VOXTRAL_REMOTE_REPO: "/root/voxtral.cpp",
  VOXTRAL_REMOTE_MODEL:
    "/root/models/Voxtral-Mini-4B-Realtime-2602/Voxtral-Mini-4B-Realtime-2602-Q4_K_M.gguf",
  VOXTRAL_REMOTE_BUILD: "/root/voxtral.cpp/build-vulkan",
  VOXTRAL_FIXTURE_2MIN: "/root/voxtral-fixtures/voxTest2min.m4a",
  VOXTRAL_FIXTURE_4MIN: "/root/voxtral-fixtures/voxTest4min.m4a",
  VOXTRAL_ARTIFACT_DIR: path.join(DEFAULT_LOCAL_REPO, "tests/node/.artifacts"),
});

function requiredString(name, value) {
  if (typeof value !== "string" || value.trim() === "") {
    throw new Error(`${name} must be a non-empty string`);
  }
  return value.trim();
}

/** Load and validate the harness configuration without caching process.env. */
export function loadEnvironment(env = process.env, overrides = {}) {
  const values = {};
  for (const key of ENVIRONMENT_KEYS) {
    values[key] = requiredString(key, overrides[key] ?? env[key] ?? DEFAULT_ENVIRONMENT[key]);
  }

  const localRepo = path.resolve(values.VOXTRAL_LOCAL_REPO);
  const artifactDir = path.resolve(values.VOXTRAL_ARTIFACT_DIR);
  const remoteRepo = path.posix.normalize(values.VOXTRAL_REMOTE_REPO);
  const remoteBuild = path.posix.normalize(values.VOXTRAL_REMOTE_BUILD);

  if (!path.posix.isAbsolute(remoteRepo) || remoteRepo === "/" || remoteRepo === "/root") {
    throw new Error("VOXTRAL_REMOTE_REPO must be an absolute safe subdirectory");
  }
  if (path.posix.dirname(remoteBuild) !== remoteRepo) {
    throw new Error("VOXTRAL_REMOTE_BUILD must be a direct child of VOXTRAL_REMOTE_REPO");
  }

  return Object.freeze({
    localRepo,
    localBuild: path.join(localRepo, "build-local"),
    localBinary: path.join(localRepo, "build-local/voxtral"),
    gpuHost: values.VOXTRAL_GPU_HOST,
    gpuUser: values.VOXTRAL_GPU_USER,
    gpuPassword: values.VOXTRAL_GPU_PASSWORD,
    remoteRepo,
    remoteModel: path.posix.normalize(values.VOXTRAL_REMOTE_MODEL),
    remoteBuild,
    remoteBinary: path.posix.join(remoteBuild, "voxtral"),
    remoteFixture2min: path.posix.normalize(values.VOXTRAL_FIXTURE_2MIN),
    remoteFixture4min: path.posix.normalize(values.VOXTRAL_FIXTURE_4MIN),
    remoteSmokeAudio: path.posix.join(remoteRepo, "samples/8297-275156-0000.wav"),
    localSmokeAudio: path.join(localRepo, "samples/8297-275156-0000.wav"),
    artifactDir,
  });
}

/** Replace secrets in strings before diagnostics or persistence. */
export function redactSecrets(value, secrets = []) {
  let result = String(value);
  for (const secret of secrets.filter((item) => typeof item === "string" && item.length > 0)) {
    result = result.split(secret).join("<redacted>");
  }
  return result;
}

/** Return a serializable configuration that is safe to print. */
export function describeEnvironment(config) {
  return {
    ...config,
    gpuPassword: "<redacted>",
  };
}

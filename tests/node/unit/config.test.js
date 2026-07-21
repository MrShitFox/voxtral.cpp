import { describe, expect, test } from "vitest";

import { describeEnvironment, loadEnvironment, redactSecrets } from "../config/environment.js";

describe("environment configuration", () => {
  test("uses defaults and derives paths on every call", () => {
    const first = loadEnvironment({});
    const second = loadEnvironment({ VOXTRAL_GPU_HOST: "gpu.example" });
    expect(first.gpuHost).toBe("192.168.2.136");
    expect(second.gpuHost).toBe("gpu.example");
    expect(second.remoteBinary).toBe("/root/voxtral.cpp/build-vulkan/voxtral");
  });

  test("environment values override defaults", () => {
    expect(loadEnvironment({ VOXTRAL_GPU_USER: "tester" }).gpuUser).toBe("tester");
  });

  test("rejects empty values", () => {
    expect(() => loadEnvironment({ VOXTRAL_GPU_HOST: " " })).toThrow(/non-empty/u);
  });

  test("rejects unsafe remote paths", () => {
    expect(() => loadEnvironment({ VOXTRAL_REMOTE_REPO: "/root" })).toThrow(/safe subdirectory/u);
    expect(() => loadEnvironment({ VOXTRAL_REMOTE_BUILD: "/tmp/build" })).toThrow(/direct child/u);
  });

  test("redacts diagnostics and serialized configuration", () => {
    const config = loadEnvironment({}, { VOXTRAL_GPU_PASSWORD: "top-secret" });
    expect(redactSecrets("password=top-secret", [config.gpuPassword])).toBe("password=<redacted>");
    expect(JSON.stringify(describeEnvironment(config))).not.toContain("top-secret");
  });
});

import { describe, expect, test } from "vitest";

import { loadEnvironment } from "../config/environment.js";
import { assertSafeSyncPaths, buildRsyncInvocation, buildSshInvocation, shellQuote } from "../helpers/remote.js";

describe("remote helpers", () => {
  const config = loadEnvironment({}, { VOXTRAL_GPU_PASSWORD: "unit-secret" });

  test("quotes remote shell values", () => {
    expect(shellQuote("a'b")).toBe("'a'\\''b'");
  });

  test("keeps password out of SSH argv", () => {
    const invocation = buildSshInvocation("true", config);
    expect(invocation.args).toContain("root@192.168.2.136");
    expect(invocation.args.join(" ")).not.toContain("unit-secret");
    expect(invocation.env.SSHPASS).toBe("unit-secret");
  });

  test("builds guarded rsync invocation and excludes generated data", () => {
    const invocation = buildRsyncInvocation(config);
    expect(invocation.args).toContain("--delete");
    expect(invocation.args).toContain("--exclude=build*/");
    expect(invocation.args.at(-1)).toBe("root@192.168.2.136:/root/voxtral.cpp/");
    expect(invocation.args.join(" ")).not.toContain("unit-secret");
  });

  test("refuses any destructive destination outside the fixed repository", () => {
    expect(() => assertSafeSyncPaths({ ...config, remoteRepo: "/root", remoteBuild: "/root/build-vulkan" })).toThrow(/Refusing/u);
    expect(() => assertSafeSyncPaths({ ...config, remoteRepo: "/root/other", remoteBuild: "/root/other/build-vulkan" })).toThrow(/Refusing/u);
  });
});

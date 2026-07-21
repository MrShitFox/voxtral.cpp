import { describe, expect, test } from "vitest";

import { StreamingEventCollector, StreamingTransport } from "../helpers/streaming-transport.js";

describe("future streaming transport contract", () => {
  test("base transport never pretends that streaming exists", async () => {
    const transport = new StreamingTransport();
    for (const operation of ["connect", "configure", "appendPcm", "finish", "cancel", "close"]) {
      await expect(transport[operation]()).rejects.toThrow(/Not implemented/u);
    }
  });

  test("collector timestamps and returns supported events", async () => {
    const collector = new StreamingEventCollector();
    const pending = collector.waitFor("final", { timeoutMs: 1_000 });
    collector.add({ type: "token", token: 42 });
    collector.add({ type: "partial", text: "hel" });
    collector.add({ type: "final", text: "hello" });
    expect(await pending).toMatchObject({ type: "final", text: "hello" });
    expect(collector.events).toHaveLength(3);
    expect(collector.events[0].receivedAt).toMatch(/^\d{4}-/u);
  });

  test("collector supports timeout, abort and explicit cancellation", async () => {
    const collector = new StreamingEventCollector();
    await expect(collector.waitFor("completed", { timeoutMs: 10 })).rejects.toThrow(/Timed out/u);
    const controller = new AbortController();
    const aborted = collector.waitFor("error", { signal: controller.signal });
    controller.abort();
    await expect(aborted).rejects.toThrow(/aborted/u);
    const cancelled = collector.waitFor("token");
    collector.cancelPending("cancelled by test");
    await expect(cancelled).rejects.toThrow(/cancelled by test/u);
  });

  test("rejects unknown event kinds", () => {
    const collector = new StreamingEventCollector();
    expect(() => collector.add({ type: "fake-success" })).toThrow(/Unsupported/u);
  });
});

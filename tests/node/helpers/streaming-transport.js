const EVENT_TYPES = new Set(["token", "partial", "final", "error", "completed"]);

/** Abstract future transport. It cannot produce a successful fake stream. */
export class StreamingTransport {
  async connect() { throw new Error("Not implemented: StreamingTransport.connect"); }
  async configure() { throw new Error("Not implemented: StreamingTransport.configure"); }
  async appendPcm() { throw new Error("Not implemented: StreamingTransport.appendPcm"); }
  async finish() { throw new Error("Not implemented: StreamingTransport.finish"); }
  async cancel() { throw new Error("Not implemented: StreamingTransport.cancel"); }
  async close() { throw new Error("Not implemented: StreamingTransport.close"); }
}

/** Collect and await transport events while preserving receipt timestamps. */
export class StreamingEventCollector {
  #events = [];
  #waiters = new Set();

  get events() { return [...this.#events]; }

  add(event) {
    if (!event || !EVENT_TYPES.has(event.type)) throw new Error(`Unsupported streaming event type: ${event?.type}`);
    const received = { ...event, receivedAt: event.receivedAt ?? new Date().toISOString() };
    this.#events.push(received);
    for (const waiter of [...this.#waiters]) {
      if (waiter.type === received.type) waiter.resolve(received);
    }
    return received;
  }

  waitFor(type, { timeoutMs = 5_000, signal } = {}) {
    if (!EVENT_TYPES.has(type)) return Promise.reject(new Error(`Unsupported streaming event type: ${type}`));
    const existing = this.#events.find((event) => event.type === type);
    if (existing) return Promise.resolve(existing);
    return new Promise((resolve, reject) => {
      let timer;
      const waiter = {
        type,
        resolve: (event) => {
          cleanup();
          resolve(event);
        },
        reject: (error) => {
          cleanup();
          reject(error);
        },
      };
      const abort = () => waiter.reject(new Error(`Waiting for ${type} was aborted`));
      const cleanup = () => {
        clearTimeout(timer);
        signal?.removeEventListener("abort", abort);
        this.#waiters.delete(waiter);
      };
      this.#waiters.add(waiter);
      timer = setTimeout(() => waiter.reject(new Error(`Timed out waiting for ${type} after ${timeoutMs} ms`)), timeoutMs);
      if (signal?.aborted) abort();
      else signal?.addEventListener("abort", abort, { once: true });
    });
  }

  cancelPending(reason = "Streaming event collection cancelled") {
    for (const waiter of [...this.#waiters]) waiter.reject(new Error(reason));
  }
}

# Voxtral realtime browser demo

This is a dependency-free development UI built with plain HTML, CSS, vanilla
JavaScript, Web Audio API, `AudioWorklet`, and native `WebSocket`.

## Start server

Native browser WebSocket cannot add an `Authorization: Bearer ...` header. The
demo therefore uses the server's explicit unauthenticated development mode.
On the RX 6600 host, run:

```bash
cd /root/voxtral.cpp
./build-release/voxtral-server \
  --model /root/models/Voxtral-Mini-4B-Realtime-2602/Voxtral-Mini-4B-Realtime-2602-Q4_K_M.gguf \
  --listen 0.0.0.0 \
  --port 8080 \
  --no-auth \
  --allow-insecure-no-auth
```

The binary's build RUNPATH resolves
`/root/voxtral.cpp/build-release/libvoxtral.so.1`, so this build does not
require `LD_LIBRARY_PATH`.

The default endpoint in the UI is:

```text
ws://192.168.2.136:8080/v1/realtime/transcription
```

## Serve demo

Serve the directory from trusted localhost. Do not open `index.html` with a
`file://` URL because microphone capture and `AudioWorklet` require a secure
browser context.

```bash
cd /home/glebus/Desktop/Code/cppShit/voxtral.cpp/examples/web
python3 -m http.server 8000 --bind 127.0.0.1
```

Open:

```text
http://127.0.0.1:8000
```

Press **Start microphone**. The demo opens the WebSocket, sends
`session.configure`, waits for `session.created`, requests microphone access,
resamples the device rate to mono 16 kHz PCM16LE, and sends 80 ms chunks.

**Security warning:** The demo uses unauthenticated `ws://` for development on
a trusted LAN. Do not expose this configuration to the public Internet.
Production deployment needs a reverse proxy, TLS/WSS, and browser-compatible
authentication; those are intentionally not implemented by this demo.

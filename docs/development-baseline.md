# Development baseline: Vulkan inference on Radeon RX 6600

Baseline captured on 2026-07-21 before development of a public streaming runtime or network API.
The source checkout started clean at commit `b2b239724afd9b29d87259351fd6d1eba674f631` on branch
`main`. The only runtime correction made while capturing the baseline is described under
[Known limitations and findings](#7-known-limitations-and-findings).

## 1. Environment

| Component | Local development machine | GPU server |
| --- | --- | --- |
| OS | NixOS 26.05 (Yarara), kernel 7.1.3 | Arch Linux, kernel 7.1.4-arch1-1 |
| CPU | Intel Xeon E5-2690 v2, 10 cores | Intel Xeon E5-2690 v2, 10 cores |
| GPU | Not used for this baseline | AMD Radeon RX 6600, Navi 23 |
| Kernel driver | N/A | `amdgpu` |
| Vulkan | N/A | Vulkan 1.4.350, Mesa RADV 26.1.5 |
| CMake | 4.1.6 | 4.4.0 |
| Ninja | 1.13.2 | 1.13.2 |
| Compiler | GCC/G++ 15.2.0 | GCC/G++ 16.1.1 |
| ccache | 4.13.6 | 4.13.6 |
| Node.js / npm | 24.18.0 / 11.16.0 | 26.4.0 / 12.0.1 |

The bundled GGML dependency is fetched by CMake from `ggml-org/ggml` at commit
`5cecdad692d868e28dbd2f7c468504770108f30c` (reported version `0.9.6`, build commit
`5cecdad6`). No other dependency is fetched by the native build.

The project builds these project targets:

- `voxtral_lib`, the shared runtime library;
- `voxtral`, the inference CLI;
- `voxtral-quantize`, the GGUF quantizer;
- optional Python help targets `voxtral_convert` and `voxtral_quantize_all` when a Python
  interpreter is found.

Backend selection uses the GGML runtime registry. `--gpu auto` selects the first registered GPU
and falls back to CPU when none is available. `--gpu vulkan` restricts GPU selection to a Vulkan
registry device, but the current implementation still falls back to CPU if no matching device is
found. A successful Vulkan context also creates a CPU backend for unsupported scheduler operations;
this is reported as `CPU fallback` and is not the same as failure to select the GPU.

The CLI accepts RIFF/WAVE with 16-bit integer PCM or 32-bit IEEE float samples. Samples from all
channels are averaged to mono. The WAV sample-rate field is read but is currently neither validated
nor resampled, so input must be prepared externally at 16 kHz. MP3 and FLAC are not parsed directly.

`VOXTRAL_NATIVE_OPT=ON` adds `-march=native -mtune=native` to `voxtral_lib` and
`voxtral-quantize` on non-Apple, non-MSVC builds. Bundled GGML independently enables its native CPU
variant. Vulkan is enabled only when GGML receives `-DGGML_VULKAN=ON`.

The existing `tests/test_voxtral_reference.py` can exit successfully without the original Hugging
Face checkpoint only when a runnable GGUF/binary or committed golden fixture is supplied; otherwise
it reports a skip. It was not used as a test client in this session. Local CMake registered zero
CTest tests because Python was not present in the active NixOS build environment.

## 2. Build

Local configuration and build:

```bash
cd /home/glebus/Desktop/Code/cppShit/voxtral.cpp
export CCACHE_DIR=/tmp/voxtral-ccache-local

cmake -S . -B build-local -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DVOXTRAL_NATIVE_OPT=ON \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache

cmake --build build-local -j"$(nproc)"
```

The writable `CCACHE_DIR` is needed in the managed NixOS environment because its default cache is
read-only. Configure took 8.831 s. The first complete successful build took 20.130 s and emitted no
compiler warnings; a no-change rebuild took 0.017 s. The initial attempt before setting
`CCACHE_DIR` failed immediately with `ccache: error: Read-only file system` and is not counted as a
build measurement.

| Local executable | Size |
| --- | ---: |
| `build-local/voxtral` | 330,416 bytes |
| `build-local/voxtral-quantize` | 2,615,080 bytes |

Server configuration and build:

```bash
cd /root/voxtral.cpp
cmake -S . -B build-vulkan -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DGGML_VULKAN=ON \
  -DVOXTRAL_NATIVE_OPT=ON \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache

cmake --build build-vulkan -j8
```

Configure took 12.035 s. The first Vulkan build, including shader generation, took 190.143 s and
emitted no compiler warnings; a no-change rebuild took 0.122 s.

| Server executable | Size |
| --- | ---: |
| `build-vulkan/voxtral` | 401,416 bytes |
| `build-vulkan/voxtral-quantize` | 2,704,664 bytes |

Build directories are not copied between hosts. Their compiler paths, ABI, CMake cache,
`-march=native` output, generated Vulkan shaders and runtime search paths are host-specific. Only
source files are synchronized; each host configures and builds independently.

## 3. Model

| Property | Value |
| --- | --- |
| Path | `/root/models/Voxtral-Mini-4B-Realtime-2602/Voxtral-Mini-4B-Realtime-2602-Q4_K_M.gguf` |
| Size | 2,897,727,648 bytes (2.7 GiB as reported by `ls -lh`) |
| Format | GGUF, Q4_K_M; runtime architecture `voxtral_realtime` |
| SHA-256 | `90131ebe4882735c185dba0ed16ce90c3f59da32776ec85a50fca482fe72e137` |

The GGUF remains on the GPU server and is excluded from synchronization and Git.

## 4. Test audio

| Audio | Duration | Codec | Rate / channels | Origin | SHA-256 |
| --- | ---: | --- | --- | --- | --- |
| `samples/8297-275156-0000.wav` | 3.580 s | PCM S16LE | 16 kHz / mono | Tracked repository fixture; upstream dataset provenance is not declared in this repository | `eb7c0b19b8d83f7c832785b23d4783d7bfc0b5b08ef7aa953b360176a8382fbe` |
| `/tmp/voxtral-baseline-long.wav` | 31.760 s | PCM S16LE | 16 kHz / mono | Four concatenated plays of tracked `samples/8297-275156-0002.wav`, generated with `ffmpeg -stream_loop 3`; execution/stability only | `a671398b3bc6880dcc290a20e6b22c6896e99c51b788d93a897bbf06486066bf` |

The source used for the synthetic long file is 7.940 s and has SHA-256
`2dc0a6844833784c34cd8093f63d4dbf903e7ec780060470bc52ac08bbff4a58`.

## 5. Results

Wall time is measured around the complete CLI process and therefore includes GGUF load, context
creation, audio preprocessing, inference and cleanup. `VmHWM` was sampled from `/proc/<pid>/status`
because GNU `/usr/bin/time` is not installed on either host. RTF is `wall time / audio duration`.
Every row is a new CLI process and reloads the model; “repeat” can benefit from OS/Vulkan caches but
is not an in-process warm-model measurement.

| Run | Audio duration | Cold/warm | Backend | Wall | RTF | Max RSS | Exit | Transcript |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| short-cold-fixed | 3.580 s | first successful process after rebuild | Vulkan | 3.715 s | 1.0377 | 413,476 KiB | 0 | T1 |
| short-repeat-1 | 3.580 s | repeat; model reloaded | Vulkan | 3.148 s | 0.8793 | 414,160 KiB | 0 | T1 |
| short-repeat-2 | 3.580 s | repeat; model reloaded | Vulkan | 3.211 s | 0.8969 | 414,080 KiB | 0 | T1 |
| long-synthetic | 31.760 s | repeat; model reloaded | Vulkan | 9.619 s | 0.3029 | 414,168 KiB | 0 | T2 |

T1, identical in all three short runs:

> What are you doing here? He asked.

T2:

> We have both seen the same newspaper, of course, and you have been the first to clear the thing
> up. That's it, isn't it? We have both seen the same newspaper, of course, and you have been the
> first to clear the thing up. That's it, isn't it? We have both seen the same newspaper, of course,
> and you have been the first to clear the thing up. That's it, isn't it? We have both seen the same
> newspaper, of course, and you have been the first to clear the thing up. That's it, isn't it?

Reported internal timings were:

| Run | Model load | Encoder | Adapter | Prefill | Decode | CLI processing summary |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| short-cold-fixed | 2054.04 ms | 123.1 ms | 1.0 ms | 374.2 ms | 981.3 ms | 3665.83 ms |
| short-repeat-1 | 1997.00 ms | 123.7 ms | 1.0 ms | 75.1 ms | 799.2 ms | 3119.76 ms |
| short-repeat-2 | 1997.66 ms | 122.6 ms | 1.0 ms | 73.5 ms | 796.7 ms | 3155.50 ms |
| long-synthetic | 1978.98 ms | 1040.7 ms | 2.7 ms | 73.3 ms | 6028.0 ms | 9597.05 ms |

Live `/sys/class/drm/card1/device/gpu_busy_percent` sampling reached 99% during every measured run.
Peak sampled VRAM usage was 4,724,043,776 bytes for the short runs and 4,809,924,608 bytes for the
long run. A process that has just exited can leave driver allocations cached, so peak absolute VRAM
is more comparable than the per-run delta.

## 6. Vulkan evidence

Independent evidence collected during this baseline:

1. `vulkaninfo --summary` reported a discrete `AMD Radeon RX 6600 (RADV NAVI23)`, driver ID
   `DRIVER_ID_MESA_RADV`, driver `radv`, Mesa `26.1.5-arch1.1`.
2. `build-vulkan/CMakeCache.txt` contains `GGML_VULKAN:BOOL=ON`; compile commands for project code
   contain `-DGGML_USE_VULKAN`.
3. `ldd build-vulkan/voxtral` resolves both `libggml-vulkan.so.0` and `libvulkan.so.1`.
4. Every successful runtime printed:

   ```text
   ggml_vulkan: 0 = AMD Radeon RX 6600 (RADV NAVI23) (radv) ...
   voxtral_I: backend: VULKAN (CPU fallback 10 threads)
   [summary] build_backend_flags: ... GGML_USE_VULKAN=ON ...
   [summary] runtime_backends: Vulkan(1 dev), CPU(1 dev)
   ```

5. Independent live counters reached 99% GPU busy and approximately 4.81 GB peak VRAM during
   inference. No run printed `no GPU backend available, using CPU`.

## 7. Known limitations and findings

- The original Vulkan run reproducibly crashed with `SIGSEGV` after adapter completion. GDB located
  the write at `clear_kv_cache()` in `src/voxtral.cpp`: host `memset()` was used on Vulkan device
  tensor data. The minimal baseline fix uses backend-aware `ggml_backend_buffer_clear()` and does
  not change cache layout or inference semantics. Reproduction evidence is retained on the server
  at `/tmp/voxtral-baseline-short-cold.log` (failed run) and
  `/tmp/voxtral-baseline-gdb.log`; all post-fix runs complete successfully.
- `kv_cache_shift_left()` still directly accesses tensor data from the host. The measured clips do
  not approach the 8192-token cache window, so this path was not exercised or changed. It must be
  audited before relying on very long-lived GPU streaming sessions.
- At the baseline commit, the public path was batch file transcription. It had no public
  streaming C API, incremental Mel frontend, WebSocket server or multi-user scheduler.
- Each CLI invocation reloads approximately 2.9 GB of model weights and allocates its context again.
- The synthetic long WAV measures execution, memory and stability only; it is not an ASR quality
  benchmark.
- A CPU backend is intentionally present in the Vulkan scheduler for unsupported operations. The
  baseline proves Vulkan selection and GPU activity, not that every individual graph node runs on
  the GPU.
- The WAV loader does not validate the declared sample rate. Non-16-kHz input can silently produce
  incorrect timing/features and should be rejected or converted by callers.

## 8. Reproduction commands

Set connection variables locally. Do not commit the password:

```bash
export VOXTRAL_GPU_HOST=192.168.2.136
export VOXTRAL_GPU_USER=root
export VOXTRAL_GPU_PASSWORD='<server password>'
export SSHPASS="$VOXTRAL_GPU_PASSWORD"
```

Synchronize source only:

```bash
sshpass -e rsync -az --delete \
  --exclude='.git/' \
  --exclude='build*/' \
  --exclude='node_modules/' \
  --exclude='.cache/' \
  --exclude='*.gguf' \
  /home/glebus/Desktop/Code/cppShit/voxtral.cpp/ \
  "$VOXTRAL_GPU_USER@$VOXTRAL_GPU_HOST:/root/voxtral.cpp/"
```

Verify the device, build and model:

```bash
sshpass -e ssh -F /dev/null -o StrictHostKeyChecking=accept-new \
  "$VOXTRAL_GPU_USER@$VOXTRAL_GPU_HOST" '
    set -e
    vulkaninfo --summary
    cd /root/voxtral.cpp
    grep -E "GGML_VULKAN|Vulkan|VOXTRAL_NATIVE_OPT|CMAKE_BUILD_TYPE" \
      build-vulkan/CMakeCache.txt
    sha256sum \
      /root/models/Voxtral-Mini-4B-Realtime-2602/Voxtral-Mini-4B-Realtime-2602-Q4_K_M.gguf
  '
```

Recreate the long fixture and inspect both audio files:

```bash
sshpass -e ssh -F /dev/null "$VOXTRAL_GPU_USER@$VOXTRAL_GPU_HOST" '
  cd /root/voxtral.cpp
  ffmpeg -y -stream_loop 3 -i samples/8297-275156-0002.wav \
    -ar 16000 -ac 1 -c:a pcm_s16le /tmp/voxtral-baseline-long.wav
  ffprobe -v error -show_entries format=duration \
    -show_entries stream=codec_name,sample_rate,channels -of json \
    samples/8297-275156-0000.wav
  ffprobe -v error -show_entries format=duration \
    -show_entries stream=codec_name,sample_rate,channels -of json \
    /tmp/voxtral-baseline-long.wav
'
```

Run inference (replace the audio path with `/tmp/voxtral-baseline-long.wav` for the long case):

```bash
sshpass -e ssh -F /dev/null "$VOXTRAL_GPU_USER@$VOXTRAL_GPU_HOST" '
  set -o pipefail
  cd /root/voxtral.cpp
  ./build-vulkan/voxtral \
    --model /root/models/Voxtral-Mini-4B-Realtime-2602/Voxtral-Mini-4B-Realtime-2602-Q4_K_M.gguf \
    --audio samples/8297-275156-0000.wav \
    --gpu vulkan \
    --log-level info \
    2>&1 | tee /tmp/voxtral-baseline-short.log
'
```

Run the local Node.js baseline harness:

```bash
cd /home/glebus/Desktop/Code/cppShit/voxtral.cpp/tests/node
npm ci
VOXTRAL_GPU_HOST=192.168.2.136 \
VOXTRAL_GPU_USER=root \
VOXTRAL_GPU_PASSWORD="$VOXTRAL_GPU_PASSWORD" \
npm test
```

The complete per-run server logs can be regenerated using the commands above. This session kept
them under `/tmp/voxtral-baseline-*.{stdout,stderr,log}` and did not add them, the GGUF, generated
WAV, build trees or `node_modules` to Git.

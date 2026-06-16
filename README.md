# Voxtral.cpp

A ggml-based C++ implementation of Voxtral.

## Models

Two model families are supported:

- [**Voxtral-Mini-4B-Realtime**](https://huggingface.co/andrijdavid/Voxtral-Mini-4B-Realtime-2602-GGUF) (`general.architecture = voxtral_realtime`) - streaming
  model that emits one token per 80 ms audio frame. Built for low-latency/longform.
- [**Voxtral-Mini-3B-2507**](https://huggingface.co/andrijdavid/Voxtral-Mini-3B-2507-GGUF) (`general.architecture = voxtral`) - offline audio-LLM that
  generates only the text tokens (Whisper-style), much faster for batch transcription of
  short clips. Convert from the official `mistralai/Voxtral-Mini-3B-2507` checkpoint:

  ```bash
  # needs consolidated.safetensors + params.json + tekken.json in the model dir
  python tools/convert_voxtral_to_gguf.py --model-dir models/voxtral-3b \
    --output models/voxtral-3b/voxtral-3b.gguf --out-type q4_k_m
  ./build/voxtral --model models/voxtral-3b/voxtral-3b.gguf --audio clip.wav
  ```

## Voxtral References

- Official Mistral Voxtral announcement: https://mistral.ai/news/voxtral
- Mistral Audio & Transcription docs: https://docs.mistral.ai/capabilities/audio_transcription
- Mistral Audio Transcriptions API: https://docs.mistral.ai/api/endpoint/audio/transcriptions

## Model Weights (GGUF)

Quantized GGUF weights used by this project are hosted on Hugging Face:

- https://huggingface.co/andrijdavid/Voxtral-Mini-4B-Realtime-2602-GGUF
- https://huggingface.co/andrijdavid/Voxtral-Mini-3B-2507-GGUF

The `download_model.sh` script downloads from that repository.

## Quickstart

### 1. Download the model

Download the pre-converted GGUF model from Hugging Face:

```bash
# Recommended: Q4_K_M (best quality/size trade-off, ~2.7 GB)
./tools/download_model.sh Q4_K_M
```

Other quants are available (`Q8_0`, `Q5_K`, `Q4_0`, `Q2_K`, …); `Q4_K_M` is the
recommended default and transcribes with no perceptible quality loss vs full
precision.

### 2. Build

Build the project using CMake:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Or via Homebrew (`Formula/voxtral.rb`):

```bash
brew tap andrijdavid/voxtral https://github.com/andrijdavid/voxtral.cpp
brew install --HEAD voxtral
```

### 3. Audio Preparation

The model expects **16-bit PCM WAV** files at **16kHz (mono)**. You can use `ffmpeg` to convert your audio files:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 -c:a pcm_s16le output.wav
```

### 4. Run Inference

```bash
./build/voxtral \
  --model models/voxtral/Q4_K_M.gguf \
  --audio path/to/input.wav
```

By default the GPU is auto-detected (`--gpu auto`): **Metal** on Apple Silicon,
**CUDA** on NVIDIA, **Vulkan** otherwise, falling back to CPU. Force a backend
with `--gpu metal|cuda|vulkan|none`. Thread count for the CPU path defaults to
the machine's hardware concurrency; override with `--threads N`.

---

## Advanced Usage

### Manual Quantization

You can quantize an existing GGUF file using the native quantizer:

```bash
./build/voxtral-quantize \
  models/voxtral/voxtral.gguf \
  models/voxtral/voxtral-q6_k.gguf \
  Q6_K \
  8
```

### `voxtral-quantize`

Command format:

```bash
./build/voxtral-quantize <input.gguf> <output.gguf> <type> [nthreads]
```

Examples:

```bash
# 1) Quantize to Q4_0 using default thread count
./build/voxtral-quantize models/voxtral/voxtral.gguf models/voxtral/Q4_0.gguf Q4_0

# 2) Quantize to Q6_K using 8 threads
./build/voxtral-quantize models/voxtral/voxtral.gguf models/voxtral/Q6_K.gguf Q6_K 8
```

Supported `type` values:

`Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q4_K_M`

Notes:

- Input must be a Voxtral GGUF (`general.architecture = voxtral_realtime`).
- `Q4_K_M` uses a mixed strategy internally (some tensors kept at higher precision).
- `nthreads` is optional; when omitted, hardware concurrency is used.

## Testing

The test suite runs over `samples/*.wav` files.

### Numeric Parity Check

To verify numeric parity against the reference implementation:

```bash
python3 tests/test_voxtral_reference.py
```

### Custom Tolerances

You can override comparison tolerances via environment variables:
- `VOXTRAL_TEST_ATOL` (default: 1e-2)
- `VOXTRAL_TEST_RTOL` (default: 1e-2)
- `VOXTRAL_TEST_THREADS`

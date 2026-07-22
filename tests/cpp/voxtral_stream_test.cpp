// ============================================================================
// voxtral-stream-test — model-driven harness for the incremental stream runtime.
//
// This is NOT a production CLI or server. It exists only to drive the internal
// voxtral_stream path from a chunked or paced PCM feed and emit a single
// machine-readable JSON object on stdout (backend logs go to stderr), so the
// Node.js acceptance suite can assert residence/backlog, chunk invariance,
// batch parity and the ownership contract (model shared; each stream owns its
// own context).
//
// Usage:
//   voxtral-stream-test --model M.gguf --wav in.wav [--gpu auto|vulkan|none]
//                       [--plan-file plan.txt] [--mode MODE] [--max-tokens N]
//                       [--realtime-ms N] [--manual-oracle] [--ab]
//
//   --plan-file : text file, one integer per line = a feed's sample count
//                 (0 = an explicit zero-length feed). Must sum to the WAV's
//                 sample count. Takes precedence over --mode.
//   --mode      : full | 80ms | 160ms | 320ms | 480ms | 1000ms |
//                 single-sample | seeded-random:SEED   (default: full)
//   --realtime-ms : pace chunk arrival from one monotonic start deadline. The
//                   built-in fixed N-ms plan is used unless --plan-file wins.
//   --skip-parity : skip the second full batch encoder pass (use only for timing
//                   sweeps; correctness runs leave parity enabled).
//   --manual-oracle : compare the production flash-KV encoder tensor with a
//                     second global encoder pass using manual attention.
//   --ab        : load the model ONCE, then create two streams A and B from it
//                 (each owns its own context), run the same plan through both
//                 sequentially, and emit both results plus a distinctContexts
//                 flag. Proves model-shared / context-per-stream ownership.
// ============================================================================

#include "voxtral-stream.h"
#include "voxtral.h"
#include "voxtral-mel.h"
#include "voxtral-internal.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// ----------------------------------------------------------------------------
// Minimal SHA-256 (public-domain style) over raw bytes.
// ----------------------------------------------------------------------------
namespace {

struct Sha256 {
    uint32_t h[8];
    uint64_t len = 0;
    uint8_t  buf[64];
    size_t   buf_len = 0;

    Sha256() {
        static const uint32_t init[8] = {
            0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
            0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
        std::memcpy(h, init, sizeof(init));
    }

    static uint32_t rotr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }

    void block(const uint8_t * p) {
        static const uint32_t k[64] = {
            0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
            0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
            0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
            0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
            0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
            0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
            0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
            0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u};
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = (uint32_t(p[i * 4]) << 24) | (uint32_t(p[i * 4 + 1]) << 16) |
                   (uint32_t(p[i * 4 + 2]) << 8) | uint32_t(p[i * 4 + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            uint32_t ch = (e & f) ^ (~e & g);
            uint32_t t1 = hh + S1 + ch + k[i] + w[i];
            uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t t2 = S0 + maj;
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }

    void update(const void * data, size_t n) {
        const uint8_t * p = static_cast<const uint8_t *>(data);
        len += n;
        while (n > 0) {
            size_t take = 64 - buf_len;
            if (take > n) take = n;
            std::memcpy(buf + buf_len, p, take);
            buf_len += take; p += take; n -= take;
            if (buf_len == 64) { block(buf); buf_len = 0; }
        }
    }

    std::string hex() {
        uint64_t bits = len * 8;
        uint8_t pad = 0x80;
        update(&pad, 1);
        uint8_t zero = 0;
        while (buf_len != 56) update(&zero, 1);
        uint8_t lenbe[8];
        for (int i = 0; i < 8; ++i) lenbe[i] = uint8_t(bits >> (56 - i * 8));
        // update() bumps len, but the final block is emitted correctly regardless.
        update(lenbe, 8);
        static const char * hexd = "0123456789abcdef";
        std::string out;
        out.reserve(64);
        for (int i = 0; i < 8; ++i) {
            for (int j = 3; j >= 0; --j) {
                uint8_t byte = uint8_t(h[i] >> (j * 8));
                out.push_back(hexd[byte >> 4]);
                out.push_back(hexd[byte & 0xf]);
            }
        }
        return out;
    }
};

// ----------------------------------------------------------------------------
// Tiny WAV reader: mono 16 kHz PCM16 only (the streaming contract).
// ----------------------------------------------------------------------------
bool read_wav_pcm16_mono16k(const std::string & path, std::vector<int16_t> & out, std::string & err) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin) { err = "cannot open wav: " + path; return false; }

    char riff[4]; fin.read(riff, 4);
    if (std::memcmp(riff, "RIFF", 4) != 0) { err = "not RIFF"; return false; }
    fin.seekg(4, std::ios::cur);
    char wave[4]; fin.read(wave, 4);
    if (std::memcmp(wave, "WAVE", 4) != 0) { err = "not WAVE"; return false; }

    uint16_t fmt = 0, ch = 0, bits = 0;
    uint32_t rate = 0, data_size = 0;
    bool have_fmt = false, have_data = false;
    while (fin.good() && !(have_fmt && have_data)) {
        char id[4]; fin.read(id, 4);
        uint32_t sz = 0; fin.read(reinterpret_cast<char *>(&sz), 4);
        if (!fin.good()) break;
        if (std::memcmp(id, "fmt ", 4) == 0) {
            fin.read(reinterpret_cast<char *>(&fmt), 2);
            fin.read(reinterpret_cast<char *>(&ch), 2);
            fin.read(reinterpret_cast<char *>(&rate), 4);
            fin.seekg(6, std::ios::cur);   // byte_rate(4) + block_align(2)
            fin.read(reinterpret_cast<char *>(&bits), 2);
            if (sz > 16) fin.seekg(sz - 16, std::ios::cur);
            have_fmt = true;
        } else if (std::memcmp(id, "data", 4) == 0) {
            data_size = sz;
            have_data = true;
        } else {
            fin.seekg(sz, std::ios::cur);
        }
    }
    if (!have_fmt || !have_data) { err = "missing fmt/data chunk"; return false; }
    if (fmt != 1 || bits != 16 || ch != 1 || rate != 16000) {
        std::ostringstream os;
        os << "unsupported wav (need mono 16kHz PCM16): fmt=" << fmt << " ch=" << ch
           << " rate=" << rate << " bits=" << bits;
        err = os.str();
        return false;
    }
    const size_t n = data_size / 2;
    out.resize(n);
    fin.read(reinterpret_cast<char *>(out.data()), std::streamsize(n * 2));
    if (size_t(fin.gcount()) != n * 2) { err = "short read on data chunk"; return false; }
    return true;
}

// ----------------------------------------------------------------------------
// Chunk plan generation for the built-in --mode (createChunkPlan-style).
// ----------------------------------------------------------------------------
uint32_t seeded_next(uint32_t & state) {
    // Mirrors tests/node/helpers/chunks.js seededGenerator for reproducibility.
    state += 0x6d2b79f5u;
    uint32_t v = state;
    v = (v ^ (v >> 15)) * (v | 1u);
    v ^= v + (v ^ (v >> 7)) * (v | 61u);
    return v ^ (v >> 14);
}

std::vector<size_t> plan_from_mode(const std::string & mode, size_t total) {
    std::vector<size_t> counts;
    auto fixed = [&](size_t k) {
        for (size_t off = 0; off < total; off += k) counts.push_back(std::min(k, total - off));
    };
    if (mode == "full" || mode.empty()) {
        if (total > 0) counts.push_back(total);
    } else if (mode == "80ms")   fixed(1280);
    else if (mode == "160ms")    fixed(2560);
    else if (mode == "320ms")    fixed(5120);
    else if (mode == "480ms")    fixed(7680);
    else if (mode == "1000ms")   fixed(16000);
    else if (mode.size() > 2 && mode.compare(mode.size() - 2, 2, "ms") == 0) {
        const long ms = std::strtol(mode.c_str(), nullptr, 10);
        if (ms > 0) fixed((size_t) ms * VOXTRAL_SAMPLE_RATE / 1000);
    }
    else if (mode == "single-sample") { for (size_t i = 0; i < total; ++i) counts.push_back(1); }
    else if (mode.rfind("seeded-random:", 0) == 0) {
        uint32_t seed = uint32_t(std::strtoul(mode.c_str() + 14, nullptr, 10));
        uint32_t state = seed;
        const size_t lo = 1, hi = 16000;
        size_t off = 0;
        while (off < total) {
            size_t want = lo + size_t(double(seeded_next(state)) / 4294967296.0 * double(hi - lo + 1));
            size_t take = std::min(want, total - off);
            counts.push_back(take);
            off += take;
        }
    } else {
        // Unknown mode -> full.
        if (total > 0) counts.push_back(total);
    }
    return counts;
}

bool plan_from_file(const std::string & path, std::vector<size_t> & counts, std::string & err) {
    std::ifstream fin(path);
    if (!fin) { err = "cannot open plan file: " + path; return false; }
    std::string line;
    while (std::getline(fin, line)) {
        // Trim.
        size_t a = line.find_first_not_of(" \t\r\n");
        if (a == std::string::npos) continue;
        size_t b = line.find_last_not_of(" \t\r\n");
        std::string tok = line.substr(a, b - a + 1);
        char * end = nullptr;
        long v = std::strtol(tok.c_str(), &end, 10);
        if (!end || *end != '\0' || v < 0) { err = "invalid plan entry: " + tok; return false; }
        counts.push_back(size_t(v));
    }
    return true;
}

std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += char(c);   // pass raw UTF-8 through
                }
        }
    }
    return out;
}

void log_stderr(voxtral_log_level lvl, const std::string & msg) {
    const char * tag = "I";
    if (lvl == voxtral_log_level::error) tag = "E";
    else if (lvl == voxtral_log_level::warn) tag = "W";
    else if (lvl == voxtral_log_level::debug) tag = "D";
    std::cerr << "voxtral_" << tag << ": " << msg << "\n";
}

// ----------------------------------------------------------------------------
// A single stream run: create a stream from the shared model (owning its own
// context), feed the plan, finish, and capture everything the acceptance suite
// needs. The stream is created and destroyed by the caller so an --ab run can
// keep A and B alive together to prove their contexts are distinct.
// ----------------------------------------------------------------------------
struct StreamRun {
    std::string          state;
    std::string          finishStatus;
    uint64_t             samplesReceived = 0;
    uint64_t             samplesConsumed = 0;
    uint64_t             feedCalls       = 0;
    uint64_t             inferenceRuns   = 0;
    uint64_t             pcmFloats       = 0;
    std::string          pcmSha;
    std::vector<int32_t> tokens;
    std::string          text;
    std::vector<voxtral_stream_event> events;
    bool                 contextOwned = false;
    bool                 ok           = false;

    // Incremental Mel frontend evidence.
    bool     incrementalMel          = false;
    int64_t  melFrames               = 0;
    int64_t  melFramesBeforeFinish   = 0;
    int64_t  melFramesFlushedAtFinish= 0;
    int64_t  dftFramesComputed       = 0;
    int64_t  pcmRetainedSamples      = 0;
    int64_t  pcmPeakRetainedSamples  = 0;
    int64_t  pcmBaseSample           = 0;
    bool     fullPcmBufferedAtFinish = false;
    bool     melHistoryRetained      = false;
    std::string melSha;
    double   melMaxAbsDeltaVsBatch   = 0.0;

    // Incremental causal encoder evidence.
    bool     incrementalEncoder      = false;
    int64_t  encoderFrames           = 0;
    int64_t  encoderFramesBeforeFinish   = 0;
    int64_t  encoderFramesFlushedAtFinish= 0;
    int64_t  encoderExecutions       = 0;
    int64_t  encoderInputFramesProcessed = 0;
    int64_t  encoderFramesRecomputed = 0;
    int64_t  encoderMaxWindowFrames  = 0;
    int64_t  encoderPeakContextFrames= 0;
    int64_t  encoderContextFramesRetained = 0;
    int64_t  encoderStateBytes       = 0;
    int64_t  encoderOutputAccumulatedBytes = 0;
    std::string encoderSha;
    double   encoderMaxAbsDeltaVsBatch = 0.0;
    double   encoderMaxAbsDeltaVsManual = 0.0;
    double   encoderManualMeanAbsDelta   = 0.0;
    double   encoderManualRmsDelta       = 0.0;
    double   encoderManualReferenceRms   = 0.0;
    double   encoderManualCosineSimilarity = 0.0;
    std::string encoderManualSha;
    bool     manualOracleChecked         = false;
    bool     fullMelReencodedAtFinish  = false;
    voxtral_encoder_metrics encMetrics;   // full strategy + KV work/memory instrumentation

    // Per-feed latency (data feeds only): incremental Mel + encoder work per feed.
    int64_t  dataFeeds            = 0;
    double   feedLatencyMeanMs    = 0.0;
    double   feedLatencyP50Ms     = 0.0;
    double   feedLatencyP95Ms     = 0.0;
    double   feedLatencyMaxMs     = 0.0;
    double   feedLatencyWarmMaxMs = 0.0;   // max excluding warmup (first few feeds)
    double   finishLatencyMs      = 0.0;

    // Scheduler shape/work and residence telemetry.
    std::string encoderScheduler;
    int32_t encoderLogicalBatchFrames = 0;
    int32_t encoderPhysicalQueryRows = 0;
    int64_t encoderLogicalFramesSubmitted = 0;
    int64_t encoderPhysicalQueryRowsEvaluated = 0;
    int64_t encoderPaddingRowsEvaluated = 0;
    double encoderPhysicalOverheadRatio = 0.0;
    double encoderFirstFrameAbsoluteMs = 0.0;
    double encoderFirstFrameResidenceMs = 0.0;
    double firstMelFrameAbsoluteMs = 0.0;
    double firstAdapterGroupAbsoluteMs = 0.0;
    double firstAdapterGroupResidenceMs = 0.0;
    double firstEightFrameGroupAbsoluteMs = 0.0;
    double firstEightFrameGroupResidenceMs = 0.0;
    double encoderResidenceP50Ms = 0.0;
    double encoderResidenceP95Ms = 0.0;
    double encoderResidenceP99Ms = 0.0;
    double encoderResidenceMaxMs = 0.0;
    double adapterGroupResidenceP50Ms = 0.0;
    double adapterGroupResidenceP95Ms = 0.0;
    double adapterGroupResidenceP99Ms = 0.0;
    double adapterGroupResidenceMaxMs = 0.0;
    double encoderComputeP50Ms = 0.0;
    double encoderComputeP95Ms = 0.0;
    double encoderComputeP99Ms = 0.0;
    double encoderComputeMaxMs = 0.0;
    double encoderComputeWarmMaxMs = 0.0;

    bool pacedRealtime = false;
    int32_t paceChunkMs = 0;
    double audioDurationMs = 0.0;
    double wallDurationMs = 0.0;
    double realtimeFactor = 0.0;
    double feedStartLatenessP50Ms = 0.0;
    double feedStartLatenessP95Ms = 0.0;
    double feedStartLatenessMaxMs = 0.0;
    double feedFinishLatenessP50Ms = 0.0;
    double feedFinishLatenessP95Ms = 0.0;
    double feedFinishLatenessMaxMs = 0.0;
    double backlogP50Ms = 0.0;
    double backlogP95Ms = 0.0;
    double backlogP99Ms = 0.0;
    double backlogMaxMs = 0.0;
    double finalBacklogMs = 0.0;
    double backlogGrowthSlopeMsPerSec = 0.0;
    double finishFrontendMs = 0.0;
    double finishEncoderMs = 0.0;
    double finishDecoderMs = 0.0;
    int64_t decoderKvAllocatedBytes = 0;
    int64_t modelLoadedVramBytes = 0;
    int64_t streamIdleVramBytes = 0;
    int64_t afterFinishVramBytes = 0;
    int64_t afterDestroyVramBytes = 0;
    bool parityChecked = true;

    // Session 7: device-resident incremental adapter + decoder.
    bool     usesIncrementalDecode = false;
    int64_t  adapterGroupsCommitted = 0;
    int64_t  adapterCommitCalls = 0;
    int64_t  decoderSteps = 0;
    int64_t  decoderTokensEmitted = 0;
    int64_t  decoderPosition = 0;
    bool     decoderPrefillComplete = false;
    int64_t  tokensBeforeFinish = 0;
    int64_t  tokensFlushedAtFinish = 0;
    double   firstAdapterCommitMs = 0.0;
    double   firstDecoderStepMs = 0.0;
    double   firstTokenMs = 0.0;
    double   firstVisibleTextMs = 0.0;
    int64_t  adapterInputD2hBytes = 0;
    int64_t  adapterOutputD2hBytes = 0;
    int64_t  logitsD2hBytes = 0;
    int64_t  tokenIdD2hBytes = 0;
    uint64_t partialTextRevision = 0;
};

struct DriveOptions {
    bool paced = false;
    int32_t pace_chunk_ms = 0;
    bool check_parity = true;
    bool check_manual_oracle = false;
};

double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const double idx = p * (double) (v.size() - 1);
    const size_t lo = (size_t) idx;
    const size_t hi = std::min(lo + 1, v.size() - 1);
    const double frac = idx - (double) lo;
    return v[lo] + (v[hi] - v[lo]) * frac;
}

int64_t read_vram_used_bytes() {
    // Linux/RADV acceptance instrumentation. Other backends/platforms simply
    // report zero; this never participates in runtime scheduling.
    std::ifstream in("/sys/class/drm/card1/device/mem_info_vram_used");
    int64_t value = 0;
    if (in) in >> value;
    return value;
}

// Direct batch-vs-incremental encoder tensor parity: recompute the batch encoder
// output over the SAME even-trimmed Mel the stream used (voxtral_stream_mel_data)
// and return the max abs delta against the stream's accumulated incremental
// encoder output, over the frames the adapter actually consumes (a multiple of
// the downsample factor). -1 if unavailable.
double encoder_delta_vs_batch(const voxtral_context * cctx,
                              const float * mel, int32_t mel_frames,
                              const float * inc_enc, int32_t inc_frames) {
    if (!cctx || !mel || mel_frames <= 0 || !inc_enc || inc_frames <= 0) return -1.0;
    voxtral_context * ctx = const_cast<voxtral_context *>(cctx);
    std::vector<float> batch_enc;
    int32_t batch_frames = 0;
    if (!voxtral_encode_mel_batch_internal(*ctx, mel, mel_frames, batch_enc, batch_frames)) return -1.0;
    if (batch_frames <= 0) return -1.0;
    // The adapter consumes a multiple of 4 enc frames; compare that many.
    const int32_t inc_used = (inc_frames / 4) * 4;
    if (batch_frames != inc_used) return 1e9;   // frame-count mismatch surfaces loudly
    double md = 0.0;
    const int64_t n = (int64_t) batch_frames * VOXTRAL_ENC_DIM;
    for (int64_t i = 0; i < n; ++i) {
        const double d = std::fabs((double) batch_enc[(size_t) i] - (double) inc_enc[(size_t) i]);
        if (d > md) md = d;
    }
    if (getenv("VOXTRAL_ENC_KV_DIAG")) {
        auto frame_delta = [&](int32_t f) {
            double fm = 0.0;
            for (int32_t c = 0; c < VOXTRAL_ENC_DIM; ++c) {
                const size_t i = (size_t) f * VOXTRAL_ENC_DIM + c;
                fm = std::max(fm, std::fabs((double) batch_enc[i] - (double) inc_enc[i]));
            }
            return fm;
        };
        int first_bad = -1, bad_frames = 0;
        for (int32_t f = 0; f < batch_frames; ++f) if (frame_delta(f) > 1e-4) { if (first_bad<0) first_bad=f; ++bad_frames; }
        std::fprintf(stderr, "[ENC_KV_DIAG] frames=%d firstBad=%d badFrames=%d maxDelta=%.6g\n",
                     batch_frames, first_bad, bad_frames, md);
        std::fprintf(stderr, "[ENC_KV_DIAG] curve:");
        for (int32_t f : {0, 1, 2, 4, 8, 16, 32, 64, 100, 150, 200, 250, 300, 303, 304, 320, 350, 375}) {
            if (f < batch_frames) std::fprintf(stderr, " f%d=%.2e", f, frame_delta(f));
        }
        std::fprintf(stderr, "\n");
    }
    return md;
}

// Independent attention oracle: the stream has already produced its tensor with
// fused flash attention. Re-run only the global encoder with the explicit
// softmax(QK^T+mask)V implementation and compare host tensors. This is an
// opt-in harness operation; production code never mutates process environment.
struct TensorComparison {
    bool ok = false;
    double max_abs = 0.0;
    double mean_abs = 0.0;
    double rms_delta = 0.0;
    double reference_rms = 0.0;
    double cosine = 0.0;
    std::string reference_sha;
};

TensorComparison encoder_delta_vs_manual(const voxtral_context * cctx,
                                          const float * mel, int32_t mel_frames,
                                          const float * inc_enc, int32_t inc_frames) {
    TensorComparison cmp;
    if (!cctx || !mel || mel_frames <= 0 || !inc_enc || inc_frames <= 0) return cmp;
    const char * old = std::getenv("VOXTRAL_ENC_KV_MANUAL");
    const bool had_old = old != nullptr;
    const std::string old_value = old ? old : "";
    setenv("VOXTRAL_ENC_KV_MANUAL", "1", 1);
    voxtral_context * ctx = const_cast<voxtral_context *>(cctx);
    std::vector<float> manual_enc;
    int32_t manual_frames = 0;
    const bool encoded = voxtral_encode_mel_batch_internal(*ctx, mel, mel_frames, manual_enc, manual_frames);
    if (had_old) setenv("VOXTRAL_ENC_KV_MANUAL", old_value.c_str(), 1);
    else unsetenv("VOXTRAL_ENC_KV_MANUAL");
    const int32_t inc_used = (inc_frames / 4) * 4;
    if (!encoded || manual_frames != inc_used) return cmp;

    const size_t n = (size_t) manual_frames * VOXTRAL_ENC_DIM;
    long double sum_abs = 0.0L, sum_sq = 0.0L, ref_sq = 0.0L;
    long double got_sq = 0.0L, dot = 0.0L;
    for (size_t i = 0; i < n; ++i) {
        const double ref = manual_enc[i];
        const double got = inc_enc[i];
        const double d = got - ref;
        cmp.max_abs = std::max(cmp.max_abs, std::fabs(d));
        sum_abs += std::fabs(d);
        sum_sq += d * d;
        ref_sq += ref * ref;
        got_sq += got * got;
        dot += got * ref;
    }
    cmp.mean_abs = (double) (sum_abs / (long double) n);
    cmp.rms_delta = std::sqrt((double) (sum_sq / (long double) n));
    cmp.reference_rms = std::sqrt((double) (ref_sq / (long double) n));
    const long double denom = std::sqrt(got_sq * ref_sq);
    cmp.cosine = denom > 0.0L ? (double) (dot / denom) : 0.0;
    Sha256 sha;
    sha.update(manual_enc.data(), manual_enc.size() * sizeof(float));
    cmp.reference_sha = sha.hex();
    cmp.ok = true;
    return cmp;
}

// Reconstruct the batch even-trimmed Mel of the fully padded audio (the exact
// reference the incremental frontend must reproduce) so the harness can report a
// true batch-vs-incremental max abs delta. Uses the model's own Hann / mel
// filters via the context so the reference is apples-to-apples.
double mel_delta_vs_batch(const voxtral_context * ctx,
                          const std::vector<int16_t> & pcm16,
                          const float * inc_mel, int32_t inc_frames) {
    const float * hann = voxtral_ctx_hann_window(ctx);
    const float * filt = voxtral_ctx_mel_filters(ctx);
    if (!ctx || !hann || !filt || !inc_mel || inc_frames <= 0) return -1.0;

    const int32_t n_mel   = VOXTRAL_MEL_N_MEL;
    const int64_t n_raw   = (int64_t) pcm16.size();
    const int64_t mult    = VOXTRAL_RAW_AUDIO_LENGTH_PER_TOK;
    const int64_t left    = (int64_t) VOXTRAL_N_LEFT_PAD_TOKENS * mult;
    const int64_t align   = (mult - (n_raw % mult)) % mult;
    const int64_t right   = align + (int64_t) VOXTRAL_N_RIGHT_PAD_TOKENS * mult;

    std::vector<float> padded((size_t) (left + n_raw + right), 0.0f);
    for (int64_t i = 0; i < n_raw; ++i) {
        padded[(size_t) (left + i)] = (float) pcm16[(size_t) i] / 32768.0f;
    }

    const int32_t nf_full = voxtral_mel_batch_frame_count((int32_t) padded.size());
    std::vector<float> mel((size_t) n_mel * std::max(0, nf_full), 0.0f);
    int32_t nf = 0;
    voxtral_mel_compute_batch(padded.data(), (int32_t) padded.size(), filt, hann, mel.data(), &nf);

    // Even trim (drop first frame if odd), matching compute_mel_even.
    const bool drop_first = (nf % 2 != 0);
    const int32_t n_out = drop_first ? (nf - 1) : nf;
    if (n_out != inc_frames) return 1e9;   // frame-count mismatch surfaces loudly
    const int32_t src_off = drop_first ? 1 : 0;

    double md = 0.0;
    for (int32_t m = 0; m < n_mel; ++m) {
        for (int32_t j = 0; j < n_out; ++j) {
            const double ref = mel[(size_t) m * nf + (j + src_off)];
            const double got = inc_mel[(size_t) m * n_out + j];
            const double d = std::fabs(ref - got);
            if (d > md) md = d;
        }
    }
    return md;
}

// Feed `counts` into an already-created stream, finish it and capture results.
StreamRun drive_stream(voxtral_stream * stream,
                       const std::vector<int16_t> & pcm16,
                       const std::vector<size_t> & counts,
                       const DriveOptions & options = {}) {
    StreamRun r;
    r.parityChecked = options.check_parity;

    const auto wall_start = std::chrono::steady_clock::now();
    if (options.paced) {
        const int64_t start_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            wall_start.time_since_epoch()).count();
        // Anchor absolute timestamps to capture start, not to the first
        // (possibly 80/160 ms-late) block delivery. Arrival timestamps remain
        // the actual feed times, so residence measures delivery latency.
        voxtral_stream_set_timeline_start_internal(stream, start_ns);
    }
    size_t off = 0;
    uint64_t feed_calls = 0;
    std::vector<double> feed_ms;
    std::vector<double> start_lateness_ms;
    std::vector<double> finish_lateness_ms;
    std::vector<double> backlog_ms;
    std::vector<double> backlog_audio_s;
    uint64_t audio_cursor = 0;
    voxtral_status fst = voxtral_status::ok;
    for (size_t c : counts) {
        // A captured chunk becomes callable only when its final sample has
        // arrived. Anchor every deadline to the common monotonic start so
        // scheduler/OS delays never accumulate as sleep drift.
        const auto end_deadline = wall_start + std::chrono::microseconds(
            (long long) ((double) (audio_cursor + c) * 1'000'000.0 / VOXTRAL_SAMPLE_RATE));
        if (options.paced) std::this_thread::sleep_until(end_deadline);
        const auto before_feed = std::chrono::steady_clock::now();
        if (options.paced) {
            start_lateness_ms.push_back(std::max(0.0,
                std::chrono::duration<double, std::milli>(before_feed - end_deadline).count()));
        }
        const int16_t * ptr = (c == 0) ? nullptr : (pcm16.data() + off);
        const auto tf0 = before_feed;
        fst = voxtral_stream_feed_pcm16_internal(stream, ptr, c);
        const auto tf1 = std::chrono::steady_clock::now();
        ++feed_calls;
        if (c > 0) feed_ms.push_back(std::chrono::duration<double, std::milli>(tf1 - tf0).count());
        if (options.paced) {
            const double late = std::max(0.0,
                std::chrono::duration<double, std::milli>(tf1 - end_deadline).count());
            finish_lateness_ms.push_back(late);
            // `late` is residence of this delivered payload after its capture
            // deadline. Realtime backlog is the part that survives until the
            // NEXT capture deadline; work finishing inside one chunk cadence
            // overlaps capture of the next chunk and does not accumulate.
            const double cadence_ms = options.pace_chunk_ms > 0
                ? (double) options.pace_chunk_ms
                : (double) c * 1000.0 / VOXTRAL_SAMPLE_RATE;
            backlog_ms.push_back(std::max(0.0, late - cadence_ms));
            backlog_audio_s.push_back((double) (audio_cursor + c) / VOXTRAL_SAMPLE_RATE);
        }
        if (fst != voxtral_status::ok) {
            std::cerr << "feed failed: " << voxtral_stream_status_name(fst)
                      << " (" << voxtral_stream_last_error(stream) << ")\n";
            break;
        }
        off += c;
        audio_cursor += c;
        // Drain events during feed (realistic consumer), so the bounded event queue
        // never overflows on a long incremental stream and no TOKEN is lost.
        { voxtral_stream_event dev; while (voxtral_stream_poll_event(stream, dev)) r.events.push_back(dev); }
    }

    voxtral_status finst = voxtral_status::internal_error;
    if (fst == voxtral_status::ok) {
        const auto tfin0 = std::chrono::steady_clock::now();
        finst = voxtral_stream_finish_internal(stream);
        const auto tfin1 = std::chrono::steady_clock::now();
        r.finishLatencyMs = std::chrono::duration<double, std::milli>(tfin1 - tfin0).count();
        if (finst != voxtral_status::ok) {
            std::cerr << "finish failed: " << voxtral_stream_status_name(finst)
                      << " (" << voxtral_stream_last_error(stream) << ")\n";
        }
    }

    r.dataFeeds = (int64_t) feed_ms.size();
    if (!feed_ms.empty()) {
        double sum = 0.0;
        for (double x : feed_ms) sum += x;
        r.feedLatencyMeanMs = sum / (double) feed_ms.size();
        r.feedLatencyP50Ms  = percentile(feed_ms, 0.50);
        r.feedLatencyP95Ms  = percentile(feed_ms, 0.95);
        r.feedLatencyMaxMs  = percentile(feed_ms, 1.00);
        // Warmup-excluded max: drop the first few feeds (Vulkan pipeline/shader
        // creation happens on the first encoder graph execution).
        std::vector<double> warm(feed_ms.begin() + std::min<size_t>(feed_ms.size(), 3), feed_ms.end());
        r.feedLatencyWarmMaxMs = warm.empty() ? r.feedLatencyMaxMs : percentile(warm, 1.00);
    }

    r.pacedRealtime = options.paced;
    r.paceChunkMs = options.pace_chunk_ms;
    r.audioDurationMs = (double) pcm16.size() * 1000.0 / VOXTRAL_SAMPLE_RATE;
    r.wallDurationMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - wall_start).count();
    r.realtimeFactor = r.audioDurationMs > 0.0 ? r.wallDurationMs / r.audioDurationMs : 0.0;
    if (!start_lateness_ms.empty()) {
        r.feedStartLatenessP50Ms = percentile(start_lateness_ms, 0.50);
        r.feedStartLatenessP95Ms = percentile(start_lateness_ms, 0.95);
        r.feedStartLatenessMaxMs = percentile(start_lateness_ms, 1.00);
        r.feedFinishLatenessP50Ms = percentile(finish_lateness_ms, 0.50);
        r.feedFinishLatenessP95Ms = percentile(finish_lateness_ms, 0.95);
        r.feedFinishLatenessMaxMs = percentile(finish_lateness_ms, 1.00);
        r.backlogP50Ms = percentile(backlog_ms, 0.50);
        r.backlogP95Ms = percentile(backlog_ms, 0.95);
        r.backlogP99Ms = percentile(backlog_ms, 0.99);
        r.backlogMaxMs = percentile(backlog_ms, 1.00);
        r.finalBacklogMs = backlog_ms.back();
        if (backlog_ms.size() > 1 && backlog_audio_s.size() == backlog_ms.size()) {
            // Least-squares slope over the whole run, not a fragile first/last
            // difference. A sustained realtime deficit therefore remains
            // visible even when one boundary feed is an outlier.
            double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
            for (size_t i = 0; i < backlog_ms.size(); ++i) {
                sx += backlog_audio_s[i];
                sy += backlog_ms[i];
                sxx += backlog_audio_s[i] * backlog_audio_s[i];
                sxy += backlog_audio_s[i] * backlog_ms[i];
            }
            const double n = (double) backlog_ms.size();
            const double denom = n * sxx - sx * sx;
            if (denom > 0.0) r.backlogGrowthSlopeMsPerSec = (n * sxy - sx * sy) / denom;
        }
    }

    // Chunk-invariant SHA-256 of the full canonical PCM (maintained incrementally
    // by the stream; no full-PCM retention needed).
    r.pcmSha = voxtral_stream_pcm_sha256(stream);

    // Incremental Mel frontend evidence.
    r.incrementalMel           = voxtral_stream_uses_incremental_mel(stream);
    r.melFrames                = voxtral_stream_mel_frames(stream);
    r.melFramesBeforeFinish    = voxtral_stream_mel_frames_before_finish(stream);
    r.melFramesFlushedAtFinish = voxtral_stream_mel_frames_flushed_at_finish(stream);
    r.dftFramesComputed        = voxtral_stream_dft_frames_computed(stream);
    r.pcmRetainedSamples       = voxtral_stream_pcm_retained_samples(stream);
    r.pcmPeakRetainedSamples   = voxtral_stream_pcm_peak_retained_samples(stream);
    r.pcmBaseSample            = voxtral_stream_pcm_base_sample(stream);
    r.fullPcmBufferedAtFinish  = voxtral_stream_full_pcm_buffered_at_finish(stream);
    r.melHistoryRetained      = voxtral_stream_mel_history_retained(stream);

    // SHA-256 over the assembled even Mel matrix (byte layout: channel-major
    // [n_mel, n_frames] float32) + max abs delta vs the batch Mel of the same
    // audio, both while the stream (and its context) is still alive.
    const float * inc_mel   = voxtral_stream_mel_data(stream);
    const int32_t inc_frames = voxtral_stream_mel_data_frames(stream);
    {
        Sha256 msha;
        if (inc_mel && inc_frames > 0) {
            msha.update(inc_mel, (size_t) VOXTRAL_MEL_N_MEL * (size_t) inc_frames * sizeof(float));
        }
        r.melSha = msha.hex();
    }
    if (options.check_parity) {
        const voxtral_context * ctx =
            static_cast<const voxtral_context *>(voxtral_stream_context_ptr(stream));
        const double d = mel_delta_vs_batch(ctx, pcm16, inc_mel, inc_frames);
        r.melMaxAbsDeltaVsBatch = (d < 0.0) ? 0.0 : d;
    }

    // Incremental causal encoder evidence.
    r.incrementalEncoder            = voxtral_stream_uses_incremental_encoder(stream);
    r.encoderFrames                 = voxtral_stream_encoder_frames(stream);
    r.encoderFramesBeforeFinish     = voxtral_stream_encoder_frames_before_finish(stream);
    r.encoderFramesFlushedAtFinish  = voxtral_stream_encoder_frames_flushed_at_finish(stream);
    r.encoderExecutions             = voxtral_stream_encoder_executions(stream);
    r.encoderInputFramesProcessed   = voxtral_stream_encoder_input_frames_processed(stream);
    r.encoderFramesRecomputed       = voxtral_stream_encoder_frames_recomputed(stream);
    r.encoderMaxWindowFrames        = voxtral_stream_encoder_max_window_frames(stream);
    r.encoderPeakContextFrames      = voxtral_stream_encoder_peak_context_frames(stream);
    r.encoderContextFramesRetained  = voxtral_stream_encoder_context_frames_retained(stream);
    r.encoderStateBytes             = voxtral_stream_encoder_state_bytes(stream);
    r.encoderOutputAccumulatedBytes = voxtral_stream_encoder_output_accumulated_bytes(stream);
    r.encMetrics                    = voxtral_stream_encoder_metrics_full(stream);
    r.encoderScheduler              = r.encMetrics.encoderScheduler ? r.encMetrics.encoderScheduler : "static";
    r.encoderLogicalBatchFrames    = r.encMetrics.encoderLogicalBatchFrames;
    r.encoderPhysicalQueryRows     = r.encMetrics.encoderPhysicalQueryRows;
    r.encoderLogicalFramesSubmitted= r.encMetrics.encoderLogicalFramesSubmitted;
    r.encoderPhysicalQueryRowsEvaluated = r.encMetrics.encoderPhysicalQueryRowsEvaluated;
    r.encoderPaddingRowsEvaluated   = r.encMetrics.encoderPaddingRowsEvaluated;
    r.encoderPhysicalOverheadRatio  = r.encMetrics.encoderUniqueFrames > 0
        ? (double) r.encoderPhysicalQueryRowsEvaluated / (double) r.encMetrics.encoderUniqueFrames : 0.0;
    r.encoderFirstFrameAbsoluteMs  = r.encMetrics.encoderFirstFrameAbsoluteMs;
    r.encoderFirstFrameResidenceMs = r.encMetrics.encoderFirstFrameResidenceMs;
    r.firstMelFrameAbsoluteMs      = voxtral_stream_first_mel_absolute_ms(stream);
    r.firstAdapterGroupAbsoluteMs  = r.encMetrics.firstAdapterGroupAbsoluteMs;
    r.firstAdapterGroupResidenceMs = r.encMetrics.firstAdapterGroupResidenceMs;
    r.firstEightFrameGroupAbsoluteMs = r.encMetrics.firstEightFrameGroupAbsoluteMs;
    r.firstEightFrameGroupResidenceMs = r.encMetrics.firstEightFrameGroupResidenceMs;
    r.encoderResidenceP50Ms        = r.encMetrics.encoderResidenceP50Ms;
    r.encoderResidenceP95Ms        = r.encMetrics.encoderResidenceP95Ms;
    r.encoderResidenceP99Ms        = r.encMetrics.encoderResidenceP99Ms;
    r.encoderResidenceMaxMs        = r.encMetrics.encoderResidenceMaxMs;
    r.adapterGroupResidenceP50Ms   = r.encMetrics.adapterGroupResidenceP50Ms;
    r.adapterGroupResidenceP95Ms   = r.encMetrics.adapterGroupResidenceP95Ms;
    r.adapterGroupResidenceP99Ms   = r.encMetrics.adapterGroupResidenceP99Ms;
    r.adapterGroupResidenceMaxMs   = r.encMetrics.adapterGroupResidenceMaxMs;
    r.encoderComputeP50Ms          = r.encMetrics.encoderComputeP50Ms;
    r.encoderComputeP95Ms          = r.encMetrics.encoderComputeP95Ms;
    r.encoderComputeP99Ms          = r.encMetrics.encoderComputeP99Ms;
    r.encoderComputeMaxMs          = r.encMetrics.encoderComputeMaxMs;
    r.encoderComputeWarmMaxMs      = r.encMetrics.encoderComputeWarmMaxMs;
    r.finishFrontendMs             = voxtral_stream_finish_frontend_ms(stream);
    r.finishEncoderMs              = voxtral_stream_finish_encoder_ms(stream);
    r.finishDecoderMs              = voxtral_stream_finish_decoder_ms(stream);
    r.decoderKvAllocatedBytes      = voxtral_stream_decoder_kv_allocated_bytes(stream);
    r.fullMelReencodedAtFinish      = false;   // finish runs at most the last 1-2 encoder chunks

    const float * inc_enc    = voxtral_stream_encoder_output_data(stream);
    const int32_t inc_enc_frames = voxtral_stream_encoder_output_frames_count(stream);
    {
        Sha256 esha;
        if (inc_enc && inc_enc_frames > 0) {
            const int32_t used = (inc_enc_frames / 4) * 4;
            esha.update(inc_enc, (size_t) used * VOXTRAL_ENC_DIM * sizeof(float));
        }
        r.encoderSha = esha.hex();
    }
    if (options.check_parity) {
        const voxtral_context * ctx =
            static_cast<const voxtral_context *>(voxtral_stream_context_ptr(stream));
        const double d = encoder_delta_vs_batch(ctx, inc_mel, inc_frames, inc_enc, inc_enc_frames);
        r.encoderMaxAbsDeltaVsBatch = (d < 0.0) ? 0.0 : d;
    }
    if (options.check_manual_oracle) {
        const voxtral_context * ctx =
            static_cast<const voxtral_context *>(voxtral_stream_context_ptr(stream));
        const TensorComparison cmp = encoder_delta_vs_manual(ctx, inc_mel, inc_frames, inc_enc, inc_enc_frames);
        r.encoderMaxAbsDeltaVsManual = cmp.max_abs;
        r.encoderManualMeanAbsDelta = cmp.mean_abs;
        r.encoderManualRmsDelta = cmp.rms_delta;
        r.encoderManualReferenceRms = cmp.reference_rms;
        r.encoderManualCosineSimilarity = cmp.cosine;
        r.encoderManualSha = cmp.reference_sha;
        r.manualOracleChecked = cmp.ok;
    }

    // Drain events (order preserved).
    voxtral_stream_event ev;
    while (voxtral_stream_poll_event(stream, ev)) r.events.push_back(ev);

    r.state           = voxtral_stream_state_name(voxtral_stream_get_state(stream));
    r.finishStatus    = voxtral_stream_status_name(finst);
    r.samplesReceived = voxtral_stream_samples_received(stream);
    r.samplesConsumed = voxtral_stream_samples_consumed(stream);
    r.feedCalls       = feed_calls;
    r.inferenceRuns   = voxtral_stream_inference_runs(stream);
    r.pcmFloats       = voxtral_stream_samples_received(stream);
    r.tokens          = voxtral_stream_tokens(stream);
    r.text            = voxtral_stream_transcript(stream);
    r.contextOwned    = voxtral_stream_owns_context(stream);
    r.ok              = (finst == voxtral_status::ok);

    // Session 7 incremental adapter/decoder telemetry.
    r.usesIncrementalDecode   = voxtral_stream_uses_incremental_decode(stream);
    r.adapterGroupsCommitted  = voxtral_stream_adapter_groups_committed(stream);
    r.adapterCommitCalls      = voxtral_stream_adapter_commit_calls(stream);
    r.decoderSteps            = voxtral_stream_decoder_steps(stream);
    r.decoderTokensEmitted    = voxtral_stream_decoder_tokens_emitted(stream);
    r.decoderPosition         = voxtral_stream_decoder_position(stream);
    r.decoderPrefillComplete  = voxtral_stream_decoder_prefill_complete(stream);
    r.tokensBeforeFinish      = voxtral_stream_tokens_before_finish(stream);
    r.tokensFlushedAtFinish   = voxtral_stream_tokens_flushed_at_finish(stream);
    r.firstAdapterCommitMs    = voxtral_stream_first_adapter_commit_ms(stream);
    r.firstDecoderStepMs      = voxtral_stream_first_decoder_step_ms(stream);
    r.firstTokenMs            = voxtral_stream_first_token_ms(stream);
    r.firstVisibleTextMs      = voxtral_stream_first_visible_text_ms(stream);
    r.adapterInputD2hBytes    = voxtral_stream_adapter_input_d2h_bytes(stream);
    r.adapterOutputD2hBytes   = voxtral_stream_adapter_output_d2h_bytes(stream);
    r.logitsD2hBytes          = voxtral_stream_logits_d2h_bytes(stream);
    r.tokenIdD2hBytes         = voxtral_stream_token_id_d2h_bytes(stream);
    r.partialTextRevision     = voxtral_stream_partial_text_revision(stream);
    return r;
}

// Emit a StreamRun's fields into an open JSON object body (no surrounding braces).
void write_run_fields(std::ostringstream & js, const StreamRun & r) {
    js << "\"state\":\"" << r.state << "\",";
    js << "\"finishStatus\":\"" << r.finishStatus << "\",";
    js << "\"samplesReceived\":" << r.samplesReceived << ",";
    js << "\"samplesConsumed\":" << r.samplesConsumed << ",";
    js << "\"feedCalls\":" << r.feedCalls << ",";
    js << "\"inferenceRuns\":" << r.inferenceRuns << ",";
    js << "\"pcmFloats\":" << r.pcmFloats << ",";
    js << "\"pcmSha256\":\"" << r.pcmSha << "\",";
    js << "\"incrementalMel\":" << (r.incrementalMel ? "true" : "false") << ",";
    js << "\"melFrames\":" << r.melFrames << ",";
    js << "\"melFramesBeforeFinish\":" << r.melFramesBeforeFinish << ",";
    js << "\"melFramesFlushedAtFinish\":" << r.melFramesFlushedAtFinish << ",";
    js << "\"dftFramesComputed\":" << r.dftFramesComputed << ",";
    js << "\"melSha256\":\"" << r.melSha << "\",";
    js << "\"melMaxAbsDeltaVsBatch\":" << r.melMaxAbsDeltaVsBatch << ",";
    js << "\"pcmRetainedSamples\":" << r.pcmRetainedSamples << ",";
    js << "\"pcmPeakRetainedSamples\":" << r.pcmPeakRetainedSamples << ",";
    js << "\"pcmBaseSample\":" << r.pcmBaseSample << ",";
    js << "\"fullPcmBufferedAtFinish\":" << (r.fullPcmBufferedAtFinish ? "true" : "false") << ",";
    js << "\"melHistoryRetained\":" << (r.melHistoryRetained ? "true" : "false") << ",";
    // --- Incremental causal encoder ---
    const voxtral_encoder_metrics & em = r.encMetrics;
    const double workRatio = em.encoderUniqueFrames > 0
        ? (double) em.encoderTransformerFramesComputed / (double) em.encoderUniqueFrames : 0.0;
    js << "\"incrementalEncoder\":" << (r.incrementalEncoder ? "true" : "false") << ",";
    js << "\"encoderStrategy\":\"" << (em.strategy ? em.strategy : "per-layer-kv") << "\",";
    js << "\"encoderScheduler\":\"" << (em.encoderScheduler ? em.encoderScheduler : "static") << "\",";
    js << "\"encoderLogicalBatchFrames\":" << em.encoderLogicalBatchFrames << ",";
    js << "\"encoderPhysicalQueryRows\":" << em.encoderPhysicalQueryRows << ",";
    js << "\"encoderLogicalFramesSubmitted\":" << em.encoderLogicalFramesSubmitted << ",";
    js << "\"encoderPhysicalQueryRowsEvaluated\":" << em.encoderPhysicalQueryRowsEvaluated << ",";
    js << "\"encoderPhysicalRowsEvaluated\":" << em.encoderPhysicalQueryRowsEvaluated << ",";
    js << "\"encoderPaddingRowsEvaluated\":" << em.encoderPaddingRowsEvaluated << ",";
    const double physicalOverhead = em.encoderUniqueFrames > 0
        ? (double) em.encoderPhysicalQueryRowsEvaluated / (double) em.encoderUniqueFrames : 0.0;
    js << "\"encoderPhysicalOverheadRatio\":" << physicalOverhead << ",";
    js << "\"encoderPaddingOverheadRatio\":" << physicalOverhead << ",";
    js << "\"referenceStrategyAvailable\":true,";
    // Per-layer KV work instrumentation (Stage 16).
    js << "\"encoderUniqueFrames\":" << em.encoderUniqueFrames << ",";
    js << "\"encoderTransformerFramesComputed\":" << em.encoderTransformerFramesComputed << ",";
    js << "\"encoderTransformerRealFramesComputed\":" << em.encoderTransformerFramesComputed << ",";
    js << "\"encoderRealFramesComputed\":" << em.encoderTransformerFramesComputed << ",";
    js << "\"encoderFrameLayerEvaluations\":" << em.encoderFrameLayerEvaluations << ",";
    js << "\"encoderWorkRatio\":" << workRatio << ",";
    js << "\"encoderKvAppends\":" << em.encoderKvAppends << ",";
    js << "\"encoderKvEvictions\":" << em.encoderKvEvictions << ",";
    js << "\"encoderKvWraps\":" << em.encoderKvWraps << ",";
    js << "\"encoderKvMaterializedFrames\":" << em.encoderKvMaterializedFrames << ",";
    js << "\"encoderGraphExecutions\":" << em.encoderGraphExecutions << ",";
    js << "\"encoderMaxNewFramesPerExecution\":" << em.encoderMaxNewFramesPerExecution << ",";
    js << "\"encoderWarmupFrames\":" << em.encoderWarmupFrames << ",";
    js << "\"encoderFramesComputedDuringFinish\":" << em.encoderFramesComputedDuringFinish << ",";
    // Per-layer KV memory instrumentation (Stage 17).
    js << "\"encoderKvAllocatedBytes\":" << em.encoderKvAllocatedBytes << ",";
    js << "\"encoderKvLogicalFrames\":" << em.encoderKvLogicalFrames << ",";
    js << "\"encoderKvPeakLogicalFrames\":" << em.encoderKvPeakLogicalFrames << ",";
    js << "\"encoderKvCapacityFrames\":" << em.encoderKvCapacityFrames << ",";
    js << "\"encoderKvElementSize\":" << em.encoderKvElementSize << ",";
    js << "\"encoderMelRetainedFrames\":" << em.encoderMelRetainedFrames << ",";
    js << "\"encoderMelPeakRetainedFrames\":" << em.encoderMelPeakRetainedFrames << ",";
    js << "\"encoderMelRetainedBytes\":" << em.encoderMelRetainedBytes << ",";
    js << "\"encoderOutputQueuedFrames\":" << em.encoderOutputQueuedFrames << ",";
    js << "\"encoderOutputPeakQueuedFrames\":" << em.encoderOutputPeakQueuedFrames << ",";
    js << "\"encoderFrames\":" << r.encoderFrames << ",";
    js << "\"encoderFramesBeforeFinish\":" << r.encoderFramesBeforeFinish << ",";
    js << "\"encoderFramesFlushedAtFinish\":" << r.encoderFramesFlushedAtFinish << ",";
    js << "\"encoderExecutions\":" << r.encoderExecutions << ",";
    js << "\"encoderInputFramesProcessed\":" << r.encoderInputFramesProcessed << ",";
    js << "\"encoderFramesRecomputed\":" << r.encoderFramesRecomputed << ",";
    js << "\"encoderMaxWindowFrames\":" << r.encoderMaxWindowFrames << ",";
    js << "\"encoderPeakContextFrames\":" << r.encoderPeakContextFrames << ",";
    js << "\"encoderContextFramesRetained\":" << r.encoderContextFramesRetained << ",";
    js << "\"encoderStateBytes\":" << r.encoderStateBytes << ",";
    js << "\"encoderOutputAccumulatedBytes\":" << r.encoderOutputAccumulatedBytes << ",";
    js << "\"encoderSha256\":\"" << r.encoderSha << "\",";
    js << "\"encoderMaxAbsDeltaVsBatch\":" << r.encoderMaxAbsDeltaVsBatch << ",";
    js << "\"parityChecked\":" << (r.parityChecked ? "true" : "false") << ",";
    js << "\"encoderMaxAbsDeltaVsManual\":" << r.encoderMaxAbsDeltaVsManual << ",";
    js << "\"encoderManualMeanAbsDelta\":" << r.encoderManualMeanAbsDelta << ",";
    js << "\"encoderManualRmsDelta\":" << r.encoderManualRmsDelta << ",";
    js << "\"encoderManualReferenceRms\":" << r.encoderManualReferenceRms << ",";
    js << "\"encoderManualCosineSimilarity\":" << r.encoderManualCosineSimilarity << ",";
    js << "\"encoderManualSha256\":\"" << r.encoderManualSha << "\",";
    js << "\"manualOracleChecked\":" << (r.manualOracleChecked ? "true" : "false") << ",";
    js << "\"encoderFirstFrameAbsoluteMs\":" << em.encoderFirstFrameAbsoluteMs << ",";
    js << "\"encoderFirstFrameResidenceMs\":" << em.encoderFirstFrameResidenceMs << ",";
    js << "\"firstMelFrameAbsoluteMs\":" << r.firstMelFrameAbsoluteMs << ",";
    js << "\"firstAdapterGroupAbsoluteMs\":" << em.firstAdapterGroupAbsoluteMs << ",";
    js << "\"firstAdapterGroupResidenceMs\":" << em.firstAdapterGroupResidenceMs << ",";
    js << "\"firstEightFrameGroupAbsoluteMs\":" << em.firstEightFrameGroupAbsoluteMs << ",";
    js << "\"firstEightFrameGroupResidenceMs\":" << em.firstEightFrameGroupResidenceMs << ",";
    js << "\"encoderResidenceP50Ms\":" << em.encoderResidenceP50Ms << ",";
    js << "\"encoderResidenceP95Ms\":" << em.encoderResidenceP95Ms << ",";
    js << "\"encoderResidenceP99Ms\":" << em.encoderResidenceP99Ms << ",";
    js << "\"encoderResidenceMaxMs\":" << em.encoderResidenceMaxMs << ",";
    js << "\"adapterGroupResidenceP50Ms\":" << em.adapterGroupResidenceP50Ms << ",";
    js << "\"adapterGroupResidenceP95Ms\":" << em.adapterGroupResidenceP95Ms << ",";
    js << "\"adapterGroupResidenceP99Ms\":" << em.adapterGroupResidenceP99Ms << ",";
    js << "\"adapterGroupResidenceMaxMs\":" << em.adapterGroupResidenceMaxMs << ",";
    js << "\"encoderComputeP50Ms\":" << em.encoderComputeP50Ms << ",";
    js << "\"encoderComputeP95Ms\":" << em.encoderComputeP95Ms << ",";
    js << "\"encoderComputeP99Ms\":" << em.encoderComputeP99Ms << ",";
    js << "\"encoderComputeMaxMs\":" << em.encoderComputeMaxMs << ",";
    js << "\"encoderComputeWarmMaxMs\":" << em.encoderComputeWarmMaxMs << ",";
    js << "\"fullMelReencodedAtFinish\":" << (r.fullMelReencodedAtFinish ? "true" : "false") << ",";
    js << "\"dataFeeds\":" << r.dataFeeds << ",";
    js << "\"feedLatencyMeanMs\":" << r.feedLatencyMeanMs << ",";
    js << "\"feedLatencyP50Ms\":" << r.feedLatencyP50Ms << ",";
    js << "\"feedLatencyP95Ms\":" << r.feedLatencyP95Ms << ",";
    js << "\"feedLatencyMaxMs\":" << r.feedLatencyMaxMs << ",";
    js << "\"feedLatencyWarmMaxMs\":" << r.feedLatencyWarmMaxMs << ",";
    js << "\"finishLatencyMs\":" << r.finishLatencyMs << ",";
    js << "\"finishFrontendMs\":" << r.finishFrontendMs << ",";
    js << "\"finishEncoderMs\":" << r.finishEncoderMs << ",";
    js << "\"finishDecoderMs\":" << r.finishDecoderMs << ",";
    js << "\"frontendFinishMs\":" << r.finishFrontendMs << ",";
    js << "\"encoderFinishMs\":" << r.finishEncoderMs << ",";
    js << "\"decoderFinishMs\":" << r.finishDecoderMs << ",";
    js << "\"decoderKvAllocatedBytes\":" << r.decoderKvAllocatedBytes << ",";
    js << "\"modelLoadedVramBytes\":" << r.modelLoadedVramBytes << ",";
    js << "\"streamIdleVramBytes\":" << r.streamIdleVramBytes << ",";
    js << "\"afterFinishVramBytes\":" << r.afterFinishVramBytes << ",";
    js << "\"afterDestroyVramBytes\":" << r.afterDestroyVramBytes << ",";
    js << "\"pacedRealtime\":" << (r.pacedRealtime ? "true" : "false") << ",";
    js << "\"paceChunkMs\":" << r.paceChunkMs << ",";
    js << "\"audioDurationMs\":" << r.audioDurationMs << ",";
    js << "\"wallDurationMs\":" << r.wallDurationMs << ",";
    js << "\"realtimeFactor\":" << r.realtimeFactor << ",";
    js << "\"feedStartLatenessP50Ms\":" << r.feedStartLatenessP50Ms << ",";
    js << "\"feedStartLatenessP95Ms\":" << r.feedStartLatenessP95Ms << ",";
    js << "\"feedStartLatenessMaxMs\":" << r.feedStartLatenessMaxMs << ",";
    js << "\"feedFinishLatenessP50Ms\":" << r.feedFinishLatenessP50Ms << ",";
    js << "\"feedFinishLatenessP95Ms\":" << r.feedFinishLatenessP95Ms << ",";
    js << "\"feedFinishLatenessMaxMs\":" << r.feedFinishLatenessMaxMs << ",";
    js << "\"backlogP50Ms\":" << r.backlogP50Ms << ",";
    js << "\"backlogP95Ms\":" << r.backlogP95Ms << ",";
    js << "\"backlogP99Ms\":" << r.backlogP99Ms << ",";
    js << "\"backlogMaxMs\":" << r.backlogMaxMs << ",";
    js << "\"finalBacklogMs\":" << r.finalBacklogMs << ",";
    js << "\"backlogGrowthSlopeMsPerSec\":" << r.backlogGrowthSlopeMsPerSec << ",";
    js << "\"contextOwnedByStream\":" << (r.contextOwned ? "true" : "false") << ",";
    // --- Session 7 incremental adapter + decoder ---
    js << "\"usesIncrementalDecode\":" << (r.usesIncrementalDecode ? "true" : "false") << ",";
    js << "\"adapterGroupsCommitted\":" << r.adapterGroupsCommitted << ",";
    js << "\"adapterCommitCalls\":" << r.adapterCommitCalls << ",";
    js << "\"decoderSteps\":" << r.decoderSteps << ",";
    js << "\"decoderTokensEmitted\":" << r.decoderTokensEmitted << ",";
    js << "\"decoderPosition\":" << r.decoderPosition << ",";
    js << "\"decoderPrefillComplete\":" << (r.decoderPrefillComplete ? "true" : "false") << ",";
    js << "\"tokensBeforeFinish\":" << r.tokensBeforeFinish << ",";
    js << "\"tokensFlushedAtFinish\":" << r.tokensFlushedAtFinish << ",";
    js << "\"firstAdapterCommitMs\":" << r.firstAdapterCommitMs << ",";
    js << "\"firstDecoderStepMs\":" << r.firstDecoderStepMs << ",";
    js << "\"firstTokenMs\":" << r.firstTokenMs << ",";
    js << "\"firstVisibleTextMs\":" << r.firstVisibleTextMs << ",";
    js << "\"adapterInputD2hBytes\":" << r.adapterInputD2hBytes << ",";
    js << "\"adapterOutputD2hBytes\":" << r.adapterOutputD2hBytes << ",";
    js << "\"logitsD2hBytes\":" << r.logitsD2hBytes << ",";
    js << "\"tokenIdD2hBytes\":" << r.tokenIdD2hBytes << ",";
    js << "\"partialTextRevision\":" << r.partialTextRevision << ",";
    js << "\"tokens\":[";
    for (size_t i = 0; i < r.tokens.size(); ++i) { if (i) js << ","; js << r.tokens[i]; }
    js << "],";
    js << "\"text\":\"" << json_escape(r.text) << "\",";
    js << "\"events\":[";
    for (size_t i = 0; i < r.events.size(); ++i) {
        if (i) js << ",";
        const voxtral_stream_event & e = r.events[i];
        js << "{\"type\":\"" << voxtral_stream_event_name(e.type) << "\"";
        if (e.type == voxtral_stream_event_type::error) {
            js << ",\"code\":" << e.error_code;
        } else if (e.type == voxtral_stream_event_type::token) {
            js << ",\"sequence\":" << e.sequence
               << ",\"token\":" << e.token
               << ",\"decoderPosition\":" << e.decoder_position
               << ",\"audioEndSample\":" << e.audio_end_sample
               << ",\"special\":" << (e.special ? "true" : "false");
        } else if (e.type == voxtral_stream_event_type::partial_text) {
            js << ",\"revision\":" << e.revision
               << ",\"stablePrefixBytes\":" << e.stable_prefix_bytes
               << ",\"audioEndSample\":" << e.audio_end_sample
               << ",\"utf8Bytes\":" << e.text.size();
        }
        js << "}";
    }
    js << "]";
}

} // namespace

int main(int argc, char ** argv) {
    std::string model_path, wav_path, plan_file, mode = "full";
    voxtral_gpu_backend gpu = voxtral_gpu_backend::auto_detect;
    int32_t max_tokens = 0;
    int32_t realtime_ms = 0;
    bool skip_parity = false;
    bool manual_oracle = false;
    bool ab = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto val = [&](const char * name) -> const char * {
            if (i + 1 >= argc) { std::cerr << "missing value for " << name << "\n"; return nullptr; }
            return argv[++i];
        };
        if (a == "--model")            { const char * v = val("--model"); if (!v) return 2; model_path = v; }
        else if (a == "--wav")         { const char * v = val("--wav"); if (!v) return 2; wav_path = v; }
        else if (a == "--plan-file")   { const char * v = val("--plan-file"); if (!v) return 2; plan_file = v; }
        else if (a == "--mode")        { const char * v = val("--mode"); if (!v) return 2; mode = v; }
        else if (a == "--max-tokens")  { const char * v = val("--max-tokens"); if (!v) return 2; max_tokens = int32_t(std::strtol(v, nullptr, 10)); }
        else if (a == "--realtime-ms") { const char * v = val("--realtime-ms"); if (!v) return 2; realtime_ms = int32_t(std::strtol(v, nullptr, 10)); if (realtime_ms <= 0) return 2; }
        else if (a == "--skip-parity") { skip_parity = true; }
        else if (a == "--manual-oracle") { manual_oracle = true; }
        else if (a == "--ab")          { ab = true; }
        else if (a == "--gpu") {
            const char * v = val("--gpu"); if (!v) return 2;
            std::string g = v;
            if (g == "none") gpu = voxtral_gpu_backend::none;
            else if (g == "auto") gpu = voxtral_gpu_backend::auto_detect;
            else if (g == "cuda") gpu = voxtral_gpu_backend::cuda;
            else if (g == "metal") gpu = voxtral_gpu_backend::metal;
            else if (g == "vulkan") gpu = voxtral_gpu_backend::vulkan;
            else { std::cerr << "invalid --gpu\n"; return 2; }
        } else { std::cerr << "unknown option: " << a << "\n"; return 2; }
    }

    if (model_path.empty() || wav_path.empty()) {
        std::cerr << "usage: voxtral-stream-test --model M.gguf --wav in.wav "
                     "[--gpu ...] [--plan-file f] [--mode m] [--max-tokens n] [--realtime-ms n] [--ab]\n";
        return 2;
    }

    // Read audio.
    std::vector<int16_t> pcm16;
    std::string err;
    if (!read_wav_pcm16_mono16k(wav_path, pcm16, err)) {
        std::cerr << "wav error: " << err << "\n";
        return 3;
    }
    const size_t total = pcm16.size();

    // Build the feed plan.
    std::vector<size_t> counts;
    if (!plan_file.empty()) {
        if (!plan_from_file(plan_file, counts, err)) { std::cerr << err << "\n"; return 4; }
        size_t sum = 0; for (size_t c : counts) sum += c;
        if (sum != total) {
            std::cerr << "plan sample sum " << sum << " != wav samples " << total << "\n";
            return 4;
        }
    } else {
        counts = plan_from_mode(mode, total);
    }

    if (realtime_ms > 0 && plan_file.empty()) {
        counts = plan_from_mode(std::to_string(realtime_ms) + "ms", total);
        // Keep the accepted matrix explicit even if an unusual value is passed.
        if (counts.empty() && total > 0) counts = plan_from_mode("80ms", total);
    }

    DriveOptions drive_options;
    drive_options.paced = realtime_ms > 0;
    // A plan file may contain an irregular paced timeline. In that case the
    // per-feed sample count is the cadence; a fixed --realtime-ms value merely
    // enables pacing and must not distort backlog subtraction.
    drive_options.pace_chunk_ms = plan_file.empty() ? realtime_ms : 0;
    drive_options.check_parity = !skip_parity;
    drive_options.check_manual_oracle = manual_oracle;

    // Load the model ONCE (shared, immutable). Each stream will create and own
    // its own execution context from it via voxtral_stream_create_internal.
    voxtral_log_callback logger = log_stderr;
    voxtral_model * model = voxtral_model_load_from_file(model_path, logger, gpu);
    if (!model) { std::cerr << "failed to load model\n"; return 5; }
    const int64_t model_loaded_vram_bytes = read_vram_used_bytes();

    voxtral_context_params cp;
    cp.log_level = voxtral_log_level::info;
    cp.logger    = logger;
    cp.gpu       = gpu;

    voxtral_stream_params sp;
    sp.max_tokens = max_tokens;
    sp.retain_mel_history = !skip_parity || manual_oracle;

    int exit_code = 0;
    std::ostringstream js;

    if (ab) {
        // Two streams from one model, both alive: their contexts must be distinct
        // (each stream owns its own). Execution stays strictly sequential — A is
        // fully finished before B is fed; this is NOT concurrent streaming.
        voxtral_stream * a = voxtral_stream_create_internal(model, cp, sp);
        voxtral_stream * b = voxtral_stream_create_internal(model, cp, sp);
        if (!a || !b) {
            std::cerr << "failed to create A/B streams\n";
            voxtral_stream_destroy_internal(a);
            voxtral_stream_destroy_internal(b);
            voxtral_model_free(model);
            return 7;
        }
        const void * ctx_a = voxtral_stream_context_ptr(a);
        const void * ctx_b = voxtral_stream_context_ptr(b);
        const bool distinct = (ctx_a != nullptr && ctx_b != nullptr && ctx_a != ctx_b);

        StreamRun run_a = drive_stream(a, pcm16, counts, drive_options);
        StreamRun run_b = drive_stream(b, pcm16, counts, drive_options);

        // Destroy each stream (frees its OWN context); the model outlives both.
        voxtral_stream_destroy_internal(a);
        voxtral_stream_destroy_internal(b);

        js << "{";
        js << "\"mode\":\"ab\",";
        js << "\"modelShared\":true,";
        js << "\"threading\":\"externally_serialized\",";
        js << "\"distinctContexts\":" << (distinct ? "true" : "false") << ",";
        js << "\"sampleRate\":" << sp.sample_rate << ",";
        js << "\"channels\":" << sp.channels << ",";
        js << "\"wavSamples\":" << total << ",";
        js << "\"a\":{"; write_run_fields(js, run_a); js << "},";
        js << "\"b\":{"; write_run_fields(js, run_b); js << "}";
        js << "}";

        exit_code = (run_a.ok && run_b.ok && distinct) ? 0 : 1;
    } else {
        voxtral_stream * stream = voxtral_stream_create_internal(model, cp, sp);
        if (!stream) { std::cerr << "failed to create stream\n"; voxtral_model_free(model); return 7; }

        const int64_t stream_idle_vram_bytes = read_vram_used_bytes();
        StreamRun run = drive_stream(stream, pcm16, counts, drive_options);
        run.modelLoadedVramBytes = model_loaded_vram_bytes;
        run.streamIdleVramBytes = stream_idle_vram_bytes;
        run.afterFinishVramBytes = read_vram_used_bytes();

        // Destroy the stream (frees its owned context), then free the model.
        voxtral_stream_destroy_internal(stream);
        run.afterDestroyVramBytes = read_vram_used_bytes();

        js << "{";
        js << "\"mode\":\"single\",";
        js << "\"modelShared\":true,";
        js << "\"threading\":\"externally_serialized\",";
        js << "\"sampleRate\":" << sp.sample_rate << ",";
        js << "\"channels\":" << sp.channels << ",";
        js << "\"wavSamples\":" << total << ",";
        write_run_fields(js, run);
        js << "}";

        exit_code = run.ok ? 0 : 1;
    }

    std::cout << js.str() << "\n";

    voxtral_model_free(model);
    return exit_code;
}

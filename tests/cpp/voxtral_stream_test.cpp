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
//                       [--realtime-ms N] [--warmup] [--manual-oracle] [--ab] [--kv-parity]
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
//   --kv-parity : sequential FP16 production, same-shape FP16/F32 numerical
//                 plans, F32 production-topology plan, and accepted F32/32-row
//                 finish-only oracle runs from one shared model.
// ============================================================================

#include "voxtral-stream.h"
#include "voxtral.h"
#include "voxtral-mel.h"
#include "voxtral-internal.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
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

#if defined(__GLIBC__)
#include <malloc.h>
#endif

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
struct ProcessMemorySnapshot {
    int64_t rssKiB = 0;
    int64_t pssKiB = 0;
    int64_t anonymousRssKiB = 0;
    int64_t sharedCleanKiB = 0;
    int64_t sharedDirtyKiB = 0;
    int64_t privateCleanKiB = 0;
    int64_t privateDirtyKiB = 0;
    int64_t fileBackedRssKiB = 0;
    int64_t sharedLibraryRssKiB = 0;
    int64_t vulkanDriverRssKiB = 0;
    int64_t heapMappingRssKiB = 0;
    int64_t heapUsedBytes = 0;
    int64_t heapRetainedFreeBytes = 0;
    int64_t heapArenaBytes = 0;
    int64_t vramBytes = 0;
};

struct RolloverMemorySample {
    int64_t wrap = 0;
    int64_t absolutePosition = -1;
    int64_t settledAfterDecoderSteps = 0;
    ProcessMemorySnapshot before;
    ProcessMemorySnapshot after;
    ProcessMemorySnapshot settled;
    bool settledCaptured = false;
    uint64_t graphObjects = 0;
    uint64_t graphAllocations = 0;
    uint64_t encoderAllocations = 0;
    uint64_t adapterAllocations = 0;
    uint64_t decoderAllocations = 0;
    int64_t decoderKvBytesMoved = 0;
    int64_t decoderKvFullBufferMoves = 0;
};

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
    bool                 warmupApplied = false;
    double               warmupMs = 0.0;
    double               modelLoadMs = 0.0;
    double               contextCreationMs = 0.0;
    double               streamStartMs = 0.0;
    double               coldFirstDecoderStepMs = 0.0;
    double               coldFirstTokenMs = 0.0;
    double               coldFirstVisibleTextMs = 0.0;
    double               warmModelFirstDecoderStepMs = 0.0;
    double               warmModelFirstTokenMs = 0.0;
    double               warmModelFirstVisibleTextMs = 0.0;

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
    std::string adapterSha;
    int64_t  encoderShaRows = 0;
    int64_t  adapterShaRows = 0;
    int64_t  outputShaDiagnosticD2hBytes = 0;
    double   encoderMaxAbsDeltaVsBatch = 0.0;
    double   encoderMaxAbsDeltaVsManual = 0.0;
    double   encoderManualMeanAbsDelta   = 0.0;
    double   encoderManualRmsDelta       = 0.0;
    double   encoderManualReferenceRms   = 0.0;
    double   encoderManualCosineSimilarity = 0.0;
    std::string encoderManualSha;
    bool     manualOracleChecked         = false;
    std::vector<float> numericalEncoderOutput;
    std::vector<float> numericalAdapterOutput;
    std::vector<float> numericalDecoderHidden;
    std::vector<float> numericalDecoderLogits;
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
    double postInputDrainMs = 0.0;
    int64_t terminalPartialChunkSamples = 0;
    double terminalPartialChunkMs = 0.0;
    double terminalPartialFinishLatenessMs = 0.0;
    uint64_t deadlineMisses = 0;
    double deadlineMissRate = 0.0;
    voxtral_backlog_metrics encoderBacklog;
    voxtral_backlog_metrics adapterBacklog;
    voxtral_backlog_metrics decoderBacklog;
    double finishFrontendMs = 0.0;
    double finishEncoderMs = 0.0;
    double finishDecoderMs = 0.0;
    int64_t decoderKvAllocatedBytes = 0;
    int64_t modelLoadedVramBytes = 0;
    int64_t streamIdleVramBytes = 0;
    int64_t afterFinishVramBytes = 0;
    int64_t afterDestroyVramBytes = 0;
    int64_t modelLoadedRssKiB = 0;
    int64_t streamIdleRssKiB = 0;
    int64_t afterFinishRssKiB = 0;
    int64_t afterDestroyRssKiB = 0;
    ProcessMemorySnapshot modelLoadedMemory;
    ProcessMemorySnapshot beforeWarmupMemory;
    ProcessMemorySnapshot afterWarmupMemory;
    ProcessMemorySnapshot afterFinishMemory;
    ProcessMemorySnapshot afterDestroyMemory;
    ProcessMemorySnapshot afterMallocTrimMemory;
    bool mallocTrimRequested = false;
    bool mallocTrimAvailable = false;
    bool mallocTrimApplied = false;
    int32_t mallocTrimReturn = -1;
    std::vector<RolloverMemorySample> rolloverMemory;
    bool parityChecked = true;
    voxtral_runtime_profile steadyRuntimeProfile;
    voxtral_runtime_profile runtimeProfile;

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
    double   firstDecoderStepEligibilityMs = -1.0;
    double   firstDecoderStepOverheadMs = -1.0;
    double   firstTokenEligibilityMs = -1.0;
    double   firstTokenOverheadMs = -1.0;
    double   firstPartialEligibilityMs = -1.0;
    double   firstPartialOverheadMs = -1.0;
    int64_t  adapterInputD2hBytes = 0;
    int64_t  adapterOutputD2hBytes = 0;
    int64_t  logitsD2hBytes = 0;
    int64_t  tokenIdD2hBytes = 0;
    int64_t  encoderOutputD2hBytes = 0;
    uint64_t partialTextRevision = 0;
    bool     eventHistoryRetained = true;
    uint64_t retainedEventHistoryCount = 0;
    uint64_t tokenOutputBytes = 0;
    uint64_t transcriptOutputBytes = 0;

    // Session 7.1: decoder mode + event-queue telemetry + backpressure evidence.
    std::string decoderMode = "incremental";
    uint64_t eventsEmitted = 0;
    uint64_t tokenEventsCount = 0;
    uint64_t partialEventsCount = 0;
    uint64_t partialEventsCoalesced = 0;
    uint64_t eventQueueHighWatermark = 0;
    uint64_t eventQueueOverflowAttempts = 0;
    uint64_t eventsDropped = 0;
    int64_t  maxEventsBound = 0;          // queue bound applied to this run (0 = default)
    int64_t  feedQueueFullReturns = 0;    // feeds that returned queue_full (backpressure)
    bool     backpressureObserved = false;
    std::string lastFeedStatus = "ok";
};

struct DriveOptions {
    bool paced = false;
    int32_t pace_chunk_ms = 0;
    bool check_parity = true;
    bool check_manual_oracle = false;
    bool capture_numerical = false;
    bool capture_rollover_memory = false;
    bool retain_event_history = true;
    // Session 7.1 backpressure exercise: shrink the event queue to `max_events`
    // and, when `backpressure` is set, DELIBERATELY stop draining until feed
    // reports queue_full — then drain and continue. Proves mandatory events are
    // never dropped and the transcript is identical to the always-drain run.
    int32_t max_events = 0;      // 0 = leave the default bound
    bool    backpressure = false;
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

int64_t read_process_rss_kib() {
    std::ifstream in("/proc/self/status");
    std::string key;
    while (in >> key) {
        if (key == "VmRSS:") {
            int64_t value = 0;
            std::string unit;
            in >> value >> unit;
            return value;
        }
        std::string rest;
        std::getline(in, rest);
    }
    return 0;
}

int64_t parse_kib_value(const std::string & line) {
    const size_t colon = line.find(':');
    if (colon == std::string::npos) return 0;
    std::istringstream input(line.substr(colon + 1));
    int64_t value = 0;
    input >> value;
    return value;
}

ProcessMemorySnapshot read_process_memory_snapshot() {
    ProcessMemorySnapshot out;
    {
        std::ifstream in("/proc/self/smaps_rollup");
        std::string line;
        while (std::getline(in, line)) {
            if      (line.rfind("Rss:", 0) == 0)           out.rssKiB = parse_kib_value(line);
            else if (line.rfind("Pss:", 0) == 0)           out.pssKiB = parse_kib_value(line);
            else if (line.rfind("Anonymous:", 0) == 0)     out.anonymousRssKiB = parse_kib_value(line);
            else if (line.rfind("Shared_Clean:", 0) == 0)  out.sharedCleanKiB = parse_kib_value(line);
            else if (line.rfind("Shared_Dirty:", 0) == 0)  out.sharedDirtyKiB = parse_kib_value(line);
            else if (line.rfind("Private_Clean:", 0) == 0) out.privateCleanKiB = parse_kib_value(line);
            else if (line.rfind("Private_Dirty:", 0) == 0) out.privateDirtyKiB = parse_kib_value(line);
        }
    }
    out.fileBackedRssKiB = std::max<int64_t>(0, out.rssKiB - out.anonymousRssKiB);

    // Mapping attribution is diagnostic: a driver shared object can also appear
    // in the shared-library total.  Keep those categories explicit instead of
    // pretending they form a disjoint accounting identity.
    {
        std::ifstream in("/proc/self/smaps");
        std::string line;
        bool mapping_driver = false;
        bool mapping_library = false;
        bool mapping_heap = false;
        while (std::getline(in, line)) {
            const size_t dash = line.find('-');
            const size_t space = line.find(' ');
            const bool header = dash != std::string::npos &&
                                space != std::string::npos && dash < space;
            if (header) {
                std::string lower = line;
                std::transform(lower.begin(), lower.end(), lower.begin(),
                               [](unsigned char c) { return (char) std::tolower(c); });
                mapping_driver =
                    lower.find("radv") != std::string::npos ||
                    lower.find("vulkan") != std::string::npos ||
                    lower.find("amdgpu") != std::string::npos ||
                    lower.find("libdrm") != std::string::npos ||
                    lower.find("mesa") != std::string::npos;
                mapping_library =
                    lower.find(".so") != std::string::npos ||
                    lower.find("/lib/") != std::string::npos ||
                    lower.find("/lib64/") != std::string::npos;
                mapping_heap = lower.find("[heap]") != std::string::npos;
            } else if (line.rfind("Rss:", 0) == 0) {
                const int64_t value = parse_kib_value(line);
                if (mapping_driver)  out.vulkanDriverRssKiB += value;
                if (mapping_library) out.sharedLibraryRssKiB += value;
                if (mapping_heap)    out.heapMappingRssKiB += value;
            }
        }
    }

#if defined(__GLIBC__)
    const struct mallinfo2 info = mallinfo2();
    out.heapUsedBytes = (int64_t) info.uordblks;
    out.heapRetainedFreeBytes = (int64_t) info.fordblks;
    out.heapArenaBytes = (int64_t) info.arena + (int64_t) info.hblkhd;
#endif
    out.vramBytes = read_vram_used_bytes();
    return out;
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
    size_t elements = 0;
    double max_abs = 0.0;
    double mean_abs = 0.0;
    double rms_delta = 0.0;
    double reference_rms = 0.0;
    double cosine = 0.0;
    std::string reference_sha;
};

TensorComparison compare_tensors(const std::vector<float> & got,
                                 const std::vector<float> & reference) {
    TensorComparison cmp;
    if (got.empty() || got.size() != reference.size()) return cmp;
    cmp.elements = got.size();
    long double sum_abs = 0.0L, sum_sq = 0.0L, ref_sq = 0.0L;
    long double got_sq = 0.0L, dot = 0.0L;
    for (size_t i = 0; i < got.size(); ++i) {
        const double g = got[i];
        const double r = reference[i];
        const double d = g - r;
        cmp.max_abs = std::max(cmp.max_abs, std::fabs(d));
        sum_abs += std::fabs(d);
        sum_sq += d * d;
        ref_sq += r * r;
        got_sq += g * g;
        dot += g * r;
    }
    const long double n = (long double) got.size();
    cmp.mean_abs = (double) (sum_abs / n);
    cmp.rms_delta = std::sqrt((double) (sum_sq / n));
    cmp.reference_rms = std::sqrt((double) (ref_sq / n));
    const long double denom = std::sqrt(got_sq * ref_sq);
    cmp.cosine = denom > 0.0L ? (double) (dot / denom) : 0.0;
    cmp.ok = true;
    return cmp;
}

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
    r.eventHistoryRetained = options.retain_event_history;
    auto drain_events = [&]() {
        voxtral_stream_event event;
        while (voxtral_stream_poll_event(stream, event)) {
            if (options.retain_event_history) {
                r.events.push_back(std::move(event));
            }
        }
    };
    if (options.max_events > 0) {
        voxtral_stream_test_set_max_events(stream, (size_t) options.max_events);
        r.maxEventsBound = options.max_events;
    }

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
    bool first_feed_seen = false;
    const auto * const run_context = static_cast<const voxtral_context *>(
        voxtral_stream_context_ptr(stream));
    // Building a full runtime-profile snapshot computes percentiles from the
    // bounded reservoirs. That is intentionally diagnostic work, not part of
    // the production feed path: doing it before and after every paced feed
    // becomes progressively expensive and can manufacture backlog in a long
    // soak. Per-feed snapshots are needed only by the accelerated rollover
    // attribution mode; ordinary runs take their complete steady/final
    // snapshots once below.
    voxtral_runtime_profile previous_profile{};
    if (options.capture_rollover_memory) {
        previous_profile =
            voxtral_context_runtime_profile_internal(run_context);
    }
    voxtral_status fst = voxtral_status::ok;
    for (size_t c : counts) {
        // A captured chunk becomes callable only when its final sample has
        // arrived. Anchor every deadline to the common monotonic start so
        // scheduler/OS delays never accumulate as sleep drift.
        const auto end_deadline = wall_start + std::chrono::microseconds(
            (long long) ((double) (audio_cursor + c) * 1'000'000.0 / VOXTRAL_SAMPLE_RATE));
        if (options.paced) std::this_thread::sleep_until(end_deadline);
        const auto before_feed = std::chrono::steady_clock::now();
        if (!first_feed_seen) {
            r.streamStartMs =
                std::chrono::duration<double, std::milli>(before_feed - wall_start).count();
            first_feed_seen = true;
        }
        if (options.paced) {
            start_lateness_ms.push_back(std::max(0.0,
                std::chrono::duration<double, std::milli>(before_feed - end_deadline).count()));
        }
        voxtral_runtime_profile before_profile{};
        bool next_step_wraps = false;
        if (options.capture_rollover_memory) {
            before_profile =
                voxtral_context_runtime_profile_internal(run_context);
            next_step_wraps =
                before_profile.decoderKvCapacity > 0 &&
                before_profile.decoderNextAbsolutePosition >= before_profile.decoderKvCapacity &&
                before_profile.decoderNextAbsolutePosition % before_profile.decoderKvCapacity == 0;
        }
        ProcessMemorySnapshot before_wrap_memory;
        if (next_step_wraps) before_wrap_memory = read_process_memory_snapshot();
        const int16_t * ptr = (c == 0) ? nullptr : (pcm16.data() + off);
        const auto tf0 = before_feed;
        fst = voxtral_stream_feed_pcm16_internal(stream, ptr, c);
        const auto tf1 = std::chrono::steady_clock::now();
        if (options.capture_rollover_memory) {
            const voxtral_runtime_profile after_profile =
                voxtral_context_runtime_profile_internal(run_context);
            if (after_profile.decoderKvWraps > previous_profile.decoderKvWraps) {
                const ProcessMemorySnapshot after_memory = read_process_memory_snapshot();
                for (int64_t wrap = previous_profile.decoderKvWraps + 1;
                     wrap <= after_profile.decoderKvWraps; ++wrap) {
                    RolloverMemorySample sample;
                    sample.wrap = wrap;
                    sample.absolutePosition =
                        next_step_wraps && wrap == after_profile.decoderKvWraps
                            ? before_profile.decoderNextAbsolutePosition
                            : wrap * after_profile.decoderKvCapacity;
                    sample.before = next_step_wraps ? before_wrap_memory : after_memory;
                    sample.after = after_memory;
                    sample.graphObjects =
                        after_profile.encoderGraphBuildCount +
                        after_profile.adapterGraphBuildCount +
                        after_profile.decoderGraphBuildCount;
                    sample.graphAllocations = after_profile.graphAllocations;
                    sample.encoderAllocations = after_profile.encoderAllocations;
                    sample.adapterAllocations = after_profile.adapterAllocations;
                    sample.decoderAllocations = after_profile.decoderAllocations;
                    sample.decoderKvBytesMoved = after_profile.decoderKvBytesMoved;
                    sample.decoderKvFullBufferMoves =
                        after_profile.decoderKvFullBufferMoves;
                    r.rolloverMemory.push_back(std::move(sample));
                }
            }
            for (auto & sample : r.rolloverMemory) {
                if (!sample.settledCaptured &&
                    after_profile.decoderNextAbsolutePosition >=
                        sample.absolutePosition + 11) {
                    sample.settled = read_process_memory_snapshot();
                    sample.settledAfterDecoderSteps =
                        after_profile.decoderNextAbsolutePosition -
                        (sample.absolutePosition + 1);
                    sample.settledCaptured = true;
                }
            }
            previous_profile = after_profile;
        }
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
            const int64_t nominal_samples = options.pace_chunk_ms > 0
                ? (int64_t) options.pace_chunk_ms * VOXTRAL_SAMPLE_RATE / 1000
                : 0;
            const bool terminal_partial =
                options.pace_chunk_ms > 0 &&
                off + c == pcm16.size() &&
                (int64_t) c < nominal_samples;
            if (terminal_partial) {
                // There is no following full capture interval against which a
                // terminal short block can accumulate backlog. Keep its actual
                // completion lateness as post-input drain evidence, but exclude
                // it from the sustained cadence regression/final queue sample.
                r.terminalPartialChunkSamples = (int64_t) c;
                r.terminalPartialChunkMs =
                    (double) c * 1000.0 / VOXTRAL_SAMPLE_RATE;
                r.terminalPartialFinishLatenessMs = late;
            } else {
                backlog_ms.push_back(std::max(0.0, late - cadence_ms));
                backlog_audio_s.push_back(
                    (double) (audio_cursor + c) / VOXTRAL_SAMPLE_RATE);
            }
        }
        if (fst == voxtral_status::limit_exceeded) {
            // Explicit backpressure (queue_full), NOT a failure: the audio was
            // accepted; the decoder has output pending because the event queue is
            // full. Record it, drain, and continue — the next feed / finish resumes
            // the decoder. Mandatory events are never dropped (events_dropped == 0).
            r.backpressureObserved = true;
            r.feedQueueFullReturns++;
            drain_events();
            off += c;
            audio_cursor += c;
            fst = voxtral_status::ok;   // resumed for the next iteration / finish
            continue;
        }
        if (fst != voxtral_status::ok) {
            std::cerr << "feed failed: " << voxtral_stream_status_name(fst)
                      << " (" << voxtral_stream_last_error(stream) << ")\n";
            break;
        }
        off += c;
        audio_cursor += c;
        // A realistic consumer drains during feed so the bounded event queue never
        // overflows. The backpressure exercise deliberately withholds draining until
        // feed reports queue_full (above), to prove the mandatory-event contract.
        if (!options.backpressure) {
            drain_events();
        }
    }

    // Snapshot the hot path before bounded right-padding/EOS work. Session 8's
    // no-growth gate applies to steady audio processing; terminal graph families
    // are reported separately instead of being mistaken for per-step churn.
    r.steadyRuntimeProfile = voxtral_context_runtime_profile_internal(
        static_cast<const voxtral_context *>(voxtral_stream_context_ptr(stream)));

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
    if (options.capture_rollover_memory) {
        const voxtral_runtime_profile final_profile =
            voxtral_context_runtime_profile_internal(run_context);
        const ProcessMemorySnapshot final_memory = read_process_memory_snapshot();
        if (final_profile.decoderKvWraps > previous_profile.decoderKvWraps) {
            for (int64_t wrap = previous_profile.decoderKvWraps + 1;
                 wrap <= final_profile.decoderKvWraps; ++wrap) {
                RolloverMemorySample sample;
                sample.wrap = wrap;
                sample.absolutePosition = wrap * final_profile.decoderKvCapacity;
                sample.before = final_memory;
                sample.after = final_memory;
                sample.settled = final_memory;
                sample.settledCaptured = true;
                sample.graphObjects =
                    final_profile.encoderGraphBuildCount +
                    final_profile.adapterGraphBuildCount +
                    final_profile.decoderGraphBuildCount;
                sample.graphAllocations = final_profile.graphAllocations;
                sample.encoderAllocations = final_profile.encoderAllocations;
                sample.adapterAllocations = final_profile.adapterAllocations;
                sample.decoderAllocations = final_profile.decoderAllocations;
                sample.decoderKvBytesMoved = final_profile.decoderKvBytesMoved;
                sample.decoderKvFullBufferMoves =
                    final_profile.decoderKvFullBufferMoves;
                r.rolloverMemory.push_back(std::move(sample));
            }
        }
        for (auto & sample : r.rolloverMemory) {
            if (!sample.settledCaptured) {
                sample.settled = final_memory;
                sample.settledAfterDecoderSteps = std::max<int64_t>(
                    0, final_profile.decoderNextAbsolutePosition -
                       (sample.absolutePosition + 1));
                sample.settledCaptured = true;
            }
        }
        r.afterFinishMemory = final_memory;
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
        // The feed call is synchronous: this is the wall interval from arrival
        // of the final PCM sample to completion of all work triggered by it.
        // EOS/right-padding work remains separately accounted by finish().
        r.postInputDrainMs = finish_lateness_ms.back();
        r.backlogP50Ms = percentile(backlog_ms, 0.50);
        r.backlogP95Ms = percentile(backlog_ms, 0.95);
        r.backlogP99Ms = percentile(backlog_ms, 0.99);
        r.backlogMaxMs = percentile(backlog_ms, 1.00);
        r.finalBacklogMs = backlog_ms.empty() ? 0.0 : backlog_ms.back();
        r.deadlineMisses = (uint64_t) std::count_if(
            backlog_ms.begin(), backlog_ms.end(), [](double value) { return value > 0.0; });
        r.deadlineMissRate = backlog_ms.empty()
            ? 0.0 : (double) r.deadlineMisses / (double) backlog_ms.size();
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
    // Snapshot BEFORE the optional batch/manual parity passes below: those are
    // reference-oracle work, not part of the production stream profile.
    r.runtimeProfile = voxtral_context_runtime_profile_internal(
        static_cast<const voxtral_context *>(voxtral_stream_context_ptr(stream)));
    r.encoderBacklog = voxtral_stream_encoder_backlog(stream);
    r.adapterBacklog = voxtral_stream_adapter_backlog(stream);
    r.decoderBacklog = voxtral_stream_decoder_backlog(stream);
    r.fullMelReencodedAtFinish      = false;   // finish runs at most the last 1-2 encoder chunks

    const float * inc_enc    = voxtral_stream_encoder_output_data(stream);
    const int32_t inc_enc_frames = voxtral_stream_encoder_output_frames_count(stream);
    r.encoderShaRows = voxtral_stream_encoder_output_sha_rows(stream);
    r.adapterShaRows = voxtral_stream_adapter_output_sha_rows(stream);
    r.outputShaDiagnosticD2hBytes =
        voxtral_stream_output_sha_d2h_bytes(stream);
    r.adapterSha = voxtral_stream_adapter_output_sha256(stream);
    {
        Sha256 esha;
        if (r.encoderShaRows > 0) {
            r.encoderSha = voxtral_stream_encoder_output_sha256(stream);
        } else if (inc_enc && inc_enc_frames > 0) {
            const int32_t used = (inc_enc_frames / 4) * 4;
            esha.update(inc_enc, (size_t) used * VOXTRAL_ENC_DIM * sizeof(float));
            r.encoderSha = esha.hex();
        } else {
            r.encoderSha = esha.hex();
        }
    }
    if (options.capture_numerical && inc_enc && inc_enc_frames > 0) {
        const int32_t used = (inc_enc_frames / 4) * 4;
        r.numericalEncoderOutput.assign(
            inc_enc, inc_enc + (size_t) used * VOXTRAL_ENC_DIM);
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

    // Drain events (order preserved when history capture is enabled).
    drain_events();

    r.state           = voxtral_stream_state_name(voxtral_stream_get_state(stream));
    r.finishStatus    = voxtral_stream_status_name(finst);
    r.samplesReceived = voxtral_stream_samples_received(stream);
    r.samplesConsumed = voxtral_stream_samples_consumed(stream);
    r.feedCalls       = feed_calls;
    r.inferenceRuns   = voxtral_stream_inference_runs(stream);
    r.pcmFloats       = voxtral_stream_samples_received(stream);
    r.tokens          = voxtral_stream_tokens(stream);
    r.text            = voxtral_stream_transcript(stream);
    r.retainedEventHistoryCount = r.events.size();
    r.tokenOutputBytes = r.tokens.size() * sizeof(int32_t);
    r.transcriptOutputBytes = r.text.size();
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
    r.firstDecoderStepEligibilityMs =
        voxtral_stream_first_decoder_step_eligibility_ms(stream);
    r.firstDecoderStepOverheadMs =
        voxtral_stream_first_decoder_step_overhead_ms(stream);
    r.firstTokenEligibilityMs =
        voxtral_stream_first_token_eligibility_ms(stream);
    r.firstTokenOverheadMs =
        voxtral_stream_first_token_overhead_ms(stream);
    r.firstPartialEligibilityMs =
        voxtral_stream_first_partial_eligibility_ms(stream);
    r.firstPartialOverheadMs =
        voxtral_stream_first_partial_overhead_ms(stream);
    r.adapterInputD2hBytes    = voxtral_stream_adapter_input_d2h_bytes(stream);
    r.adapterOutputD2hBytes   = voxtral_stream_adapter_output_d2h_bytes(stream);
    r.logitsD2hBytes          = voxtral_stream_logits_d2h_bytes(stream);
    r.tokenIdD2hBytes         = voxtral_stream_token_id_d2h_bytes(stream);
    r.encoderOutputD2hBytes   = voxtral_stream_encoder_output_d2h_bytes(stream);
    r.partialTextRevision     = voxtral_stream_partial_text_revision(stream);

    if (options.capture_numerical) {
        const auto * ctx = static_cast<const voxtral_context *>(
            voxtral_stream_context_ptr(stream));
        const auto & production_encoder =
            voxtral_context_diagnostic_encoder_output_internal(ctx);
        if (!production_encoder.empty()) {
            r.numericalEncoderOutput = production_encoder;
        }
        r.numericalAdapterOutput =
            voxtral_context_diagnostic_adapter_output_internal(ctx);
        r.numericalDecoderHidden =
            voxtral_context_diagnostic_first_hidden_internal(ctx);
        r.numericalDecoderLogits =
            voxtral_context_diagnostic_first_logits_internal(ctx);
    }

    // Session 7.1: decoder mode + event-queue telemetry.
    r.decoderMode                = voxtral_stream_decoder_mode(stream);
    r.eventsEmitted              = voxtral_stream_events_emitted(stream);
    r.tokenEventsCount           = voxtral_stream_token_events(stream);
    r.partialEventsCount         = voxtral_stream_partial_events(stream);
    r.partialEventsCoalesced     = voxtral_stream_partial_events_coalesced(stream);
    r.eventQueueHighWatermark    = voxtral_stream_event_queue_high_watermark(stream);
    r.eventQueueOverflowAttempts = voxtral_stream_event_queue_overflow_attempts(stream);
    r.eventsDropped              = voxtral_stream_events_dropped(stream);
    switch (voxtral_stream_last_feed_status(stream)) {
        case voxtral_stream_feed_status::ok:          r.lastFeedStatus = "ok"; break;
        case voxtral_stream_feed_status::would_block: r.lastFeedStatus = "would_block"; break;
        case voxtral_stream_feed_status::queue_full:  r.lastFeedStatus = "queue_full"; break;
        case voxtral_stream_feed_status::cancelled:   r.lastFeedStatus = "cancelled"; break;
        case voxtral_stream_feed_status::failed:      r.lastFeedStatus = "failed"; break;
    }
    return r;
}

void write_memory_snapshot(std::ostringstream & js,
                           const ProcessMemorySnapshot & memory) {
    js << "{";
    js << "\"rssKiB\":" << memory.rssKiB << ",";
    js << "\"pssKiB\":" << memory.pssKiB << ",";
    js << "\"anonymousRssKiB\":" << memory.anonymousRssKiB << ",";
    js << "\"sharedCleanKiB\":" << memory.sharedCleanKiB << ",";
    js << "\"sharedDirtyKiB\":" << memory.sharedDirtyKiB << ",";
    js << "\"privateCleanKiB\":" << memory.privateCleanKiB << ",";
    js << "\"privateDirtyKiB\":" << memory.privateDirtyKiB << ",";
    js << "\"fileBackedRssKiB\":" << memory.fileBackedRssKiB << ",";
    js << "\"sharedLibraryRssKiB\":" << memory.sharedLibraryRssKiB << ",";
    js << "\"vulkanDriverRssKiB\":" << memory.vulkanDriverRssKiB << ",";
    js << "\"heapMappingRssKiB\":" << memory.heapMappingRssKiB << ",";
    js << "\"heapUsedBytes\":" << memory.heapUsedBytes << ",";
    js << "\"heapRetainedFreeBytes\":" << memory.heapRetainedFreeBytes << ",";
    js << "\"heapArenaBytes\":" << memory.heapArenaBytes << ",";
    js << "\"vramBytes\":" << memory.vramBytes;
    js << "}";
}

// Emit a StreamRun's fields into an open JSON object body (no surrounding braces).
void write_run_fields(std::ostringstream & js, const StreamRun & r) {
    js << "\"state\":\"" << r.state << "\",";
    js << "\"finishStatus\":\"" << r.finishStatus << "\",";
    js << "\"warmupApplied\":" << (r.warmupApplied ? "true" : "false") << ",";
    js << "\"warmupMs\":" << r.warmupMs << ",";
    js << "\"modelLoadMs\":" << r.modelLoadMs << ",";
    js << "\"contextCreationMs\":" << r.contextCreationMs << ",";
    js << "\"streamStartMs\":" << r.streamStartMs << ",";
    js << "\"coldFirstDecoderStepMs\":" << r.coldFirstDecoderStepMs << ",";
    js << "\"coldFirstTokenMs\":" << r.coldFirstTokenMs << ",";
    js << "\"coldFirstVisibleTextMs\":" << r.coldFirstVisibleTextMs << ",";
    js << "\"warmModelFirstDecoderStepMs\":" << r.warmModelFirstDecoderStepMs << ",";
    js << "\"warmModelFirstTokenMs\":" << r.warmModelFirstTokenMs << ",";
    js << "\"warmModelFirstVisibleTextMs\":" << r.warmModelFirstVisibleTextMs << ",";
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
    js << "\"adapterSha256\":\"" << r.adapterSha << "\",";
    js << "\"encoderShaRows\":" << r.encoderShaRows << ",";
    js << "\"adapterShaRows\":" << r.adapterShaRows << ",";
    js << "\"outputShaDiagnosticD2hBytes\":"
       << r.outputShaDiagnosticD2hBytes << ",";
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
    js << "\"profileEnabled\":" << (r.runtimeProfile.enabled ? "true" : "false") << ",";
    js << "\"profileStages\":{";
    for (size_t i = 0; i < r.runtimeProfile.stages.size(); ++i) {
        if (i) js << ",";
        const auto stage = static_cast<voxtral_profile_stage>(i);
        const auto & p = r.runtimeProfile.stages[i];
        js << "\"" << voxtral_profile_stage_name(stage) << "\":{";
        js << "\"count\":" << p.count << ",";
        js << "\"totalMs\":" << p.totalMs << ",";
        js << "\"meanMs\":" << p.meanMs << ",";
        js << "\"p50Ms\":" << p.p50Ms << ",";
        js << "\"p95Ms\":" << p.p95Ms << ",";
        js << "\"p99Ms\":" << p.p99Ms << ",";
        js << "\"maxMs\":" << p.maxMs << "}";
    }
    js << "},";
    js << "\"totalGpuComputeMs\":" << r.runtimeProfile.totalGpuComputeMs << ",";
    js << "\"totalPipelineComputeMs\":" << r.runtimeProfile.totalPipelineComputeMs << ",";
    const double profiledRtf = r.audioDurationMs > 0.0
        ? (r.runtimeProfile.totalPipelineComputeMs + r.finishLatencyMs) / r.audioDurationMs : 0.0;
    js << "\"pipelineRtf\":" << profiledRtf << ",";
    js << "\"computeHeadroomRatio\":" << (1.0 - profiledRtf) << ",";
    js << "\"gpuBusyMeanMsPerFeed\":"
       << (r.dataFeeds > 0 ? r.runtimeProfile.totalGpuComputeMs / (double) r.dataFeeds : 0.0) << ",";
    js << "\"gpuBusyMeasurement\":\"synchronized_graph_wall_time\",";
    js << "\"gpuTimestampQueries\":false,";
    js << "\"cpuWallMeanMsPerFeed\":"
       << (r.dataFeeds > 0 ? r.runtimeProfile.totalPipelineComputeMs / (double) r.dataFeeds : 0.0) << ",";
    js << "\"encoderGraphBuildCount\":" << r.runtimeProfile.encoderGraphBuildCount << ",";
    js << "\"adapterGraphBuildCount\":" << r.runtimeProfile.adapterGraphBuildCount << ",";
    js << "\"decoderGraphBuildCount\":" << r.runtimeProfile.decoderGraphBuildCount << ",";
    js << "\"encoderAllocations\":" << r.runtimeProfile.encoderAllocations << ",";
    js << "\"adapterAllocations\":" << r.runtimeProfile.adapterAllocations << ",";
    js << "\"decoderAllocations\":" << r.runtimeProfile.decoderAllocations << ",";
    js << "\"steadyEncoderGraphBuildCount\":"
       << r.steadyRuntimeProfile.encoderGraphBuildCount << ",";
    js << "\"steadyAdapterGraphBuildCount\":"
       << r.steadyRuntimeProfile.adapterGraphBuildCount << ",";
    js << "\"steadyDecoderGraphBuildCount\":"
       << r.steadyRuntimeProfile.decoderGraphBuildCount << ",";
    js << "\"steadyEncoderAllocations\":"
       << r.steadyRuntimeProfile.encoderAllocations << ",";
    js << "\"steadyAdapterAllocations\":"
       << r.steadyRuntimeProfile.adapterAllocations << ",";
    js << "\"steadyDecoderAllocations\":"
       << r.steadyRuntimeProfile.decoderAllocations << ",";
    js << "\"graphAllocations\":" << r.runtimeProfile.graphAllocations << ",";
    js << "\"allocationCounterUnit\":\"scheduler_graph_allocation_pass\",";
    js << "\"individualTensorAllocationsInstrumented\":false,";
    js << "\"backendSyncCount\":" << r.runtimeProfile.backendSyncCount << ",";
    js << "\"commandSubmitCount\":" << r.runtimeProfile.commandSubmitCount << ",";
    js << "\"tensorSetCount\":" << r.runtimeProfile.tensorSetCount << ",";
    js << "\"tensorGetCount\":" << r.runtimeProfile.tensorGetCount << ",";
    js << "\"kvF16Bytes\":" << r.runtimeProfile.kvF16Bytes << ",";
    js << "\"temporaryF32KvBytes\":" << r.runtimeProfile.temporaryF32KvBytes << ",";
    js << "\"decoderKvCapacity\":" << r.runtimeProfile.decoderKvCapacity << ",";
    js << "\"decoderKvUsed\":" << r.runtimeProfile.decoderKvUsed << ",";
    js << "\"decoderKvWraps\":" << r.runtimeProfile.decoderKvWraps << ",";
    js << "\"decoderKvEvictions\":" << r.runtimeProfile.decoderKvEvictions << ",";
    js << "\"decoderKvBytesMoved\":" << r.runtimeProfile.decoderKvBytesMoved << ",";
    js << "\"decoderKvFullBufferMoves\":" << r.runtimeProfile.decoderKvFullBufferMoves << ",";
    js << "\"decoderOldestAbsolutePosition\":"
       << r.runtimeProfile.decoderOldestAbsolutePosition << ",";
    js << "\"decoderNextAbsolutePosition\":"
       << r.runtimeProfile.decoderNextAbsolutePosition << ",";
    js << "\"decoderKvElementSize\":" << r.runtimeProfile.decoderKvElementSize << ",";
    js << "\"decoderFirstWrapAbsolutePosition\":"
       << r.runtimeProfile.decoderFirstWrapAbsolutePosition << ",";
    const double firstWrapAudioMs = r.runtimeProfile.decoderFirstWrapAbsolutePosition >= 0
        ? std::max(0.0,
            ((double) (r.runtimeProfile.decoderFirstWrapAbsolutePosition + 1) *
             VOXTRAL_RAW_AUDIO_LENGTH_PER_TOK -
             (double) VOXTRAL_N_LEFT_PAD_TOKENS * VOXTRAL_RAW_AUDIO_LENGTH_PER_TOK) *
            1000.0 / VOXTRAL_SAMPLE_RATE)
        : 0.0;
    js << "\"decoderFirstWrapAudioMs\":" << firstWrapAudioMs << ",";
    js << "\"decoderPreWrapP99Ms\":" << r.runtimeProfile.decoderPreWrapP99Ms << ",";
    js << "\"decoderWrapStepMs\":" << r.runtimeProfile.decoderWrapStepMs << ",";
    js << "\"decoderPostWrapP99Ms\":" << r.runtimeProfile.decoderPostWrapP99Ms << ",";
    js << "\"argmaxEmbeddedInDecoderGraph\":true,";
    js << "\"modelLoadedVramBytes\":" << r.modelLoadedVramBytes << ",";
    js << "\"streamIdleVramBytes\":" << r.streamIdleVramBytes << ",";
    js << "\"afterFinishVramBytes\":" << r.afterFinishVramBytes << ",";
    js << "\"afterDestroyVramBytes\":" << r.afterDestroyVramBytes << ",";
    js << "\"modelLoadedRssKiB\":" << r.modelLoadedRssKiB << ",";
    js << "\"streamIdleRssKiB\":" << r.streamIdleRssKiB << ",";
    js << "\"afterFinishRssKiB\":" << r.afterFinishRssKiB << ",";
    js << "\"afterDestroyRssKiB\":" << r.afterDestroyRssKiB << ",";
    js << "\"memoryAttribution\":{";
    js << "\"mappingCategoriesMayOverlap\":true,";
    js << "\"vulkanAllocationCountAvailable\":false,";
    js << "\"hostAllocationCountAvailable\":false,";
    js << "\"individualBackendBufferCountAvailable\":false},";
    js << "\"memorySnapshots\":{";
    js << "\"modelLoaded\":"; write_memory_snapshot(js, r.modelLoadedMemory); js << ",";
    js << "\"beforeWarmup\":"; write_memory_snapshot(js, r.beforeWarmupMemory); js << ",";
    js << "\"afterWarmup\":"; write_memory_snapshot(js, r.afterWarmupMemory); js << ",";
    js << "\"afterFinish\":"; write_memory_snapshot(js, r.afterFinishMemory); js << ",";
    js << "\"afterDestroy\":"; write_memory_snapshot(js, r.afterDestroyMemory); js << ",";
    js << "\"afterMallocTrim\":"; write_memory_snapshot(js, r.afterMallocTrimMemory);
    js << "},";
    js << "\"mallocTrimRequested\":" << (r.mallocTrimRequested ? "true" : "false") << ",";
    js << "\"mallocTrimAvailable\":" << (r.mallocTrimAvailable ? "true" : "false") << ",";
    js << "\"mallocTrimApplied\":" << (r.mallocTrimApplied ? "true" : "false") << ",";
    js << "\"mallocTrimReturn\":" << r.mallocTrimReturn << ",";
    js << "\"rolloverMemory\":[";
    for (size_t i = 0; i < r.rolloverMemory.size(); ++i) {
        if (i) js << ",";
        const auto & sample = r.rolloverMemory[i];
        js << "{";
        js << "\"wrap\":" << sample.wrap << ",";
        js << "\"absolutePosition\":" << sample.absolutePosition << ",";
        js << "\"settledAfterDecoderSteps\":"
           << sample.settledAfterDecoderSteps << ",";
        js << "\"settledCaptured\":"
           << (sample.settledCaptured ? "true" : "false") << ",";
        js << "\"before\":"; write_memory_snapshot(js, sample.before); js << ",";
        js << "\"after\":"; write_memory_snapshot(js, sample.after); js << ",";
        js << "\"settled\":"; write_memory_snapshot(js, sample.settled); js << ",";
        js << "\"graphObjects\":" << sample.graphObjects << ",";
        js << "\"graphAllocations\":" << sample.graphAllocations << ",";
        js << "\"encoderAllocations\":" << sample.encoderAllocations << ",";
        js << "\"adapterAllocations\":" << sample.adapterAllocations << ",";
        js << "\"decoderAllocations\":" << sample.decoderAllocations << ",";
        js << "\"decoderKvBytesMoved\":" << sample.decoderKvBytesMoved << ",";
        js << "\"decoderKvFullBufferMoves\":"
           << sample.decoderKvFullBufferMoves;
        js << "}";
    }
    js << "],";
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
    js << "\"postInputDrainMs\":" << r.postInputDrainMs << ",";
    js << "\"terminalPartialChunkSamples\":"
       << r.terminalPartialChunkSamples << ",";
    js << "\"terminalPartialChunkMs\":" << r.terminalPartialChunkMs << ",";
    js << "\"terminalPartialFinishLatenessMs\":"
       << r.terminalPartialFinishLatenessMs << ",";
    js << "\"deadlineMisses\":" << r.deadlineMisses << ",";
    js << "\"deadlineMissRate\":" << r.deadlineMissRate << ",";
    auto write_backlog = [&](const char * name, const voxtral_backlog_metrics & m) {
        js << "\"" << name << "\":{";
        js << "\"count\":" << m.count << ",";
        js << "\"p50Ms\":" << m.p50Ms << ",";
        js << "\"p95Ms\":" << m.p95Ms << ",";
        js << "\"p99Ms\":" << m.p99Ms << ",";
        js << "\"maxMs\":" << m.maxMs << ",";
        js << "\"finalMs\":" << m.finalMs << ",";
        js << "\"slopeMsPerSec\":" << m.slopeMsPerSec << ",";
        js << "\"deadlineMisses\":" << m.deadlineMisses << ",";
        js << "\"deadlineMissRate\":" << m.deadlineMissRate << "},";
    };
    write_backlog("encoderBacklog", r.encoderBacklog);
    write_backlog("adapterBacklog", r.adapterBacklog);
    write_backlog("decoderBacklog", r.decoderBacklog);
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
    js << "\"firstDecoderStepEligibilityMs\":"
       << r.firstDecoderStepEligibilityMs << ",";
    js << "\"firstDecoderStepOverheadMs\":"
       << r.firstDecoderStepOverheadMs << ",";
    js << "\"firstTokenEligibilityMs\":" << r.firstTokenEligibilityMs << ",";
    js << "\"firstTokenOverheadMs\":" << r.firstTokenOverheadMs << ",";
    js << "\"firstPartialEligibilityMs\":" << r.firstPartialEligibilityMs << ",";
    js << "\"firstPartialOverheadMs\":" << r.firstPartialOverheadMs << ",";
    js << "\"adapterInputD2hBytes\":" << r.adapterInputD2hBytes << ",";
    js << "\"adapterOutputD2hBytes\":" << r.adapterOutputD2hBytes << ",";
    js << "\"logitsD2hBytes\":" << r.logitsD2hBytes << ",";
    js << "\"tokenIdD2hBytes\":" << r.tokenIdD2hBytes << ",";
    js << "\"encoderOutputD2hBytes\":" << r.encoderOutputD2hBytes << ",";
    js << "\"partialTextRevision\":" << r.partialTextRevision << ",";
    js << "\"eventHistoryRetained\":"
       << (r.eventHistoryRetained ? "true" : "false") << ",";
    js << "\"retainedEventHistoryCount\":"
       << r.retainedEventHistoryCount << ",";
    js << "\"tokenOutputBytes\":" << r.tokenOutputBytes << ",";
    js << "\"transcriptOutputBytes\":" << r.transcriptOutputBytes << ",";
    // Session 7.1: decoder mode + event-queue telemetry + backpressure evidence.
    js << "\"decoderMode\":\"" << r.decoderMode << "\",";
    js << "\"eventsEmitted\":" << r.eventsEmitted << ",";
    js << "\"tokenEvents\":" << r.tokenEventsCount << ",";
    js << "\"partialEvents\":" << r.partialEventsCount << ",";
    js << "\"partialEventsCoalesced\":" << r.partialEventsCoalesced << ",";
    js << "\"eventQueueHighWatermark\":" << r.eventQueueHighWatermark << ",";
    js << "\"eventQueueOverflowAttempts\":" << r.eventQueueOverflowAttempts << ",";
    js << "\"eventsDropped\":" << r.eventsDropped << ",";
    js << "\"maxEventsBound\":" << r.maxEventsBound << ",";
    js << "\"feedQueueFullReturns\":" << r.feedQueueFullReturns << ",";
    js << "\"backpressureObserved\":" << (r.backpressureObserved ? "true" : "false") << ",";
    js << "\"lastFeedStatus\":\"" << r.lastFeedStatus << "\",";
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
               << ",\"piece\":\"" << json_escape(e.text) << "\""
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

// ============================================================================
// Session 9 — production lifecycle hardening.
//
// Reuse ONE owned context/stream across many sequential short streams via
// reset(), proving VRAM/RSS plateau, byte-identical token output every
// iteration (no stale KV / token history / event sequence), and a documented
// reset->created transition. Then exercise the lifecycle edge cases against the
// state-machine contract (finish idempotency, feed-after-finish rejection,
// cancel, reset-after-terminal, destroy under backpressure). Emits one JSON
// line for the Node acceptance harness.
// ============================================================================
int run_sequential_streams(voxtral_model * model,
                           const voxtral_context_params & cp,
                           voxtral_stream_params sp,
                           const std::vector<int16_t> & full_pcm,
                           int32_t iterations,
                           int32_t short_samples) {
    const size_t K = std::min<size_t>((size_t) std::max(1, short_samples), full_pcm.size());
    const std::vector<int16_t> pcm(full_pcm.begin(), full_pcm.begin() + K);
    const size_t chunk = (size_t) 80 * VOXTRAL_SAMPLE_RATE / 1000;   // 80 ms = 1280 samples
    std::vector<size_t> counts;
    for (size_t o = 0; o < K; o += chunk) counts.push_back(std::min<size_t>(chunk, K - o));

    DriveOptions opt;
    opt.paced = false;
    opt.check_parity = false;
    opt.check_manual_oracle = false;
    opt.retain_event_history = false;

    voxtral_stream * stream = voxtral_stream_create_internal(model, cp, sp);
    if (!stream) { std::cerr << "sequential: stream create failed\n"; return 5; }
    voxtral_stream_warmup_internal(stream);   // prepare graphs/shaders once

    struct Iter {
        int32_t i; size_t tokens; int64_t vram; int64_t rss;
        uint64_t events; std::string stateAfterFinish;
        std::string resetStatus; bool resetPristine; bool tokensMatchFirst;
    };
    std::vector<Iter> iters; iters.reserve(iterations);
    std::vector<int32_t> first_tokens;
    bool all_tokens_consistent = true, all_reset_pristine = true, all_reset_ok = true;

    for (int32_t i = 0; i < iterations; ++i) {
        StreamRun run = drive_stream(stream, pcm, counts, opt);
        Iter it{};
        it.i = i;
        it.tokens = run.tokens.size();
        it.vram = read_vram_used_bytes();
        it.rss = read_process_rss_kib();
        it.events = voxtral_stream_events_emitted(stream);
        it.stateAfterFinish = voxtral_stream_state_name(voxtral_stream_get_state(stream));
        if (i == 0) first_tokens = run.tokens;
        it.tokensMatchFirst = (run.tokens == first_tokens);
        if (!it.tokensMatchFirst) all_tokens_consistent = false;

        const voxtral_status rs = voxtral_stream_reset_internal(stream);
        it.resetStatus = voxtral_stream_status_name(rs);
        if (rs != voxtral_status::ok) all_reset_ok = false;
        voxtral_stream_event ev;
        it.resetPristine = rs == voxtral_status::ok &&
            voxtral_stream_get_state(stream) == voxtral_stream_state::created &&
            voxtral_stream_samples_received(stream) == 0 &&
            voxtral_stream_samples_consumed(stream) == 0 &&
            voxtral_stream_tokens(stream).empty() &&
            voxtral_stream_transcript(stream).empty() &&
            voxtral_stream_events_emitted(stream) == 0 &&
            !voxtral_stream_poll_event(stream, ev);
        if (!it.resetPristine) all_reset_pristine = false;
        iters.push_back(it);
    }
    voxtral_stream_destroy_internal(stream);

    // Plateau over the last min(20, N) iterations (excludes warmup fault-in).
    const size_t tail = std::min<size_t>(20, iters.size());
    int64_t vram_lo = INT64_MAX, vram_hi = 0, rss_lo = INT64_MAX, rss_hi = 0;
    for (size_t k = iters.size() - tail; k < iters.size(); ++k) {
        vram_lo = std::min(vram_lo, iters[k].vram); vram_hi = std::max(vram_hi, iters[k].vram);
        rss_lo  = std::min(rss_lo,  iters[k].rss);  rss_hi  = std::max(rss_hi,  iters[k].rss);
    }

    // ---- lifecycle edge cases against the documented state machine ----
    struct Edge { std::string name; bool pass; std::string detail; };
    std::vector<Edge> edges; bool all_edges_pass = true;
    auto add_edge = [&](const std::string & name, bool pass, const std::string & detail) {
        edges.push_back({name, pass, detail});
        if (!pass) all_edges_pass = false;
    };

    // (a) finish idempotency + feed-after-finish rejection.
    {
        voxtral_stream * s = voxtral_stream_create_internal(model, cp, sp);
        for (size_t o = 0; o < pcm.size(); o += chunk) {
            const size_t c = std::min<size_t>(chunk, pcm.size() - o);
            voxtral_stream_feed_pcm16_internal(s, pcm.data() + o, c);
        }
        const voxtral_status f1 = voxtral_stream_finish_internal(s);
        const voxtral_status f2 = voxtral_stream_finish_internal(s);   // idempotent -> ok
        const int16_t one = 0;
        const voxtral_status fa = voxtral_stream_feed_pcm16_internal(s, &one, 1);  // -> err
        add_edge("finish_twice_idempotent", f1 == voxtral_status::ok && f2 == voxtral_status::ok,
                 std::string(voxtral_stream_status_name(f1)) + "/" + voxtral_stream_status_name(f2));
        add_edge("feed_after_finish_rejected", fa != voxtral_status::ok,
                 voxtral_stream_status_name(fa));
        voxtral_stream_destroy_internal(s);
    }
    // (b) cancel mid-stream, then finish (no inference) and reset -> created.
    {
        voxtral_stream * s = voxtral_stream_create_internal(model, cp, sp);
        if (!pcm.empty()) voxtral_stream_feed_pcm16_internal(s, pcm.data(), std::min<size_t>(chunk, pcm.size()));
        const voxtral_status cx = voxtral_stream_cancel_internal(s);
        const bool cancelled = voxtral_stream_get_state(s) == voxtral_stream_state::cancelled;
        const voxtral_status fin = voxtral_stream_finish_internal(s);   // ok, no inference
        const voxtral_status rs = voxtral_stream_reset_internal(s);
        add_edge("cancel_sets_cancelled", cx == voxtral_status::ok && cancelled,
                 voxtral_stream_status_name(cx));
        add_edge("finish_after_cancel_ok", fin == voxtral_status::ok, voxtral_stream_status_name(fin));
        add_edge("reset_after_cancel_created",
                 rs == voxtral_status::ok && voxtral_stream_get_state(s) == voxtral_stream_state::created,
                 voxtral_stream_status_name(rs));
        voxtral_stream_destroy_internal(s);
    }
    // (c) destroy under backpressure: fill the bounded event queue, never drain,
    //     then destroy. Reaching the next line = no crash / no deadlock.
    {
        voxtral_stream * s = voxtral_stream_create_internal(model, cp, sp);
        voxtral_stream_test_set_max_events(s, 1);
        for (size_t o = 0; o < pcm.size(); o += chunk) {
            const size_t c = std::min<size_t>(chunk, pcm.size() - o);
            voxtral_stream_feed_pcm16_internal(s, pcm.data() + o, c);   // no poll -> queue fills
        }
        voxtral_stream_finish_internal(s);
        voxtral_stream_destroy_internal(s);   // destroy without draining
        add_edge("destroy_under_backpressure_no_crash", true, "reached");
    }
    // (d) reset from created / double reset is a safe no-op.
    {
        voxtral_stream * s = voxtral_stream_create_internal(model, cp, sp);
        const voxtral_status r1 = voxtral_stream_reset_internal(s);
        const voxtral_status r2 = voxtral_stream_reset_internal(s);
        add_edge("reset_from_created_idempotent",
                 r1 == voxtral_status::ok && r2 == voxtral_status::ok,
                 std::string(voxtral_stream_status_name(r1)) + "/" + voxtral_stream_status_name(r2));
        voxtral_stream_destroy_internal(s);
    }

    const bool overall = all_tokens_consistent && all_reset_pristine && all_reset_ok && all_edges_pass;
    std::ostringstream js;
    js << "{";
    js << "\"mode\":\"sequential-streams\",";
    js << "\"iterations\":" << iterations << ",";
    js << "\"shortSamples\":" << K << ",";
    js << "\"tokensPerStream\":" << (iters.empty() ? 0 : iters[0].tokens) << ",";
    js << "\"allTokensConsistent\":" << (all_tokens_consistent ? "true" : "false") << ",";
    js << "\"allResetPristine\":" << (all_reset_pristine ? "true" : "false") << ",";
    js << "\"allResetOk\":" << (all_reset_ok ? "true" : "false") << ",";
    js << "\"tailWindow\":" << tail << ",";
    js << "\"vramTailLoBytes\":" << vram_lo << ",\"vramTailHiBytes\":" << vram_hi << ",";
    js << "\"vramTailRangeBytes\":" << (vram_hi - vram_lo) << ",";
    js << "\"rssTailLoKiB\":" << rss_lo << ",\"rssTailHiKiB\":" << rss_hi << ",";
    js << "\"rssTailRangeKiB\":" << (rss_hi - rss_lo) << ",";
    js << "\"firstVramBytes\":" << (iters.empty() ? 0 : iters.front().vram) << ",";
    js << "\"lastVramBytes\":" << (iters.empty() ? 0 : iters.back().vram) << ",";
    js << "\"firstRssKiB\":" << (iters.empty() ? 0 : iters.front().rss) << ",";
    js << "\"lastRssKiB\":" << (iters.empty() ? 0 : iters.back().rss) << ",";
    js << "\"edges\":[";
    for (size_t e = 0; e < edges.size(); ++e) {
        if (e) js << ",";
        js << "{\"name\":\"" << edges[e].name << "\",\"pass\":" << (edges[e].pass ? "true" : "false")
           << ",\"detail\":\"" << edges[e].detail << "\"}";
    }
    js << "],";
    js << "\"allEdgesPass\":" << (all_edges_pass ? "true" : "false") << ",";
    js << "\"iters\":[";
    for (size_t k = 0; k < iters.size(); ++k) {
        if (k) js << ",";
        js << "{\"i\":" << iters[k].i << ",\"tokens\":" << iters[k].tokens
           << ",\"vram\":" << iters[k].vram << ",\"rss\":" << iters[k].rss
           << ",\"events\":" << iters[k].events
           << ",\"stateAfterFinish\":\"" << iters[k].stateAfterFinish << "\""
           << ",\"resetStatus\":\"" << iters[k].resetStatus << "\""
           << ",\"resetPristine\":" << (iters[k].resetPristine ? "true" : "false")
           << ",\"tokensMatchFirst\":" << (iters[k].tokensMatchFirst ? "true" : "false") << "}";
    }
    js << "],";
    js << "\"state\":\"" << (overall ? "completed" : "failed") << "\",";
    js << "\"ok\":" << (overall ? "true" : "false");
    js << "}";
    std::cout << js.str() << "\n";
    return overall ? 0 : 1;
}

} // namespace

int main(int argc, char ** argv) {
    const auto process_started_at = std::chrono::steady_clock::now();
    std::string model_path, wav_path, plan_file, mode = "full";
    voxtral_gpu_backend gpu = voxtral_gpu_backend::auto_detect;
    int32_t max_tokens = 0;
    int32_t realtime_ms = 0;
    bool skip_parity = false;
    bool manual_oracle = false;
    bool ab = false;
    bool kv_parity = false;
    bool warmup = false;
    int32_t max_events = 0;
    bool backpressure = false;
    bool capture_rollover_memory = false;
    bool malloc_trim_after = false;
    bool discard_event_history = false;
    int32_t synthetic_seconds = 0;    // >0: generate synthetic audio instead of reading --wav
    uint64_t max_total_samples = 0;   // >0: override the stream's full-buffer sample cap
    int32_t sequential_streams = 0;   // >0: reuse one context/stream across N short streams
    int32_t sequential_samples = 48000; // short-clip length per sequential iteration (3 s)

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
        else if (a == "--kv-parity")   { kv_parity = true; }
        else if (a == "--warmup")      { warmup = true; }
        else if (a == "--max-events")  { const char * v = val("--max-events"); if (!v) return 2; max_events = int32_t(std::strtol(v, nullptr, 10)); }
        else if (a == "--backpressure") { backpressure = true; }
        else if (a == "--capture-rollover-memory") { capture_rollover_memory = true; }
        else if (a == "--malloc-trim-after") { malloc_trim_after = true; }
        else if (a == "--discard-event-history") { discard_event_history = true; }
        else if (a == "--synthetic-seconds") { const char * v = val("--synthetic-seconds"); if (!v) return 2; synthetic_seconds = int32_t(std::strtol(v, nullptr, 10)); }
        else if (a == "--max-total-samples") { const char * v = val("--max-total-samples"); if (!v) return 2; max_total_samples = std::strtoull(v, nullptr, 10); }
        else if (a == "--sequential-streams") { const char * v = val("--sequential-streams"); if (!v) return 2; sequential_streams = int32_t(std::strtol(v, nullptr, 10)); }
        else if (a == "--sequential-samples") { const char * v = val("--sequential-samples"); if (!v) return 2; sequential_samples = int32_t(std::strtol(v, nullptr, 10)); }
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

    if (model_path.empty() || (wav_path.empty() && synthetic_seconds <= 0)) {
        std::cerr << "usage: voxtral-stream-test --model M.gguf --wav in.wav "
                     "[--gpu ...] [--plan-file f] [--mode m] [--max-tokens n] [--realtime-ms n] [--warmup] [--ab] [--kv-parity]\n"
                     "       [--max-events n] [--backpressure] [--capture-rollover-memory] [--malloc-trim-after] [--discard-event-history]\n"
                     "       [--synthetic-seconds n] [--max-total-samples n]\n";
        return 2;
    }
    if (synthetic_seconds > 0) skip_parity = true;   // no batch reference for a 10-min soak

    // Read audio, or synthesize a long deterministic clip for the soak (avoids
    // transferring a ~19 MB WAV). Low-amplitude mixed tones keep the encoder doing
    // real work without depending on a fixture.
    std::vector<int16_t> pcm16;
    std::string err;
    if (synthetic_seconds > 0) {
        const size_t n = (size_t) synthetic_seconds * VOXTRAL_SAMPLE_RATE;
        pcm16.resize(n);
        uint32_t st = 0x1234567u;
        for (size_t i = 0; i < n; ++i) {
            const double t = (double) i / VOXTRAL_SAMPLE_RATE;
            const double s = 0.06 * std::sin(2.0 * 3.14159265 * 180.0 * t)
                           + 0.04 * std::sin(2.0 * 3.14159265 * 320.0 * t);
            const double noise = ((double) (seeded_next(st) & 0xffff) / 65535.0 - 0.5) * 0.01;
            pcm16[i] = (int16_t) std::lround(std::max(-1.0, std::min(1.0, s + noise)) * 32000.0);
        }
    } else if (!read_wav_pcm16_mono16k(wav_path, pcm16, err)) {
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
    drive_options.max_events = max_events;
    drive_options.backpressure = backpressure;
    drive_options.capture_numerical = kv_parity;
    drive_options.capture_rollover_memory = capture_rollover_memory;
    drive_options.retain_event_history = !discard_event_history;

    // Load the model ONCE (shared, immutable). Each stream will create and own
    // its own execution context from it via voxtral_stream_create_internal.
    voxtral_log_callback logger = log_stderr;
    const auto model_load_started_at = std::chrono::steady_clock::now();
    voxtral_model * model = voxtral_model_load_from_file(model_path, logger, gpu);
    if (!model) { std::cerr << "failed to load model\n"; return 5; }
    const double model_load_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - model_load_started_at).count();
    const int64_t model_loaded_vram_bytes = read_vram_used_bytes();
    const int64_t model_loaded_rss_kib = read_process_rss_kib();
    const ProcessMemorySnapshot model_loaded_memory =
        read_process_memory_snapshot();

    voxtral_context_params cp;
    cp.log_level = voxtral_log_level::info;
    cp.logger    = logger;
    cp.gpu       = gpu;

    voxtral_stream_params sp;
    sp.max_tokens = max_tokens;
    sp.retain_mel_history = !skip_parity || manual_oracle;
    if (max_total_samples > 0) sp.max_total_samples = max_total_samples;

    // Session 9 production lifecycle / sequential-reuse harness (own dispatch).
    if (sequential_streams > 0) {
        return run_sequential_streams(model, cp, sp, pcm16, sequential_streams, sequential_samples);
    }

    int exit_code = 0;
    std::ostringstream js;

    auto apply_warmup = [&](voxtral_stream * stream, StreamRun & run) -> bool {
        if (!warmup) return true;
        const auto start = std::chrono::steady_clock::now();
        const voxtral_status status = voxtral_stream_warmup_internal(stream);
        run.warmupMs = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        run.warmupApplied = status == voxtral_status::ok;
        if (status != voxtral_status::ok) {
            std::cerr << "warmup failed: " << voxtral_stream_status_name(status)
                      << " (" << voxtral_stream_last_error(stream) << ")\n";
            return false;
        }
        // Warmup is a preparation operation, not a synthetic utterance. Lock
        // down that contract here so shader/graph preparation can never become
        // an invisible source of token, transcript, event or audio state.
        voxtral_stream_event warmup_event;
        const bool pristine =
            voxtral_stream_get_state(stream) == voxtral_stream_state::created &&
            voxtral_stream_samples_received(stream) == 0 &&
            voxtral_stream_samples_consumed(stream) == 0 &&
            voxtral_stream_tokens(stream).empty() &&
            voxtral_stream_transcript(stream).empty() &&
            voxtral_stream_events_emitted(stream) == 0 &&
            !voxtral_stream_poll_event(stream, warmup_event);
        if (!pristine) {
            std::cerr << "warmup polluted fresh stream state\n";
            return false;
        }
        if (voxtral_stream_warmup_internal(stream) != voxtral_status::ok) {
            std::cerr << "warmup is not idempotent\n";
            return false;
        }
        return true;
    };

    if (kv_parity) {
        // Keep only one execution context resident at a time (RX 6600 VRAM),
        // while sharing the immutable model across all plans. Test-only ring
        // and graph selectors must not leak into this production-parity mode.
        unsetenv("VOXTRAL_DECODER_KV_TEST_CAPACITY");
        unsetenv("VOXTRAL_DECODER_STEP_GRAPH");
        unsetenv("VOXTRAL_DECODER_RING_ATTENTION");
        auto run_variant = [&](const char * decoder_mode,
                               const char * encoder_kv_type,
                               const char * decoder_kv_type,
                               const char * physical_rows) {
            setenv("VOXTRAL_NUMERICAL_DIAGNOSTICS", "1", 1);
            setenv("VOXTRAL_STREAM_DECODER", decoder_mode, 1);
            setenv("VOXTRAL_ENCODER_KV_TYPE", encoder_kv_type, 1);
            setenv("VOXTRAL_DECODER_KV_TYPE", decoder_kv_type, 1);
            setenv("VOXTRAL_ENC_KV_LOGICAL_BATCH", "4", 1);
            setenv("VOXTRAL_ENC_KV_PHYSICAL_ROWS", physical_rows, 1);
            voxtral_stream * stream = voxtral_stream_create_internal(model, cp, sp);
            if (!stream) return StreamRun{};
            StreamRun run;
            if (apply_warmup(stream, run)) {
                const bool warmed = run.warmupApplied;
                const double warmup_ms = run.warmupMs;
                run = drive_stream(stream, pcm16, counts, drive_options);
                run.warmupApplied = warmed;
                run.warmupMs = warmup_ms;
            }
            voxtral_stream_destroy_internal(stream);
            return run;
        };

        StreamRun production = run_variant("incremental", "f16", "f16", "4");
        StreamRun fp16_reference = run_variant("reference", "f16", "f16", "4");
        StreamRun f32_same_shape = run_variant("reference", "f32", "f32", "4");
        StreamRun f32_production = run_variant("incremental", "f32", "f32", "4");
        StreamRun f32_reference = run_variant("reference", "f32", "f32", "32");

        // Tensor deltas use the exact production reusable topology in both
        // storage modes. The finish-only plans below remain independent token
        // and transcript oracles, rather than standing in for production math.
        const TensorComparison encoder_cmp = compare_tensors(
            production.numericalEncoderOutput,
            f32_production.numericalEncoderOutput);
        const TensorComparison adapter_cmp = compare_tensors(
            production.numericalAdapterOutput,
            f32_production.numericalAdapterOutput);
        const TensorComparison hidden_cmp = compare_tensors(
            production.numericalDecoderHidden,
            f32_production.numericalDecoderHidden);
        const TensorComparison logits_cmp = compare_tensors(
            production.numericalDecoderLogits,
            f32_production.numericalDecoderLogits);
        const bool token_parity = production.tokens == f32_reference.tokens &&
                                  fp16_reference.tokens == f32_reference.tokens &&
                                  f32_same_shape.tokens == f32_reference.tokens &&
                                  f32_production.tokens == f32_reference.tokens;
        const bool transcript_parity = production.text == f32_reference.text &&
                                       fp16_reference.text == f32_reference.text &&
                                       f32_same_shape.text == f32_reference.text &&
                                       f32_production.text == f32_reference.text;

        auto write_comparison = [&](const TensorComparison & cmp) {
            js << "{\"available\":" << (cmp.ok ? "true" : "false")
               << ",\"elements\":" << cmp.elements
               << ",\"maxAbsDelta\":" << cmp.max_abs
               << ",\"meanAbsDelta\":" << cmp.mean_abs
               << ",\"rmsDelta\":" << cmp.rms_delta
               << ",\"referenceRms\":" << cmp.reference_rms
               << ",\"normalizedRms\":"
               << (cmp.reference_rms > 0.0 ? cmp.rms_delta / cmp.reference_rms : 0.0)
               << ",\"cosineSimilarity\":" << cmp.cosine << "}";
        };

        js << "{";
        js << "\"mode\":\"kv-parity\",";
        js << "\"modelShared\":true,";
        js << "\"execution\":\"sequential\",";
        js << "\"sampleRate\":" << sp.sample_rate << ",";
        js << "\"wavSamples\":" << total << ",";
        js << "\"production\":{"; write_run_fields(js, production); js << "},";
        js << "\"fp16Reference\":{"; write_run_fields(js, fp16_reference); js << "},";
        js << "\"f32SameShape\":{"; write_run_fields(js, f32_same_shape); js << "},";
        js << "\"f32Production\":{"; write_run_fields(js, f32_production); js << "},";
        js << "\"f32Reference\":{"; write_run_fields(js, f32_reference); js << "},";
        js << "\"numerical\":{";
        js << "\"encoder\":"; write_comparison(encoder_cmp); js << ",";
        js << "\"adapter\":"; write_comparison(adapter_cmp); js << ",";
        js << "\"decoderHidden\":"; write_comparison(hidden_cmp); js << ",";
        js << "\"decoderLogits\":"; write_comparison(logits_cmp); js << "},";
        js << "\"tokenParity\":" << (token_parity ? "true" : "false") << ",";
        js << "\"transcriptParity\":" << (transcript_parity ? "true" : "false");
        js << "}";

        exit_code = (production.ok && fp16_reference.ok && f32_same_shape.ok &&
                     f32_production.ok && f32_reference.ok &&
                     encoder_cmp.ok && adapter_cmp.ok && hidden_cmp.ok && logits_cmp.ok &&
                     token_parity && transcript_parity) ? 0 : 1;
    } else if (ab) {
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

        StreamRun run_a, run_b;
        if (apply_warmup(a, run_a)) {
            const double ms = run_a.warmupMs; const bool applied = run_a.warmupApplied;
            run_a = drive_stream(a, pcm16, counts, drive_options);
            run_a.warmupMs = ms; run_a.warmupApplied = applied;
        }
        if (apply_warmup(b, run_b)) {
            const double ms = run_b.warmupMs; const bool applied = run_b.warmupApplied;
            run_b = drive_stream(b, pcm16, counts, drive_options);
            run_b.warmupMs = ms; run_b.warmupApplied = applied;
        }

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
        const auto warm_model_started_at = std::chrono::steady_clock::now();
        const auto context_create_started_at = std::chrono::steady_clock::now();
        voxtral_stream * stream = voxtral_stream_create_internal(model, cp, sp);
        if (!stream) { std::cerr << "failed to create stream\n"; voxtral_model_free(model); return 7; }
        const double context_creation_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - context_create_started_at).count();

        const int64_t stream_idle_vram_bytes = read_vram_used_bytes();
        const int64_t stream_idle_rss_kib = read_process_rss_kib();
        const ProcessMemorySnapshot before_warmup_memory =
            read_process_memory_snapshot();
        StreamRun run;
        if (!apply_warmup(stream, run)) {
            voxtral_stream_destroy_internal(stream);
            voxtral_model_free(model);
            return 8;
        }
        const double warmup_ms = run.warmupMs;
        const bool warmup_applied = run.warmupApplied;
        const ProcessMemorySnapshot after_warmup_memory =
            read_process_memory_snapshot();
        const double pre_drive_process_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - process_started_at).count();
        const double pre_drive_warm_model_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - warm_model_started_at).count();
        run = drive_stream(stream, pcm16, counts, drive_options);
        run.warmupMs = warmup_ms;
        run.warmupApplied = warmup_applied;
        run.modelLoadMs = model_load_ms;
        run.contextCreationMs = context_creation_ms;
        run.coldFirstDecoderStepMs = pre_drive_process_ms + run.firstDecoderStepMs;
        run.coldFirstTokenMs = pre_drive_process_ms + run.firstTokenMs;
        run.coldFirstVisibleTextMs = pre_drive_process_ms + run.firstVisibleTextMs;
        run.warmModelFirstDecoderStepMs = pre_drive_warm_model_ms + run.firstDecoderStepMs;
        run.warmModelFirstTokenMs = pre_drive_warm_model_ms + run.firstTokenMs;
        run.warmModelFirstVisibleTextMs = pre_drive_warm_model_ms + run.firstVisibleTextMs;
        run.modelLoadedVramBytes = model_loaded_vram_bytes;
        run.streamIdleVramBytes = stream_idle_vram_bytes;
        run.afterFinishVramBytes = read_vram_used_bytes();
        run.modelLoadedRssKiB = model_loaded_rss_kib;
        run.streamIdleRssKiB = stream_idle_rss_kib;
        run.afterFinishRssKiB = read_process_rss_kib();
        run.modelLoadedMemory = model_loaded_memory;
        run.beforeWarmupMemory = before_warmup_memory;
        run.afterWarmupMemory = after_warmup_memory;
        run.afterFinishMemory = read_process_memory_snapshot();
        run.mallocTrimRequested = malloc_trim_after;

        // Destroy the stream (frees its owned context), then free the model.
        voxtral_stream_destroy_internal(stream);
        run.afterDestroyVramBytes = read_vram_used_bytes();
        run.afterDestroyRssKiB = read_process_rss_kib();
        run.afterDestroyMemory = read_process_memory_snapshot();
#if defined(__GLIBC__)
        run.mallocTrimAvailable = true;
        if (malloc_trim_after) {
            run.mallocTrimReturn = malloc_trim(0);
            run.mallocTrimApplied = true;
            run.afterMallocTrimMemory = read_process_memory_snapshot();
        }
#else
        run.mallocTrimAvailable = false;
#endif

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

// ============================================================================
// Model-free C++ tests for the incremental causal encoder SCHEDULE (voxtral.cpp).
//
// The end-to-end numerical parity of the incremental encoder against the batch
// encoder is proven on real hardware by the Node.js GPU acceptance suite
// (encoderMaxAbsDeltaVsBatch == 0 across every chunk plan). These model-free
// tests instead lock the SCHEDULE ARITHMETIC that makes that parity possible,
// with NO GGUF model or backend:
//
//   * the conv-stem frame math (voxtral_enc_frames_for_mel_internal), incl. the
//     trunc == 0 invariant that holds for the realtime padded stream;
//   * that the incremental chunk schedule (bounded-window recomputation,
//     replaying run_encoder_chunked) emits EXACTLY the batch encoder's frames —
//     same total, contiguous coverage, no lost or duplicated frame — for every
//     feed granularity and across encoder-window boundaries;
//   * that per-chunk work stays bounded (does not grow with stream duration).
//
// The batch and incremental schedules below are faithful reimplementations of
// run_encoder_chunked and encoder_stream_drive; both call the SAME real conv
// arithmetic (voxtral_enc_frames_for_mel_internal), so a drift in that function
// is caught here, and the GPU suite catches any drift in the graph itself.
//
// Assert-based; no external test framework. Exits non-zero on any failure.
// ============================================================================

#include "voxtral.h"
#include "voxtral-internal.h"
#include "voxtral-mel.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

static int g_checks   = 0;
static int g_failures = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        ++g_checks;                                                            \
        if (!(cond)) {                                                         \
            ++g_failures;                                                      \
            std::fprintf(stderr, "  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        }                                                                      \
    } while (0)

namespace {

// Constants mirror the streaming encoder in src/voxtral.cpp. Keep in sync.
constexpr int32_t CHUNK_MEL  = 3000;   // VOXTRAL_ENC_CHUNK_MEL
constexpr int32_t OVERLAP    = 750;    // VOXTRAL_ENC_CHUNK_OVERLAP (enc frames)
constexpr int32_t MEL_STRIDE = CHUNK_MEL - 2 * OVERLAP;  // 1500
constexpr int32_t EMIT_MEL   = 256;    // VOXTRAL_ENC_STREAM_EMIT_MEL
constexpr int32_t DOWNSAMPLE = VOXTRAL_DOWNSAMPLE_FACTOR;  // 4

int32_t enc_frames_for_mel(int32_t mel) { return voxtral_enc_frames_for_mel_internal(mel); }

struct Range { int64_t start; int64_t end; };

// Faithful reimplementation of run_encoder_chunked's emit loop (voxtral.cpp).
struct BatchPlan {
    int64_t raw  = 0;   // enc_write_offset (before the /4 trim)
    int64_t used = 0;   // (raw / 4) * 4  (== ctx.enc_seq_used)
    std::vector<Range> emits;
};
BatchPlan batch_plan(int64_t total_mel) {
    BatchPlan p;
    int64_t enc_write_offset = 0, mel_offset = 0;
    int32_t chunk_idx = 0;
    while (mel_offset < total_mel) {
        const int32_t chunk_mel = (int32_t) std::min<int64_t>(CHUNK_MEL, total_mel - mel_offset);
        const int32_t skip = (chunk_idx > 0) ? OVERLAP : 0;
        const int32_t expected = enc_frames_for_mel(chunk_mel);
        if (expected - skip <= 0) break;                 // pre-check
        const int32_t chunk_seq_len = expected;          // == build_encoder_graph seq_len
        const int32_t stride = chunk_seq_len - skip;
        if (stride <= 0) break;
        p.emits.push_back({enc_write_offset, enc_write_offset + stride});
        enc_write_offset += stride;
        mel_offset += MEL_STRIDE;
        chunk_idx++;
    }
    p.raw  = enc_write_offset;
    p.used = (enc_write_offset / DOWNSAMPLE) * DOWNSAMPLE;
    return p;
}

// Faithful reimplementation of encoder_stream_drive (voxtral.cpp), driven by a
// feed plan. `feeds` sum to `pre_finish` stable frames delivered during feed
// (drive(false) after each); the remaining (total - pre_finish) frames arrive at
// finish (drive(true)). Mirrors the real Mel frontend + right-pad handoff.
struct IncrPlan {
    int64_t raw = 0;              // emitted
    int64_t input_frames = 0;     // sum of chunk Mel lengths run (recompute-inclusive)
    int64_t executions = 0;
    int64_t before_finish = 0;    // enc frames emitted during feed
    int64_t at_finish = 0;        // enc frames emitted by finish
    std::vector<Range> emits;
};

struct IncrState {
    int32_t cur_chunk = 0;
    int64_t emitted = 0;
    int64_t last_run_mel_end = 0;
    int64_t stable = 0;
    IncrPlan p;
};

void incr_drive(IncrState & s, bool final) {
    for (;;) {
        const int64_t Lc = (int64_t) s.cur_chunk * MEL_STRIDE;
        if (Lc >= s.stable) break;
        const int64_t avail = s.stable - Lc;
        const bool complete = (avail >= CHUNK_MEL);
        const int32_t skip = (s.cur_chunk == 0) ? 0 : OVERLAP;

        int32_t run_len = 0;
        bool advance = false;
        if (complete) { run_len = CHUNK_MEL; advance = true; }
        else if (final) { run_len = (int32_t) avail; advance = false; }
        else if (s.cur_chunk != 0) { break; }
        else {
            const int32_t m8 = (int32_t) ((avail / 8) * 8);
            if (m8 <= 0) break;
            const int64_t need = std::max<int64_t>(EMIT_MEL, s.last_run_mel_end / 2);
            if ((int64_t) m8 - s.last_run_mel_end < need) break;
            run_len = m8; advance = false;
        }
        if (enc_frames_for_mel(run_len) - skip <= 0) break;

        const int32_t seq_len = enc_frames_for_mel(run_len);
        s.p.executions++;
        s.p.input_frames += run_len;
        s.last_run_mel_end = Lc + run_len;

        int64_t src = s.emitted - (int64_t) s.cur_chunk * OVERLAP;
        if (src < skip) src = skip;
        if ((int64_t) seq_len > src) {
            const int64_t n_new = (int64_t) seq_len - src;
            s.p.emits.push_back({s.emitted, s.emitted + n_new});
            s.emitted = (int64_t) s.cur_chunk * OVERLAP + seq_len;
            if (final) s.p.at_finish += n_new; else s.p.before_finish += n_new;
        }
        if (advance) { s.cur_chunk++; continue; }
        break;
    }
}

IncrPlan incr_plan(int64_t total_mel, int64_t pre_finish, const std::vector<int64_t> & feeds) {
    IncrState s;
    for (int64_t f : feeds) { s.stable += f; incr_drive(s, false); }
    // Anything not delivered during feed arrives at finish (right-pad tail).
    (void) pre_finish;
    s.stable = total_mel;
    incr_drive(s, true);
    s.p.raw = s.emitted;
    return s.p;
}

// Coverage check: ranges must tile [0, total) exactly — sorted, contiguous, no
// gap and no overlap (no lost or duplicated encoder frame).
bool tiles_exactly(std::vector<Range> emits, int64_t total) {
    std::sort(emits.begin(), emits.end(),
              [](const Range & a, const Range & b) { return a.start < b.start; });
    int64_t cursor = 0;
    for (const Range & r : emits) {
        if (r.start != cursor) return false;   // gap or overlap
        if (r.end < r.start) return false;
        cursor = r.end;
    }
    return cursor == total;
}

std::vector<int64_t> feeds_fixed(int64_t total, int64_t step) {
    std::vector<int64_t> v;
    for (int64_t off = 0; off < total; off += step) v.push_back(std::min<int64_t>(step, total - off));
    return v;
}

// -------------------- tests --------------------

void test_conv_arithmetic() {
    // conv0 (k3 s1) preserves length; conv1 (k3 s2) downsamples ~2x; trunc = %4.
    // Multiple-of-8 Mel lengths (the realtime padded stream) => trunc == 0 =>
    // enc frames == mel/2 exactly. This is the invariant the whole design leans on.
    for (int32_t k = 1; k <= 2000; ++k) {
        const int32_t mel = k * 8;
        CHECK(enc_frames_for_mel(mel) == mel / 2);
    }
    // Monotonic non-decreasing in Mel length.
    int32_t prev = 0;
    for (int32_t mel = 1; mel <= 6000; ++mel) {
        const int32_t e = enc_frames_for_mel(mel);
        CHECK(e >= prev);
        prev = e;
    }
    // A few hand-checked values (even lengths, trunc applied): mel -> conv1 = mel/2,
    // then drop conv1 % 4.
    CHECK(enc_frames_for_mel(2)  == 0);   // conv1=1, trunc 1 -> 0
    CHECK(enc_frames_for_mel(8)  == 4);   // conv1=4, trunc 0
    CHECK(enc_frames_for_mel(10) == 4);   // conv1=5, trunc 1 -> 4
    CHECK(enc_frames_for_mel(3000) == 1500);   // full chunk, trunc 0
    CHECK(enc_frames_for_mel(0)  == 0);
    CHECK(enc_frames_for_mel(-5) == 0);
}

void test_schedule_equivalence() {
    // For every plausible total (multiple of 8, as the padded realtime stream
    // guarantees) and every feed granularity, the incremental schedule must emit
    // EXACTLY the batch encoder's frames.
    const std::vector<int64_t> totals = {
        8, 16, 256, 512, 752, 1000, 1496, 1504, 1512,
        2992, 3000, 3008, 4488, 4496, 4504, 6000, 7504, 9000, 12000, 24000, 48000,
    };
    const std::vector<int64_t> steps = {1, 7, 8, 80, 256, 1000, 1499, 1500, 1501, 3000, 1'000'000};
    for (int64_t total : totals) {
        const BatchPlan bp = batch_plan(total);
        for (int64_t step : steps) {
            // Deliver most frames during feed; the last few emulate the right-pad
            // tail that only arrives at finish.
            const int64_t pre = (total > 16) ? (total - 8) : total;
            std::vector<int64_t> feeds = feeds_fixed(pre, step);
            const IncrPlan ip = incr_plan(total, pre, feeds);

            CHECK(ip.raw == bp.raw);                       // identical total
            CHECK((ip.raw / DOWNSAMPLE) * DOWNSAMPLE == bp.used);
            CHECK(tiles_exactly(ip.emits, ip.raw));        // no lost/duplicated frame
            CHECK(ip.before_finish + ip.at_finish == ip.raw);
            // For any non-trivial stream, something is emitted during feed.
            if (total >= 512) CHECK(ip.before_finish > 0);
        }
    }
}

void test_window_boundary_positions() {
    // Exercise encoder-frame positions around the sliding-window boundary
    // (OVERLAP = 750 enc frames) and the chunk stride boundaries. Totals chosen so
    // the last produced enc frame sits near 0,1,W-1,W,W+1,2W.
    const std::vector<int64_t> enc_targets = {0, 1, OVERLAP - 1, OVERLAP, OVERLAP + 1, 2 * OVERLAP, 3 * OVERLAP};
    for (int64_t enc : enc_targets) {
        const int64_t total = ((2 * enc + 8) / 8) * 8;   // enough Mel for `enc` enc frames, mult of 8
        const BatchPlan bp = batch_plan(total);
        for (int64_t step : {1, 8, 256, 1500}) {
            const int64_t pre = (total > 16) ? (total - 8) : total;
            const IncrPlan ip = incr_plan(total, pre, feeds_fixed(pre, step));
            CHECK(ip.raw == bp.raw);
            CHECK(tiles_exactly(ip.emits, ip.raw));
        }
    }
}

void test_work_bound() {
    // Work per second of audio must not grow with duration. Mel is 100 Hz, so
    // seconds*100 frames (padded to a multiple of 8).
    auto mel_for_seconds = [](int64_t sec) { return ((sec * 100) / 8) * 8; };
    auto measure = [](int64_t total) {
        const int64_t pre = (total > 16) ? (total - 8) : total;
        return incr_plan(total, pre, feeds_fixed(pre, 256));
    };
    auto ratio = [&](int64_t total) {
        return (double) measure(total).input_frames / (double) total;
    };

    // (a) BOUNDED: every stream's total encoder work is a small constant multiple
    // of the unique Mel (bounded-window recomputation, never O(duration^2)). The
    // worst case is a single fully-progressive chunk 0 (~30 s), still well bounded.
    const int64_t t5   = mel_for_seconds(5);
    const int64_t t30  = mel_for_seconds(30);
    const int64_t t120 = mel_for_seconds(120);
    CHECK(ratio(t5)   < 8.0);
    CHECK(ratio(t30)  < 8.0);
    CHECK(ratio(t120) < 8.0);

    // (b) DOES NOT GROW WITH DURATION: in steady state each additional MEL_STRIDE
    // of audio costs one bounded completion run, so the MARGINAL work per Mel is
    // constant (~CHUNK_MEL/MEL_STRIDE = 2x) and does not increase as the stream
    // lengthens. The chunk-0 progressive cost is a bounded one-time transient, so
    // the overall ratio if anything DECREASES for longer streams.
    const int64_t m60  = mel_for_seconds(60);
    const int64_t m120 = mel_for_seconds(120);
    const int64_t m240 = mel_for_seconds(240);
    const double in60  = (double) measure(m60).input_frames;
    const double in120 = (double) measure(m120).input_frames;
    const double in240 = (double) measure(m240).input_frames;
    const double marg_120_60  = (in120 - in60)  / (double) (m120 - m60);
    const double marg_240_120 = (in240 - in120) / (double) (m240 - m120);
    CHECK(marg_120_60  < 3.0);
    CHECK(marg_240_120 < 3.0);
    CHECK(marg_240_120 <= marg_120_60 + 0.05);          // marginal rate does not grow
    CHECK(ratio(t120) <= ratio(t30) + 0.01);            // long-stream ratio amortizes down

    // Executions scale ~linearly with duration (one completion per MEL_STRIDE),
    // never quadratically.
    CHECK(measure(t120).executions < (t120 / MEL_STRIDE) + 30);
}

void test_zero_and_tiny() {
    // Zero-length / sub-frame streams emit nothing and never crash.
    const IncrPlan z = incr_plan(0, 0, {});
    CHECK(z.raw == 0);
    CHECK(z.emits.empty());
    CHECK(z.executions == 0);

    // A stream too short to yield a single adapter group (needs >= 8 Mel) still
    // agrees with batch (both empty / tiny).
    for (int64_t total : {8, 16, 24}) {
        const BatchPlan bp = batch_plan(total);
        const IncrPlan ip = incr_plan(total, total, feeds_fixed(total, 8));
        CHECK(ip.raw == bp.raw);
        CHECK(tiles_exactly(ip.emits, ip.raw));
    }
}

} // namespace

int main() {
    test_conv_arithmetic();
    test_schedule_equivalence();
    test_window_boundary_positions();
    test_work_bound();
    test_zero_and_tiny();

    std::fprintf(stderr, "voxtral_encoder_unit: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}

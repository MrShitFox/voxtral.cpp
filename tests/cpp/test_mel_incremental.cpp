// ============================================================================
// Model-free C++ tests for the incremental log-Mel frontend (voxtral-mel).
//
// These verify, WITHOUT a GGUF model or any backend, that the incremental
// frontend is numerically identical to the batch spectrogram over the same
// signal for every chunk plan, that each Mel frame is computed exactly once, and
// that only a bounded PCM tail is retained regardless of stream duration.
//
// Assert-based; no external test framework. Exits non-zero on any failure.
// ============================================================================

#include "voxtral-mel.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
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

constexpr int32_t N_MEL = VOXTRAL_MEL_N_MEL;   // 128
constexpr int32_t N_FFT = VOXTRAL_MEL_N_FFT;   // 400
constexpr int32_t HOP   = VOXTRAL_MEL_HOP;     // 160

std::vector<float> g_hann;
std::vector<float> g_filters;

void init_tables() {
    voxtral_mel_build_hann_window(g_hann);
    voxtral_mel_build_slaney_filters(g_filters);
}

// mulberry32-style PRNG, matching tests/node/helpers/chunks.js for reproducibility.
struct Rng {
    uint32_t state;
    explicit Rng(uint32_t seed) : state(seed) {}
    double next() {
        state += 0x6d2b79f5u;
        uint32_t v = state;
        v = (v ^ (v >> 15)) * (v | 1u);
        v ^= v + (v ^ (v >> 7)) * (v | 61u);
        return (double) (uint32_t) (v ^ (v >> 14)) / 4294967296.0;
    }
};

// -------- signal generators --------
std::vector<float> sig_zeros(int32_t n)     { return std::vector<float>((size_t) std::max(0, n), 0.0f); }
std::vector<float> sig_constant(int32_t n, float v) { return std::vector<float>((size_t) std::max(0, n), v); }
std::vector<float> sig_impulse(int32_t n) {
    std::vector<float> s((size_t) std::max(0, n), 0.0f);
    if (n > 0) s[0] = 1.0f;
    if (n > 3) s[(size_t) (n / 2)] = -0.7f;
    return s;
}
std::vector<float> sig_alternating(int32_t n) {
    std::vector<float> s((size_t) std::max(0, n));
    for (int32_t i = 0; i < n; ++i) s[(size_t) i] = (i & 1) ? -1.0f : 1.0f;
    return s;
}
std::vector<float> sig_noise(int32_t n, uint32_t seed) {
    std::vector<float> s((size_t) std::max(0, n));
    Rng rng(seed);
    for (int32_t i = 0; i < n; ++i) s[(size_t) i] = (float) (rng.next() * 2.0 - 1.0);
    return s;
}
std::vector<float> sig_tone(int32_t n, float freq_hz, uint32_t seed) {
    std::vector<float> s((size_t) std::max(0, n));
    Rng rng(seed);
    for (int32_t i = 0; i < n; ++i) {
        const float t = (float) i / (float) VOXTRAL_SAMPLE_RATE;
        s[(size_t) i] = 0.6f * sinf(2.0f * VOXTRAL_MEL_PI * freq_hz * t)
                      + 0.2f * (float) (rng.next() * 2.0 - 1.0);
    }
    return s;
}

// -------- chunk plans (sum == total) --------
std::vector<size_t> plan_full(size_t total)   { std::vector<size_t> p; if (total) p.push_back(total); return p; }
std::vector<size_t> plan_single(size_t total) { return std::vector<size_t>(total, 1); }
std::vector<size_t> plan_fixed(size_t total, size_t k) {
    std::vector<size_t> p;
    for (size_t off = 0; off < total; off += k) p.push_back(std::min(k, total - off));
    return p;
}
std::vector<size_t> plan_irregular(size_t total) {
    static const size_t sizes[] = {1, 2, 3, 5, 8, 13, 21, 400, 159, 161, 1000, 7};
    std::vector<size_t> p;
    size_t off = 0, idx = 0;
    while (off < total) {
        size_t take = std::min(sizes[idx++ % (sizeof(sizes) / sizeof(sizes[0]))], total - off);
        p.push_back(take);
        off += take;
    }
    return p;
}
std::vector<size_t> plan_seeded(size_t total, uint32_t seed) {
    std::vector<size_t> p;
    Rng rng(seed);
    size_t off = 0;
    const size_t lo = 1, hi = 4096;
    while (off < total) {
        size_t want = lo + (size_t) (rng.next() * (double) (hi - lo + 1));
        size_t take = std::min(want, total - off);
        p.push_back(take);
        off += take;
    }
    return p;
}
// Interleave zero-length feeds into an existing plan.
std::vector<size_t> with_zero_feeds(const std::vector<size_t> & base) {
    std::vector<size_t> p;
    p.push_back(0);
    for (size_t i = 0; i < base.size(); ++i) {
        p.push_back(base[i]);
        if (i % 3 == 0) p.push_back(0);
    }
    p.push_back(0);
    return p;
}

// -------- batch reference --------
struct MelResult {
    std::vector<float> mel;   // channel-major [n_mel, n_frames]
    int32_t frames = 0;
};

MelResult run_batch(const std::vector<float> & sig) {
    MelResult r;
    const int32_t max_frames = voxtral_mel_batch_frame_count((int32_t) sig.size());
    r.mel.assign((size_t) N_MEL * std::max(0, max_frames), 0.0f);
    int32_t nf = 0;
    voxtral_mel_compute_batch(sig.data(), (int32_t) sig.size(), g_filters.data(), g_hann.data(),
                              r.mel.data(), &nf);
    r.frames = nf;
    r.mel.resize((size_t) N_MEL * std::max(0, nf));
    return r;
}

// -------- incremental driver --------
struct IncResult {
    MelResult raw;                 // assemble_raw output
    MelResult even;                // assemble_even output
    voxtral_mel_metrics metrics;
    int64_t max_retained_after_feed = 0;
};

IncResult run_incremental(const std::vector<float> & sig, const std::vector<size_t> & plan,
                          bool check_retained = false) {
    IncResult r;
    voxtral_mel_frontend * fe = voxtral_mel_frontend_create(g_hann.data(), g_filters.data());
    CHECK(fe != nullptr);

    size_t off = 0;
    for (size_t c : plan) {
        const float * ptr = (c == 0) ? nullptr : (sig.data() + off);
        CHECK(voxtral_mel_frontend_feed(fe, ptr, c));
        off += c;
        const voxtral_mel_metrics m = voxtral_mel_frontend_metrics(fe);
        if (m.pcm_retained > r.max_retained_after_feed) r.max_retained_after_feed = m.pcm_retained;
        if (check_retained) {
            // A bounded tail: after every feed the retained PCM is strictly less
            // than one full window, independent of how much has been fed.
            CHECK(m.pcm_retained < N_FFT);
        }
    }
    CHECK(off == sig.size());

    voxtral_mel_frontend_finish(fe);
    voxtral_mel_frontend_assemble_raw(fe, r.raw.mel, r.raw.frames);
    voxtral_mel_frontend_assemble_even(fe, r.even.mel, r.even.frames);
    r.metrics = voxtral_mel_frontend_metrics(fe);
    voxtral_mel_frontend_destroy(fe);
    return r;
}

double max_abs_delta(const std::vector<float> & a, const std::vector<float> & b, bool & finite) {
    finite = true;
    double md = 0.0;
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) finite = false;
        const double d = std::fabs((double) a[i] - (double) b[i]);
        if (d > md) md = d;
    }
    return md;
}

// Even-trim a batch [n_mel, n_frames] like compute_mel_even (drop first if odd).
MelResult even_trim(const MelResult & in) {
    MelResult out;
    const int32_t nf = in.frames;
    const bool drop_first = (nf % 2 != 0);
    const int32_t n_out = drop_first ? (nf - 1) : nf;
    out.frames = n_out;
    if (n_out <= 0) return out;
    out.mel.resize((size_t) N_MEL * n_out);
    const int32_t src_off = drop_first ? 1 : 0;
    for (int32_t m = 0; m < N_MEL; ++m)
        for (int32_t j = 0; j < n_out; ++j)
            out.mel[(size_t) m * n_out + j] = in.mel[(size_t) m * nf + (j + src_off)];
    return out;
}

// Core assertion: incremental over `plan` == batch over `sig`, with each frame
// computed exactly once and no recomputation.
double g_worst_delta = 0.0;

void assert_parity(const char * label, const std::vector<float> & sig,
                   const std::vector<size_t> & plan, bool check_retained = false) {
    const MelResult batch = run_batch(sig);
    const IncResult inc   = run_incremental(sig, plan, check_retained);

    // Same frame count and layout.
    CHECK(inc.raw.frames == batch.frames);

    bool finite = true;
    const double d = max_abs_delta(batch.mel, inc.raw.mel, finite);
    if (d > g_worst_delta) g_worst_delta = d;
    CHECK(finite);
    // Bit-for-bit: the incremental path uses the exact same kernel and input
    // samples as the batch path.
    if (!(d <= 1e-6)) {
        std::fprintf(stderr, "  [%s] max abs delta = %.3e (frames=%d, plan feeds=%zu)\n",
                     label, d, batch.frames, plan.size());
    }
    CHECK(d <= 1e-6);

    // Every frame computed exactly once; no recomputation of earlier frames.
    CHECK(inc.metrics.dft_frames_computed == inc.raw.frames);
    CHECK(inc.metrics.frames_total == inc.raw.frames);
    CHECK(inc.metrics.frames_during_feed + inc.metrics.frames_at_finish == inc.metrics.frames_total);
    CHECK(inc.metrics.finalized);

    // Even-trim equivalence with compute_mel_even.
    const MelResult batch_even = even_trim(batch);
    CHECK(inc.even.frames == batch_even.frames);
    bool efinite = true;
    const double ed = max_abs_delta(batch_even.mel, inc.even.mel, efinite);
    CHECK(efinite);
    CHECK(ed <= 1e-6);
}

// ---------------------------------------------------------------------------

void test_boundary_lengths() {
    // Lengths straddling the frame/hop/window boundaries.
    const int32_t lens[] = {0, 1, 159, 160, 199, 200, 201, 239, 240, 319, 320,
                            359, 360, 399, 400, 479, 480, 640, 800, 1280};
    for (int32_t n : lens) {
        const std::vector<float> sig = sig_tone(n, 220.0f, 0xC0FFEEu ^ (uint32_t) n);
        const std::vector<std::vector<size_t>> plans = {
            plan_full((size_t) n),
            plan_single((size_t) n),
            plan_fixed((size_t) n, 1280),
            plan_fixed((size_t) n, 2560),
            plan_fixed((size_t) n, 7680),
            plan_irregular((size_t) n),
            plan_seeded((size_t) n, 99u + (uint32_t) n),
            with_zero_feeds(plan_fixed((size_t) n, 1280)),
        };
        for (const auto & p : plans) {
            char lbl[64];
            std::snprintf(lbl, sizeof(lbl), "len=%d feeds=%zu", n, p.size());
            assert_parity(lbl, sig, p);
        }
    }
}

void test_signal_shapes() {
    const int32_t n = 5000;   // a bit over 31 frames
    struct { const char * name; std::vector<float> sig; } cases[] = {
        {"zeros",       sig_zeros(n)},
        {"constant",    sig_constant(n, 0.3f)},
        {"impulse",     sig_impulse(n)},
        {"alternating", sig_alternating(n)},
        {"noise",       sig_noise(n, 0x1234u)},
        {"tone",        sig_tone(n, 440.0f, 0x9u)},
    };
    for (auto & c : cases) {
        const std::vector<std::vector<size_t>> plans = {
            plan_full((size_t) n),
            plan_fixed((size_t) n, 1280),
            plan_fixed((size_t) n, 2560),
            plan_fixed((size_t) n, 7680),
            plan_irregular((size_t) n),
            plan_seeded((size_t) n, 7u),
            with_zero_feeds(plan_fixed((size_t) n, 2560)),
        };
        for (const auto & p : plans) assert_parity(c.name, c.sig, p);
    }
}

void test_random_lengths() {
    Rng rng(0xABCDu);
    for (int i = 0; i < 12; ++i) {
        const int32_t n = 1 + (int32_t) (rng.next() * 40000.0);
        const std::vector<float> sig = sig_tone(n, 300.0f + 50.0f * i, (uint32_t) (1000 + i));
        assert_parity("random-len-seeded", sig, plan_seeded((size_t) n, (uint32_t) (55 + i)));
        assert_parity("random-len-irregular", sig, plan_irregular((size_t) n));
    }
}

void test_several_seconds() {
    // ~3 seconds; the parity + one-shot guarantees on a realistic length.
    const int32_t n = 3 * VOXTRAL_SAMPLE_RATE;
    const std::vector<float> sig = sig_tone(n, 523.25f, 0x5EEDu);
    assert_parity("3s-full",  sig, plan_full((size_t) n));
    assert_parity("3s-80ms",  sig, plan_fixed((size_t) n, 1280));
    assert_parity("3s-480ms", sig, plan_fixed((size_t) n, 7680));
    assert_parity("3s-rand",  sig, plan_seeded((size_t) n, 0x1u));
}

void test_zero_length_feeds_noop() {
    const int32_t n = 4000;
    const std::vector<float> sig = sig_tone(n, 200.0f, 3u);
    // A plan that is ALL zero feeds except the data must equal a plan with none.
    const MelResult a = run_incremental(sig, plan_fixed((size_t) n, 1280)).raw;
    const MelResult b = run_incremental(sig, with_zero_feeds(plan_fixed((size_t) n, 1280))).raw;
    CHECK(a.frames == b.frames);
    bool finite = true;
    CHECK(max_abs_delta(a.mel, b.mel, finite) == 0.0);
    CHECK(finite);
}

void test_repeated_finish_and_reset() {
    const int32_t n = 6000;
    const std::vector<float> sig = sig_tone(n, 330.0f, 11u);

    voxtral_mel_frontend * fe = voxtral_mel_frontend_create(g_hann.data(), g_filters.data());
    CHECK(fe != nullptr);
    CHECK(voxtral_mel_frontend_feed(fe, sig.data(), sig.size()));
    voxtral_mel_frontend_finish(fe);
    const voxtral_mel_metrics m1 = voxtral_mel_frontend_metrics(fe);

    // Repeated finish must recompute nothing and change no counters.
    voxtral_mel_frontend_finish(fe);
    const voxtral_mel_metrics m2 = voxtral_mel_frontend_metrics(fe);
    CHECK(m1.dft_frames_computed == m2.dft_frames_computed);
    CHECK(m1.frames_total == m2.frames_total);
    // Feeding after finish is rejected (no effect).
    CHECK(!voxtral_mel_frontend_feed(fe, sig.data(), 10));
    CHECK(!voxtral_mel_frontend_feed_silence(fe, 10));

    // Reset returns to a clean state; a second run matches a fresh frontend.
    voxtral_mel_frontend_reset(fe);
    const voxtral_mel_metrics m0 = voxtral_mel_frontend_metrics(fe);
    CHECK(m0.frames_total == 0);
    CHECK(m0.dft_frames_computed == 0);
    CHECK(m0.pcm_retained == 0);
    CHECK(!m0.finalized);

    CHECK(voxtral_mel_frontend_feed(fe, sig.data(), sig.size()));
    voxtral_mel_frontend_finish(fe);
    std::vector<float> mel2; int32_t f2 = 0;
    voxtral_mel_frontend_assemble_raw(fe, mel2, f2);
    voxtral_mel_frontend_destroy(fe);

    const MelResult batch = run_batch(sig);
    CHECK(f2 == batch.frames);
    bool finite = true;
    CHECK(max_abs_delta(mel2, batch.mel, finite) <= 1e-6);
    CHECK(finite);
}

void test_frames_mostly_during_feed() {
    // For multi-second audio, most frames are emitted during feed, not at finish.
    const int32_t n = 2 * VOXTRAL_SAMPLE_RATE;
    const std::vector<float> sig = sig_tone(n, 261.6f, 42u);
    const IncResult inc = run_incremental(sig, plan_fixed((size_t) n, 1280));
    CHECK(inc.metrics.frames_during_feed > inc.metrics.frames_at_finish);
    CHECK(inc.metrics.frames_at_finish >= 0);
    CHECK(inc.metrics.frames_at_finish <= 3);   // only the trailing partial window
}

void test_memory_bounded() {
    // Feed a long-ish signal in fixed streaming chunks; retained PCM must stay
    // below one window after every feed, independent of the growing total, and
    // the DFT count must equal the frame count (no recomputation).
    const int32_t n = 20 * VOXTRAL_SAMPLE_RATE;   // 20 s
    const std::vector<float> sig = sig_tone(n, 180.0f, 0xBEEFu);
    const IncResult inc = run_incremental(sig, plan_fixed((size_t) n, 1280), /*check_retained=*/true);
    const MelResult batch = run_batch(sig);
    CHECK(inc.raw.frames == batch.frames);
    CHECK(inc.metrics.dft_frames_computed == inc.raw.frames);
    CHECK(inc.max_retained_after_feed < N_FFT);
    CHECK(inc.metrics.pcm_peak_retained < N_FFT);
    // Frames scale linearly with duration (~ n/hop).
    CHECK(inc.raw.frames == n / HOP);
}

void test_bounded_mel_consumption_and_dependency() {
    // Exact STFT dependency seam used by encoder residence telemetry.
    CHECK(voxtral_mel_frame_required_sample(-1) == 0);
    CHECK(voxtral_mel_frame_required_sample(0) == 200);  // reflected left edge
    CHECK(voxtral_mel_frame_required_sample(1) == 359);
    CHECK(voxtral_mel_frame_required_sample(2) == 519);

    const std::vector<float> sig = sig_tone(10 * VOXTRAL_SAMPLE_RATE, 240.0f, 77u);
    voxtral_mel_frontend * fe = voxtral_mel_frontend_create(g_hann.data(), g_filters.data());
    CHECK(fe != nullptr);
    voxtral_mel_frontend_set_retain_history(fe, false);
    size_t off = 0;
    while (off < sig.size()) {
        const size_t take = std::min<size_t>(1280, sig.size() - off);
        CHECK(voxtral_mel_frontend_feed(fe, sig.data() + off, take));
        off += take;
        const int64_t produced = voxtral_mel_frontend_frame_count(fe);
        voxtral_mel_frontend_discard_before(fe, produced);
        CHECK(voxtral_mel_frontend_frames_base(fe) == produced);
        CHECK(voxtral_mel_frontend_frames_data(fe) == nullptr);
    }
    voxtral_mel_frontend_finish(fe);
    const int64_t final_frames = voxtral_mel_frontend_frame_count(fe);
    voxtral_mel_frontend_discard_before(fe, final_frames);
    CHECK(voxtral_mel_frontend_frames_base(fe) == final_frames);
    CHECK(voxtral_mel_frontend_frames_data(fe) == nullptr);
    const voxtral_mel_metrics metrics = voxtral_mel_frontend_metrics(fe);
    CHECK(metrics.frames_total == final_frames);
    CHECK(metrics.dft_frames_computed == final_frames);
    voxtral_mel_frontend_destroy(fe);
}

void test_bounded_mel_absolute_slice_cursor() {
    // Regression for the streaming drain contract: after discard_before(),
    // absolute frame indices must be resolved against the frontend's NEW base.
    // A stale caller-side base used to return later Mel rows for one-shot feeds
    // while retained-history parity mode accidentally hid the bug.
    const std::vector<float> sig = sig_tone(3 * VOXTRAL_SAMPLE_RATE, 310.0f, 0x6202u);
    const MelResult batch = run_batch(sig);
    voxtral_mel_frontend * fe = voxtral_mel_frontend_create(g_hann.data(), g_filters.data());
    CHECK(fe != nullptr);
    voxtral_mel_frontend_set_retain_history(fe, false);

    std::vector<float> drained;  // frame-major
    int64_t cursor = 0;
    auto drain = [&]() {
        const int64_t total = voxtral_mel_frontend_frame_count(fe);
        while (cursor < total) {
            const int32_t n = (int32_t) std::min<int64_t>(8, total - cursor);
            const float * frames = voxtral_mel_frontend_frame_data(fe, cursor);
            CHECK(frames != nullptr);
            if (!frames) return;
            drained.insert(drained.end(), frames, frames + (size_t) n * N_MEL);
            cursor += n;
            voxtral_mel_frontend_discard_before(fe, cursor);
            CHECK(voxtral_mel_frontend_frames_base(fe) == cursor);
            CHECK(voxtral_mel_frontend_frame_data(fe, cursor - 1) == nullptr);
        }
    };

    size_t off = 0;
    while (off < sig.size()) {
        const size_t take = std::min<size_t>(1280, sig.size() - off);
        CHECK(voxtral_mel_frontend_feed(fe, sig.data() + off, take));
        off += take;
        drain();
    }
    voxtral_mel_frontend_finish(fe);
    drain();
    CHECK(cursor == batch.frames);
    CHECK(drained.size() == (size_t) batch.frames * N_MEL);
    bool exact = drained.size() == (size_t) batch.frames * N_MEL;
    for (int32_t f = 0; exact && f < batch.frames; ++f) {
        for (int32_t m = 0; m < N_MEL; ++m) {
            if (drained[(size_t) f * N_MEL + m] != batch.mel[(size_t) m * batch.frames + f]) {
                exact = false;
                break;
            }
        }
    }
    CHECK(exact);
    voxtral_mel_frontend_destroy(fe);
}

void test_soak_optional() {
    if (!std::getenv("VOXTRAL_MEL_SOAK")) {
        std::printf("[SKIP] soak (set VOXTRAL_MEL_SOAK=1 to enable)\n");
        return;
    }
    const int32_t n = 3 * 60 * VOXTRAL_SAMPLE_RATE;   // 3 minutes
    const std::vector<float> sig = sig_tone(n, 200.0f, 0xF00Du);
    const IncResult inc = run_incremental(sig, plan_fixed((size_t) n, 1280), /*check_retained=*/true);
    CHECK(inc.metrics.dft_frames_computed == inc.raw.frames);
    CHECK(inc.raw.frames == n / HOP);
    CHECK(inc.metrics.pcm_peak_retained < N_FFT);
}

struct named_test { const char * name; void (*fn)(); };

} // namespace

int main() {
    init_tables();

    const named_test tests[] = {
        {"boundary_lengths",         test_boundary_lengths},
        {"signal_shapes",            test_signal_shapes},
        {"random_lengths",           test_random_lengths},
        {"several_seconds",          test_several_seconds},
        {"zero_length_feeds_noop",   test_zero_length_feeds_noop},
        {"repeated_finish_reset",    test_repeated_finish_and_reset},
        {"frames_mostly_during_feed",test_frames_mostly_during_feed},
        {"memory_bounded",           test_memory_bounded},
        {"bounded_mel_consumption",   test_bounded_mel_consumption_and_dependency},
        {"bounded_mel_slice_cursor",  test_bounded_mel_absolute_slice_cursor},
        {"soak_optional",            test_soak_optional},
    };

    for (const auto & t : tests) {
        const int before = g_failures;
        t.fn();
        std::printf("[%s] %s\n", g_failures == before ? "PASS" : "FAIL", t.name);
    }

    std::printf("\n%d checks, %d failure(s); worst max-abs-delta vs batch = %.3e\n",
                g_checks, g_failures, g_worst_delta);
    return g_failures == 0 ? 0 : 1;
}

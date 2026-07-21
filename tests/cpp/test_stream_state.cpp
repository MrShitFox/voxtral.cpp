// ============================================================================
// C++ unit tests for the internal streaming session skeleton.
//
// These tests exercise lifecycle, PCM accounting, conversion, events, reset,
// cancel and guard rails WITHOUT a model or backend (ctx == nullptr). Anything
// that requires real inference lives in the model-driven voxtral-stream-test
// executable and the Node.js acceptance suite.
//
// Assert-based; no external test framework. Exits non-zero on any failure.
// ============================================================================

#include "voxtral-stream.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
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

voxtral_stream_params default_params() {
    voxtral_stream_params p;   // 16 kHz mono by default
    return p;
}

// --- 1. create / destroy ---------------------------------------------------
void test_create_destroy() {
    voxtral_stream * s = voxtral_stream_create_internal(nullptr, default_params());
    CHECK(s != nullptr);
    CHECK(voxtral_stream_get_state(s) == voxtral_stream_state::created);
    CHECK(voxtral_stream_samples_received(s) == 0);
    CHECK(voxtral_stream_last_status(s) == voxtral_status::ok);
    CHECK(voxtral_stream_pending_events(s) == 0);
    voxtral_stream_destroy_internal(s);
    // destroy(nullptr) must be safe.
    voxtral_stream_destroy_internal(nullptr);
}

// --- 2 & 3. invalid sample rate / channels ---------------------------------
void test_invalid_format() {
    voxtral_stream_params bad_rate = default_params();
    bad_rate.sample_rate = 44100;
    CHECK(voxtral_stream_params_check(bad_rate) == voxtral_status::unsupported_audio_format);
    CHECK(voxtral_stream_create_internal(nullptr, bad_rate) == nullptr);

    voxtral_stream_params bad_ch = default_params();
    bad_ch.channels = 2;
    CHECK(voxtral_stream_params_check(bad_ch) == voxtral_status::unsupported_audio_format);
    CHECK(voxtral_stream_create_internal(nullptr, bad_ch) == nullptr);

    CHECK(voxtral_stream_params_check(default_params()) == voxtral_status::ok);
}

// --- 4. zero-length feed ---------------------------------------------------
void test_zero_length_feed() {
    voxtral_stream * s = voxtral_stream_create_internal(nullptr, default_params());
    // nullptr with zero count is valid.
    CHECK(voxtral_stream_feed_pcm16_internal(s, nullptr, 0) == voxtral_status::ok);
    CHECK(voxtral_stream_feed_f32_internal(s, nullptr, 0) == voxtral_status::ok);
    // No audio position change, and no implicit transition out of created.
    CHECK(voxtral_stream_samples_received(s) == 0);
    CHECK(voxtral_stream_get_state(s) == voxtral_stream_state::created);
    CHECK(voxtral_stream_feed_calls(s) == 2);
    voxtral_stream_destroy_internal(s);
}

// --- 5. null pointer with non-zero count -----------------------------------
void test_null_with_count() {
    voxtral_stream * s = voxtral_stream_create_internal(nullptr, default_params());
    CHECK(voxtral_stream_feed_pcm16_internal(s, nullptr, 4) == voxtral_status::invalid_argument);
    CHECK(voxtral_stream_feed_f32_internal(s, nullptr, 4) == voxtral_status::invalid_argument);
    CHECK(voxtral_stream_samples_received(s) == 0);
    voxtral_stream_destroy_internal(s);
}

// --- 6. single sample feed -------------------------------------------------
void test_single_sample() {
    voxtral_stream * s = voxtral_stream_create_internal(nullptr, default_params());
    const int16_t one = 16384;
    CHECK(voxtral_stream_feed_pcm16_internal(s, &one, 1) == voxtral_status::ok);
    CHECK(voxtral_stream_samples_received(s) == 1);
    CHECK(voxtral_stream_get_state(s) == voxtral_stream_state::running);
    CHECK(voxtral_stream_pcm_size(s) == 1);
    voxtral_stream_destroy_internal(s);
}

// --- 7 & 8. multiple chunks / sample accounting ----------------------------
void test_multiple_chunks_accounting() {
    voxtral_stream * s = voxtral_stream_create_internal(nullptr, default_params());
    std::vector<int16_t> a(100, 1), b(1, 2), c(2560, 3);
    CHECK(voxtral_stream_feed_pcm16_internal(s, a.data(), a.size()) == voxtral_status::ok);
    CHECK(voxtral_stream_feed_pcm16_internal(s, b.data(), b.size()) == voxtral_status::ok);
    CHECK(voxtral_stream_feed_pcm16_internal(s, c.data(), c.size()) == voxtral_status::ok);
    CHECK(voxtral_stream_samples_received(s) == 100 + 1 + 2560);
    CHECK(voxtral_stream_pcm_size(s) == 100 + 1 + 2560);
    // audio_ms derived from the 64-bit count, not accumulated.
    const double expect_ms = static_cast<double>(2661) * 1000.0 / 16000.0;
    CHECK(std::fabs(voxtral_stream_audio_ms(s) - expect_ms) < 1e-9);
    voxtral_stream_destroy_internal(s);
}

// --- 9. PCM16 conversion ---------------------------------------------------
void test_pcm16_conversion() {
    voxtral_stream * s = voxtral_stream_create_internal(nullptr, default_params());
    const int16_t in[] = {0, 32767, -32768, 16384, -16384};
    CHECK(voxtral_stream_feed_pcm16_internal(s, in, 5) == voxtral_status::ok);
    const float * pcm = voxtral_stream_pcm_data(s);
    CHECK(pcm != nullptr);
    CHECK(pcm[0] == 0.0f);
    CHECK(std::fabs(pcm[1] - 32767.0f / 32768.0f) < 1e-9f);
    CHECK(pcm[2] == -1.0f);                 // -32768 maps to exactly -1.0f
    CHECK(std::fabs(pcm[3] - 0.5f) < 1e-9f);
    CHECK(std::fabs(pcm[4] + 0.5f) < 1e-9f);
    voxtral_stream_destroy_internal(s);
}

// --- float32 feed: finite validation, no clamp -----------------------------
void test_f32_feed() {
    voxtral_stream * s = voxtral_stream_create_internal(nullptr, default_params());
    const float ok_vals[] = {-1.0f, 0.0f, 0.25f, 1.5f};   // 1.5 passes through (no clamp)
    CHECK(voxtral_stream_feed_f32_internal(s, ok_vals, 4) == voxtral_status::ok);
    CHECK(voxtral_stream_pcm_size(s) == 4);
    CHECK(voxtral_stream_pcm_data(s)[3] == 1.5f);

    const float nan_vals[] = {0.0f, std::nanf("")};
    CHECK(voxtral_stream_feed_f32_internal(s, nan_vals, 2) == voxtral_status::invalid_argument);
    // Rejected payload must not mutate the buffer.
    CHECK(voxtral_stream_pcm_size(s) == 4);
    CHECK(voxtral_stream_samples_received(s) == 4);
    voxtral_stream_destroy_internal(s);
}

// --- empty finish + 10/11: invalid state after finish, repeated finish -----
void test_empty_finish_and_repeat() {
    voxtral_stream * s = voxtral_stream_create_internal(nullptr, default_params());
    CHECK(voxtral_stream_finish_internal(s) == voxtral_status::ok);
    CHECK(voxtral_stream_get_state(s) == voxtral_stream_state::completed);
    CHECK(voxtral_stream_inference_runs(s) == 0);        // empty path never runs inference
    CHECK(voxtral_stream_transcript(s).empty());

    // feed after finish -> invalid_state, no crash.
    const int16_t x = 1;
    CHECK(voxtral_stream_feed_pcm16_internal(s, &x, 1) == voxtral_status::invalid_state);

    // repeated finish is idempotent and does not run inference.
    CHECK(voxtral_stream_finish_internal(s) == voxtral_status::ok);
    CHECK(voxtral_stream_inference_runs(s) == 0);
    voxtral_stream_destroy_internal(s);
}

// --- null-context non-empty finish path (documented failure) ---------------
void test_finish_without_context() {
    voxtral_stream * s = voxtral_stream_create_internal(nullptr, default_params());
    const int16_t buf[4] = {1, 2, 3, 4};
    CHECK(voxtral_stream_feed_pcm16_internal(s, buf, 4) == voxtral_status::ok);
    CHECK(voxtral_stream_finish_internal(s) == voxtral_status::backend_error);
    CHECK(voxtral_stream_get_state(s) == voxtral_stream_state::failed);
    CHECK(voxtral_stream_last_status(s) == voxtral_status::backend_error);
    CHECK(!voxtral_stream_last_error(s).empty());
    // finish after failure is rejected; feed after failure is rejected.
    CHECK(voxtral_stream_finish_internal(s) == voxtral_status::invalid_state);
    CHECK(voxtral_stream_feed_pcm16_internal(s, buf, 4) == voxtral_status::invalid_state);
    voxtral_stream_destroy_internal(s);
}

// --- 12 & 13. cancel / repeated cancel -------------------------------------
void test_cancel() {
    voxtral_stream * s = voxtral_stream_create_internal(nullptr, default_params());
    std::vector<int16_t> a(1280, 5);
    CHECK(voxtral_stream_feed_pcm16_internal(s, a.data(), a.size()) == voxtral_status::ok);
    CHECK(voxtral_stream_cancel_internal(s) == voxtral_status::ok);
    CHECK(voxtral_stream_get_state(s) == voxtral_stream_state::cancelled);

    // finish after cancel must not run inference.
    CHECK(voxtral_stream_finish_internal(s) == voxtral_status::ok);
    CHECK(voxtral_stream_inference_runs(s) == 0);
    CHECK(voxtral_stream_get_state(s) == voxtral_stream_state::cancelled);

    // repeated cancel is idempotent.
    CHECK(voxtral_stream_cancel_internal(s) == voxtral_status::ok);

    // exactly one CANCELLED event was emitted.
    int cancelled_events = 0;
    voxtral_stream_event ev;
    while (voxtral_stream_poll_event(s, ev)) {
        if (ev.type == voxtral_stream_event_type::cancelled) ++cancelled_events;
    }
    CHECK(cancelled_events == 1);

    // feed after cancel is rejected.
    CHECK(voxtral_stream_feed_pcm16_internal(s, a.data(), a.size()) == voxtral_status::invalid_state);
    voxtral_stream_destroy_internal(s);
}

// --- cancel after completion is rejected -----------------------------------
void test_cancel_after_completed() {
    voxtral_stream * s = voxtral_stream_create_internal(nullptr, default_params());
    CHECK(voxtral_stream_finish_internal(s) == voxtral_status::ok);
    CHECK(voxtral_stream_cancel_internal(s) == voxtral_status::invalid_state);
    voxtral_stream_destroy_internal(s);
}

// --- 14. reset after feed --------------------------------------------------
void test_reset_after_feed() {
    voxtral_stream * s = voxtral_stream_create_internal(nullptr, default_params());
    std::vector<int16_t> a(500, 7);
    CHECK(voxtral_stream_feed_pcm16_internal(s, a.data(), a.size()) == voxtral_status::ok);
    CHECK(voxtral_stream_reset_internal(s) == voxtral_status::ok);
    CHECK(voxtral_stream_get_state(s) == voxtral_stream_state::created);
    CHECK(voxtral_stream_samples_received(s) == 0);
    CHECK(voxtral_stream_pcm_size(s) == 0);
    CHECK(voxtral_stream_feed_calls(s) == 0);
    // Reusable: a fresh feed works and can finish (empty second run here).
    const int16_t x = 1;
    CHECK(voxtral_stream_feed_pcm16_internal(s, &x, 1) == voxtral_status::ok);
    CHECK(voxtral_stream_get_state(s) == voxtral_stream_state::running);
    voxtral_stream_destroy_internal(s);
}

// --- 15. reset after cancel ------------------------------------------------
void test_reset_after_cancel() {
    voxtral_stream * s = voxtral_stream_create_internal(nullptr, default_params());
    std::vector<int16_t> a(500, 7);
    CHECK(voxtral_stream_feed_pcm16_internal(s, a.data(), a.size()) == voxtral_status::ok);
    CHECK(voxtral_stream_cancel_internal(s) == voxtral_status::ok);
    CHECK(voxtral_stream_reset_internal(s) == voxtral_status::ok);
    CHECK(voxtral_stream_get_state(s) == voxtral_stream_state::created);
    CHECK(voxtral_stream_pending_events(s) == 0);
    // A fresh empty finish works after reset.
    CHECK(voxtral_stream_finish_internal(s) == voxtral_status::ok);
    CHECK(voxtral_stream_get_state(s) == voxtral_stream_state::completed);
    voxtral_stream_destroy_internal(s);
}

// --- 16. event ordering ----------------------------------------------------
void test_event_ordering() {
    voxtral_stream * s = voxtral_stream_create_internal(nullptr, default_params());
    CHECK(voxtral_stream_finish_internal(s) == voxtral_status::ok);
    CHECK(voxtral_stream_pending_events(s) == 2);

    voxtral_stream_event ev;
    CHECK(voxtral_stream_poll_event(s, ev));
    CHECK(ev.type == voxtral_stream_event_type::final_text);
    CHECK(ev.text.empty());
    CHECK(voxtral_stream_poll_event(s, ev));
    CHECK(ev.type == voxtral_stream_event_type::completed);
    // Draining is safe and repeatable.
    CHECK(!voxtral_stream_poll_event(s, ev));
    CHECK(!voxtral_stream_poll_event(s, ev));
    voxtral_stream_destroy_internal(s);
}

// --- 17. error clearing after reset ----------------------------------------
void test_error_cleared_by_reset() {
    voxtral_stream * s = voxtral_stream_create_internal(nullptr, default_params());
    CHECK(voxtral_stream_feed_pcm16_internal(s, nullptr, 4) == voxtral_status::invalid_argument);
    CHECK(voxtral_stream_last_status(s) == voxtral_status::invalid_argument);
    CHECK(!voxtral_stream_last_error(s).empty());
    CHECK(voxtral_stream_reset_internal(s) == voxtral_status::ok);
    CHECK(voxtral_stream_last_status(s) == voxtral_status::ok);
    CHECK(voxtral_stream_last_error(s).empty());
    voxtral_stream_destroy_internal(s);
}

// --- 18. size / overflow guards --------------------------------------------
void test_overflow_guards() {
    // Per-call limit: a huge count is rejected before any read of the buffer.
    voxtral_stream * s = voxtral_stream_create_internal(nullptr, default_params());
    int16_t tiny = 0;
    CHECK(voxtral_stream_feed_pcm16_internal(s, &tiny, (size_t)64 * 1024 * 1024 + 1)
          == voxtral_status::invalid_argument);
    CHECK(voxtral_stream_samples_received(s) == 0);
    voxtral_stream_destroy_internal(s);

    // Cumulative bound: max_total_samples enforced across feeds, overflow-safe.
    voxtral_stream_params bounded = default_params();
    bounded.max_total_samples = 10;
    voxtral_stream * b = voxtral_stream_create_internal(nullptr, bounded);
    std::vector<int16_t> eight(8, 1), five(5, 1), eleven(11, 1);
    CHECK(voxtral_stream_feed_pcm16_internal(b, eleven.data(), eleven.size())
          == voxtral_status::out_of_memory);              // single feed exceeds bound
    CHECK(voxtral_stream_feed_pcm16_internal(b, eight.data(), eight.size()) == voxtral_status::ok);
    CHECK(voxtral_stream_feed_pcm16_internal(b, five.data(), five.size())
          == voxtral_status::out_of_memory);              // cumulative 8+5 > 10
    CHECK(voxtral_stream_samples_received(b) == 8);       // unchanged by the rejected feed
    voxtral_stream_destroy_internal(b);
}

struct named_test { const char * name; void (*fn)(); };

} // namespace

int main() {
    const named_test tests[] = {
        {"create_destroy",            test_create_destroy},
        {"invalid_format",            test_invalid_format},
        {"zero_length_feed",          test_zero_length_feed},
        {"null_with_count",           test_null_with_count},
        {"single_sample",             test_single_sample},
        {"multiple_chunks_accounting",test_multiple_chunks_accounting},
        {"pcm16_conversion",          test_pcm16_conversion},
        {"f32_feed",                  test_f32_feed},
        {"empty_finish_and_repeat",   test_empty_finish_and_repeat},
        {"finish_without_context",    test_finish_without_context},
        {"cancel",                    test_cancel},
        {"cancel_after_completed",    test_cancel_after_completed},
        {"reset_after_feed",          test_reset_after_feed},
        {"reset_after_cancel",        test_reset_after_cancel},
        {"event_ordering",            test_event_ordering},
        {"error_cleared_by_reset",    test_error_cleared_by_reset},
        {"overflow_guards",           test_overflow_guards},
    };

    for (const auto & t : tests) {
        const int before = g_failures;
        t.fn();
        std::printf("[%s] %s\n", g_failures == before ? "PASS" : "FAIL", t.name);
    }

    std::printf("\n%d checks, %d failure(s)\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}

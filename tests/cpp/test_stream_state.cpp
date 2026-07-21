// ============================================================================
// C++ unit tests for the internal streaming session skeleton.
//
// These tests exercise lifecycle, ownership, PCM accounting, conversion, events,
// reset, cancel, the reentrancy guard and guard rails WITHOUT a real model or
// backend. The success inference path (owned context runs the model) is covered
// by the model-driven voxtral-stream-test executable and the Node.js acceptance
// suite. Context ownership/lifetime without a backend is exercised here via the
// documented internal test seam (fake context factory / free).
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

// Create a lifecycle-only stream (no model -> no owned context). This is the
// mode the model-free tests use for everything that does not need inference.
voxtral_stream * make_stream(const voxtral_stream_params & p = default_params()) {
    return voxtral_stream_create_internal(nullptr, voxtral_context_params{}, p);
}

// --- 1. create / destroy ---------------------------------------------------
void test_create_destroy() {
    voxtral_stream * s = make_stream();
    CHECK(s != nullptr);
    CHECK(voxtral_stream_get_state(s) == voxtral_stream_state::created);
    CHECK(voxtral_stream_samples_received(s) == 0);
    CHECK(voxtral_stream_last_status(s) == voxtral_status::ok);
    CHECK(voxtral_stream_pending_events(s) == 0);
    // A model-less stream owns no context.
    CHECK(!voxtral_stream_owns_context(s));
    CHECK(voxtral_stream_context_ptr(s) == nullptr);
    voxtral_stream_destroy_internal(s);
    // destroy(nullptr) must be safe.
    voxtral_stream_destroy_internal(nullptr);
}

// --- 2 & 3. invalid sample rate / channels ---------------------------------
void test_invalid_format() {
    voxtral_stream_params bad_rate = default_params();
    bad_rate.sample_rate = 44100;
    CHECK(voxtral_stream_params_check(bad_rate) == voxtral_status::unsupported_audio_format);
    CHECK(make_stream(bad_rate) == nullptr);

    voxtral_stream_params bad_ch = default_params();
    bad_ch.channels = 2;
    CHECK(voxtral_stream_params_check(bad_ch) == voxtral_status::unsupported_audio_format);
    CHECK(make_stream(bad_ch) == nullptr);

    CHECK(voxtral_stream_params_check(default_params()) == voxtral_status::ok);
}

// --- 4. zero-length feed ---------------------------------------------------
void test_zero_length_feed() {
    voxtral_stream * s = make_stream();
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
    voxtral_stream * s = make_stream();
    CHECK(voxtral_stream_feed_pcm16_internal(s, nullptr, 4) == voxtral_status::invalid_argument);
    CHECK(voxtral_stream_feed_f32_internal(s, nullptr, 4) == voxtral_status::invalid_argument);
    CHECK(voxtral_stream_samples_received(s) == 0);
    voxtral_stream_destroy_internal(s);
}

// --- 6. single sample feed -------------------------------------------------
void test_single_sample() {
    voxtral_stream * s = make_stream();
    const int16_t one = 16384;
    CHECK(voxtral_stream_feed_pcm16_internal(s, &one, 1) == voxtral_status::ok);
    CHECK(voxtral_stream_samples_received(s) == 1);
    CHECK(voxtral_stream_get_state(s) == voxtral_stream_state::running);
    CHECK(voxtral_stream_pcm_size(s) == 1);
    voxtral_stream_destroy_internal(s);
}

// --- 7 & 8. multiple chunks / sample accounting ----------------------------
void test_multiple_chunks_accounting() {
    voxtral_stream * s = make_stream();
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
    voxtral_stream * s = make_stream();
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
    voxtral_stream * s = make_stream();
    const float ok_vals[] = {-1.0f, 0.0f, 0.25f, 1.5f};   // 1.5 passes through (no clamp)
    CHECK(voxtral_stream_feed_f32_internal(s, ok_vals, 4) == voxtral_status::ok);
    CHECK(voxtral_stream_pcm_size(s) == 4);
    CHECK(voxtral_stream_pcm_data(s)[3] == 1.5f);

    const float nan_vals[] = {0.0f, std::nanf("")};
    CHECK(voxtral_stream_feed_f32_internal(s, nan_vals, 2) == voxtral_status::invalid_argument);
    // Rejected payload must not mutate the buffer.
    CHECK(voxtral_stream_pcm_size(s) == 4);
    CHECK(voxtral_stream_samples_received(s) == 4);

    // ±Inf is rejected the same way.
    const float inf_vals[] = {INFINITY};
    CHECK(voxtral_stream_feed_f32_internal(s, inf_vals, 1) == voxtral_status::invalid_argument);
    CHECK(voxtral_stream_pcm_size(s) == 4);
    voxtral_stream_destroy_internal(s);
}

// --- empty finish + 10/11: invalid state after finish, repeated finish -----
void test_empty_finish_and_repeat() {
    voxtral_stream * s = make_stream();
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
    voxtral_stream * s = make_stream();
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
    voxtral_stream * s = make_stream();
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
    voxtral_stream * s = make_stream();
    CHECK(voxtral_stream_finish_internal(s) == voxtral_status::ok);
    CHECK(voxtral_stream_cancel_internal(s) == voxtral_status::invalid_state);
    voxtral_stream_destroy_internal(s);
}

// --- 14. reset after feed (cheap reuse; no shrink_to_fit) -------------------
void test_reset_after_feed() {
    voxtral_stream * s = make_stream();
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

// --- functional reuse across many reset cycles -----------------------------
void test_reset_reuse_cycles() {
    voxtral_stream * s = make_stream();
    std::vector<int16_t> big(20000, 3);
    for (int cycle = 0; cycle < 4; ++cycle) {
        CHECK(voxtral_stream_feed_pcm16_internal(s, big.data(), big.size()) == voxtral_status::ok);
        CHECK(voxtral_stream_pcm_size(s) == big.size());
        CHECK(voxtral_stream_get_state(s) == voxtral_stream_state::running);
        CHECK(voxtral_stream_reset_internal(s) == voxtral_status::ok);
        CHECK(voxtral_stream_pcm_size(s) == 0);
        CHECK(voxtral_stream_get_state(s) == voxtral_stream_state::created);
    }
    voxtral_stream_destroy_internal(s);
}

// --- 15. reset after cancel ------------------------------------------------
void test_reset_after_cancel() {
    voxtral_stream * s = make_stream();
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
    voxtral_stream * s = make_stream();
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
    voxtral_stream * s = make_stream();
    CHECK(voxtral_stream_feed_pcm16_internal(s, nullptr, 4) == voxtral_status::invalid_argument);
    CHECK(voxtral_stream_last_status(s) == voxtral_status::invalid_argument);
    CHECK(!voxtral_stream_last_error(s).empty());
    CHECK(voxtral_stream_reset_internal(s) == voxtral_status::ok);
    CHECK(voxtral_stream_last_status(s) == voxtral_status::ok);
    CHECK(voxtral_stream_last_error(s).empty());
    voxtral_stream_destroy_internal(s);
}

// --- 18. size / overflow guards (per-call vs cumulative limit) --------------
void test_overflow_guards() {
    // Per-call sanity limit: a huge count is rejected before any read of the
    // buffer. This is an argument sanity ceiling, not a resource limit.
    voxtral_stream * s = make_stream();
    int16_t tiny = 0;
    CHECK(voxtral_stream_feed_pcm16_internal(s, &tiny, (size_t)64 * 1024 * 1024 + 1)
          == voxtral_status::invalid_argument);
    CHECK(voxtral_stream_samples_received(s) == 0);
    voxtral_stream_destroy_internal(s);

    // Cumulative compatibility bound: max_total_samples enforced across feeds,
    // overflow-safe. This is the documented full-buffer limit -> limit_exceeded
    // (NOT out_of_memory: it is not an allocation failure).
    voxtral_stream_params bounded = default_params();
    bounded.max_total_samples = 10;
    voxtral_stream * b = make_stream(bounded);
    std::vector<int16_t> eight(8, 1), five(5, 1), eleven(11, 1);
    CHECK(voxtral_stream_feed_pcm16_internal(b, eleven.data(), eleven.size())
          == voxtral_status::limit_exceeded);             // single feed exceeds bound
    CHECK(voxtral_stream_feed_pcm16_internal(b, eight.data(), eight.size()) == voxtral_status::ok);
    CHECK(voxtral_stream_feed_pcm16_internal(b, five.data(), five.size())
          == voxtral_status::limit_exceeded);             // cumulative 8+5 > 10
    CHECK(voxtral_stream_samples_received(b) == 8);        // unchanged by the rejected feed
    CHECK(voxtral_stream_last_status(b) == voxtral_status::limit_exceeded);
    voxtral_stream_destroy_internal(b);
}

// --- ownership: lifecycle-only stream owns no context ----------------------
void test_lifecycle_only_ownership() {
    voxtral_stream * s = make_stream();
    CHECK(s != nullptr);
    CHECK(!voxtral_stream_owns_context(s));
    CHECK(voxtral_stream_context_ptr(s) == nullptr);
    // A lifecycle-only stream still supports the full non-inference lifecycle.
    CHECK(voxtral_stream_finish_internal(s) == voxtral_status::ok);   // empty -> completed
    CHECK(voxtral_stream_get_state(s) == voxtral_stream_state::completed);
    voxtral_stream_destroy_internal(s);
}

// ---------------------------------------------------------------------------
// Ownership via the internal test seam: substitute the context factory / free
// so we can prove the stream creates, owns and frees its own context without a
// GGUF/backend, and that the model is never freed by the stream.
// ---------------------------------------------------------------------------
int g_fake_free_calls = 0;
voxtral_context * const g_fake_ctx = reinterpret_cast<voxtral_context *>(0x1234);

voxtral_context * fake_ctx_factory(voxtral_model *, const voxtral_context_params &) {
    return g_fake_ctx;
}
voxtral_context * failing_ctx_factory(voxtral_model *, const voxtral_context_params &) {
    return nullptr;
}
void fake_ctx_free(voxtral_context * c) {
    if (c == g_fake_ctx) ++g_fake_free_calls;
}

void test_owned_context_freed_on_destroy() {
    voxtral_stream_test_set_context_factory(fake_ctx_factory);
    voxtral_stream_test_set_context_free(fake_ctx_free);
    g_fake_free_calls = 0;

    // A non-null model pointer selects the owned-context path. The fake factory
    // never dereferences it, so an opaque sentinel is fine.
    voxtral_model * fake_model = reinterpret_cast<voxtral_model *>(0x1);
    voxtral_stream * s = voxtral_stream_create_internal(fake_model, voxtral_context_params{}, default_params());
    CHECK(s != nullptr);
    CHECK(voxtral_stream_get_state(s) == voxtral_stream_state::created);
    CHECK(voxtral_stream_owns_context(s));
    CHECK(voxtral_stream_context_ptr(s) == static_cast<const void *>(g_fake_ctx));

    voxtral_stream_destroy_internal(s);
    // Owned context freed exactly once; the model is never touched by destroy.
    CHECK(g_fake_free_calls == 1);

    voxtral_stream_test_set_context_factory(nullptr);   // restore defaults
    voxtral_stream_test_set_context_free(nullptr);
}

void test_context_creation_failure() {
    voxtral_stream_test_set_context_factory(failing_ctx_factory);
    voxtral_stream_test_set_context_free(fake_ctx_free);
    g_fake_free_calls = 0;

    voxtral_model * fake_model = reinterpret_cast<voxtral_model *>(0x1);
    voxtral_stream * s = voxtral_stream_create_internal(fake_model, voxtral_context_params{}, default_params());
    // Failure is surfaced as a queryable status/error, not a bare nullptr.
    CHECK(s != nullptr);
    CHECK(voxtral_stream_get_state(s) == voxtral_stream_state::failed);
    CHECK(voxtral_stream_last_status(s) == voxtral_status::backend_error);
    CHECK(!voxtral_stream_last_error(s).empty());
    CHECK(!voxtral_stream_owns_context(s));
    CHECK(voxtral_stream_context_ptr(s) == nullptr);

    voxtral_stream_destroy_internal(s);
    // No context was owned -> free must not have been called.
    CHECK(g_fake_free_calls == 0);

    voxtral_stream_test_set_context_factory(nullptr);
    voxtral_stream_test_set_context_free(nullptr);
}

// Two sequential streams from one (fake) model each own a distinct context and
// free it independently. Mirrors the model-shared / context-per-stream contract.
void test_sequential_streams_distinct_contexts() {
    voxtral_stream_test_set_context_factory(fake_ctx_factory);
    voxtral_stream_test_set_context_free(fake_ctx_free);
    g_fake_free_calls = 0;
    voxtral_model * fake_model = reinterpret_cast<voxtral_model *>(0x1);

    voxtral_stream * a = voxtral_stream_create_internal(fake_model, voxtral_context_params{}, default_params());
    CHECK(voxtral_stream_owns_context(a));
    voxtral_stream_destroy_internal(a);
    CHECK(g_fake_free_calls == 1);

    voxtral_stream * bb = voxtral_stream_create_internal(fake_model, voxtral_context_params{}, default_params());
    CHECK(voxtral_stream_owns_context(bb));
    voxtral_stream_destroy_internal(bb);
    CHECK(g_fake_free_calls == 2);   // each stream frees its own context

    voxtral_stream_test_set_context_factory(nullptr);
    voxtral_stream_test_set_context_free(nullptr);
}

// ---------------------------------------------------------------------------
// Reentrancy guard + transient `finishing` state, via the finishing hook.
// The hook runs while state == finishing and the guard is engaged, so every
// reentrant stream call must return `busy` and none may convert the in-flight
// finish into `cancelled`.
// ---------------------------------------------------------------------------
voxtral_stream_state g_hook_state    = voxtral_stream_state::created;
voxtral_status       g_hook_feed     = voxtral_status::ok;
voxtral_status       g_hook_cancel   = voxtral_status::ok;
voxtral_status       g_hook_reset    = voxtral_status::ok;
voxtral_status       g_hook_finish   = voxtral_status::ok;

void reentry_hook(voxtral_stream * s, void *) {
    g_hook_state = voxtral_stream_get_state(s);   // expected: finishing
    const int16_t x = 1;
    g_hook_feed   = voxtral_stream_feed_pcm16_internal(s, &x, 1);
    g_hook_cancel = voxtral_stream_cancel_internal(s);
    g_hook_reset  = voxtral_stream_reset_internal(s);
    g_hook_finish = voxtral_stream_finish_internal(s);
}

void test_finishing_reentrancy_guard() {
    voxtral_stream * s = make_stream();
    voxtral_stream_test_set_finishing_hook(s, reentry_hook, nullptr);

    // Empty finish: enters `finishing`, runs the hook, then completes.
    CHECK(voxtral_stream_finish_internal(s) == voxtral_status::ok);

    // The hook observed the transient state and got `busy` for every reentrant call.
    CHECK(g_hook_state  == voxtral_stream_state::finishing);
    CHECK(g_hook_feed   == voxtral_status::busy);
    CHECK(g_hook_cancel == voxtral_status::busy);
    CHECK(g_hook_reset  == voxtral_status::busy);
    CHECK(g_hook_finish == voxtral_status::busy);

    // finishing -> completed (never cancelled), no CANCELLED event emitted.
    CHECK(voxtral_stream_get_state(s) == voxtral_stream_state::completed);
    int cancelled_events = 0;
    voxtral_stream_event ev;
    while (voxtral_stream_poll_event(s, ev)) {
        if (ev.type == voxtral_stream_event_type::cancelled) ++cancelled_events;
    }
    CHECK(cancelled_events == 0);
    voxtral_stream_destroy_internal(s);
}

// --- event queue is a true hard bound --------------------------------------
void test_event_queue_hard_bound() {
    voxtral_stream * s = make_stream();
    voxtral_stream_test_set_max_events(s, 1);   // shrink the bound for the test

    // Empty finish tries to emit final_text (fits) then completed (overflows).
    CHECK(voxtral_stream_finish_internal(s) == voxtral_status::ok);
    CHECK(voxtral_stream_get_state(s) == voxtral_stream_state::completed);
    // The queue never exceeds the bound.
    CHECK(voxtral_stream_pending_events(s) == 1);
    // Overflow is recorded loudly rather than silently growing/dropping.
    CHECK(voxtral_stream_test_events_overflowed(s));

    // The retained event is the first-queued final_text; completed was dropped.
    voxtral_stream_event ev;
    CHECK(voxtral_stream_poll_event(s, ev));
    CHECK(ev.type == voxtral_stream_event_type::final_text);
    CHECK(!voxtral_stream_poll_event(s, ev));

    // reset() clears the overflow flag for reuse.
    CHECK(voxtral_stream_reset_internal(s) == voxtral_status::ok);
    CHECK(!voxtral_stream_test_events_overflowed(s));
    voxtral_stream_destroy_internal(s);
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
        {"reset_reuse_cycles",        test_reset_reuse_cycles},
        {"reset_after_cancel",        test_reset_after_cancel},
        {"event_ordering",            test_event_ordering},
        {"error_cleared_by_reset",    test_error_cleared_by_reset},
        {"overflow_guards",           test_overflow_guards},
        {"lifecycle_only_ownership",  test_lifecycle_only_ownership},
        {"owned_context_freed",       test_owned_context_freed_on_destroy},
        {"context_creation_failure",  test_context_creation_failure},
        {"sequential_distinct_ctx",   test_sequential_streams_distinct_contexts},
        {"finishing_reentrancy_guard",test_finishing_reentrancy_guard},
        {"event_queue_hard_bound",    test_event_queue_hard_bound},
    };

    for (const auto & t : tests) {
        const int before = g_failures;
        t.fn();
        std::printf("[%s] %s\n", g_failures == before ? "PASS" : "FAIL", t.name);
    }

    std::printf("\n%d checks, %d failure(s)\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}

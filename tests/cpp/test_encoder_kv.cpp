// ============================================================================
// Model-free C++ tests for the per-layer encoder KV-cache RING PLANNER
// (voxtral_enc_kv_plan, Session 6.1). NO GGUF model, NO backend.
//
// The end-to-end numerical parity of the KV encoder against the batch encoder is
// proven on real hardware by the Node.js GPU acceptance suite
// (encoderMaxAbsDeltaVsBatch <= 1e-5 across every chunk plan and rollover). These
// tests instead lock the RING ARITHMETIC that the GGML graph consumes verbatim:
//
//   * ring capacity / max-new / window metadata and the capacity invariant;
//   * append + wrap: the physical segments of a write map pos -> pos%capacity;
//   * the causal window READ tiles [win_start, win_end) in logical order and never
//     observes a frame the WRITE clobbered (correctness after a write, incl. wrap);
//   * eviction: the window advances by exactly n_new, dropping the oldest frames;
//   * absolute positions survive rollover (slot != position);
//   * the causal-window mask predicate matches the sliding-window rule;
//   * the conv input window is even-aligned with >= 2 enc frames of real left
//     context (bit-exact conv), degrading to the true-start zero-pad for q<2;
//   * overflow guards: n_new in (0, max_new], q_start >= 0.
//
// The core method is a GROUND-TRUTH ring model: a slot->abs map is mutated by each
// batch's write segments, then the read segments are checked to enumerate exactly
// the causal window with the correct absolute position at every physical slot. A
// wrap that clobbered a live frame, a lost/duplicated frame, or a slot/position
// confusion all surface as a mismatch here.
//
// Assert-based; no external framework. Exits non-zero on any failure.
// ============================================================================

#include "voxtral-cpp.h"
#include "voxtral-internal.h"

#include <algorithm>
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

const int32_t CAP    = voxtral_enc_kv_capacity_internal();
const int32_t MAXNEW = voxtral_enc_kv_max_new_internal();
const int32_t WIN    = voxtral_enc_kv_window_internal();

// Ground-truth physical ring: ring[s] = absolute enc frame currently in slot s
// (-1 = never written). One per simulated stream.
struct RingModel {
    std::vector<int64_t> slot;
    int32_t cap;
    explicit RingModel(int32_t c) : slot((size_t) c, -1), cap(c) {}

    // Apply a batch's write segments exactly as the graph would: the new frames
    // [q_start, q_start+n_new) are laid down in logical order across the segments.
    void apply_writes(const voxtral_enc_kv_plan_t & p) {
        int64_t abs = p.q_start;
        for (int32_t s = 0; s < p.write_nseg; ++s) {
            for (int32_t i = 0; i < p.write_seg[s].len; ++i) {
                slot[(size_t) (p.write_seg[s].slot + i)] = abs++;
            }
        }
    }
};

// Verify a plan's read segments enumerate EXACTLY [win_start, win_end) in logical
// order, that each physical slot holds the expected absolute position (i.e. the
// write did not clobber a live window frame and slot == abs%cap), and that write
// segments cover [q_start, q_start+n_new) with slot == abs%cap.
void verify_plan_against_model(const voxtral_enc_kv_plan_t & p, const RingModel & m) {
    // WRITE segments: contiguous logical coverage of the new frames, slot=abs%cap.
    int64_t wabs = p.q_start;
    int32_t wtotal = 0;
    for (int32_t s = 0; s < p.write_nseg; ++s) {
        for (int32_t i = 0; i < p.write_seg[s].len; ++i) {
            const int32_t slot = p.write_seg[s].slot + i;
            CHECK(slot >= 0 && slot < CAP);              // in range
            CHECK(slot == (int32_t) (wabs % CAP));       // physical mapping
            ++wabs; ++wtotal;
        }
    }
    CHECK(wtotal == p.n_new);

    // READ segments: tile the causal window in logical order, correct abs per slot.
    int64_t rabs = p.win_start;
    int32_t rtotal = 0;
    for (int32_t s = 0; s < p.read_nseg; ++s) {
        for (int32_t i = 0; i < p.read_seg[s].len; ++i) {
            const int32_t slot = p.read_seg[s].slot + i;
            CHECK(slot >= 0 && slot < CAP);
            CHECK((int32_t) (rabs % CAP) == slot);          // physical mapping
            CHECK(m.slot[(size_t) slot] == rabs);           // not clobbered / correct frame
            ++rabs; ++rtotal;
        }
    }
    CHECK(rtotal == p.win_len);
    CHECK(rabs == p.win_end);
}

// Drive a stream: feed `batches` (each n_new <= MAXNEW), maintaining the ground
// truth ring, verifying every plan after its write. Returns the final head.
int64_t simulate(const std::vector<int32_t> & batches) {
    RingModel m(CAP);
    int64_t head = 0;
    int64_t total_evicted = 0;
    for (int32_t n : batches) {
        voxtral_enc_kv_plan_t p;
        const bool ok = voxtral_enc_kv_plan(head, n, CAP, WIN, p);
        CHECK(ok);
        if (!ok) break;
        // A batch of n queries needs the union of their windows: <= window-1+n
        // frames, always contiguous within the ring capacity.
        CHECK(p.win_len <= WIN - 1 + n);
        CHECK(p.win_len <= CAP);
        // Logical-order segments (ascending slots within contiguous, wrap resets).
        CHECK(p.read_nseg >= 1 && p.read_nseg <= 2);
        CHECK(p.write_nseg >= 1 && p.write_nseg <= 2);
        m.apply_writes(p);
        verify_plan_against_model(p, m);
        total_evicted += p.evicted;
        head += n;
        // Eviction bookkeeping: the cache logically retains [max(0,head-win), head),
        // i.e. min(head, win) <= win frames, and total appended (== head) minus
        // total evicted equals that logical count.
        const int64_t logical = head - std::max<int64_t>(0, head - WIN);
        CHECK(logical <= WIN);
        CHECK(head - total_evicted == logical);
    }
    return head;
}

// -------------------- tests --------------------

void test_metadata() {
    CHECK(WIN == 750);
    CHECK(MAXNEW >= 1);
    CHECK(CAP == WIN + MAXNEW);
    CHECK(CAP - WIN == MAXNEW);   // exactly max_new frames of headroom
}

void test_scheduler_configuration() {
    const char * names[] = {
        "VOXTRAL_ENC_KV_MODE", "VOXTRAL_ENC_KV_GRID",
        "VOXTRAL_ENC_KV_LOGICAL_BATCH", "VOXTRAL_ENC_KV_PHYSICAL_ROWS",
    };
    struct Saved { std::string name; bool set; std::string value; };
    std::vector<Saved> saved;
    for (const char * name : names) {
        const char * value = std::getenv(name);
        saved.push_back({name, value != nullptr, value ? value : ""});
        unsetenv(name);
    }

    CHECK(voxtral_enc_kv_logical_batch_internal() == 4);
    CHECK(voxtral_enc_kv_physical_rows_internal() == 4);
    setenv("VOXTRAL_ENC_KV_PHYSICAL_ROWS", "4", 1);
    CHECK(voxtral_enc_kv_logical_batch_internal() == 4);
    CHECK(voxtral_enc_kv_physical_rows_internal() == 4);
    setenv("VOXTRAL_ENC_KV_PHYSICAL_ROWS", "8", 1);
    CHECK(voxtral_enc_kv_physical_rows_internal() == 8);
    setenv("VOXTRAL_ENC_KV_PHYSICAL_ROWS", "16", 1);
    CHECK(voxtral_enc_kv_physical_rows_internal() == 16);
    setenv("VOXTRAL_ENC_KV_LOGICAL_BATCH", "8", 1);
    setenv("VOXTRAL_ENC_KV_PHYSICAL_ROWS", "64", 1);
    CHECK(voxtral_enc_kv_logical_batch_internal() == 8);
    CHECK(voxtral_enc_kv_physical_rows_internal() == 64);
    setenv("VOXTRAL_ENC_KV_MODE", "throughput", 1);
    CHECK(voxtral_enc_kv_logical_batch_internal() == 128);
    CHECK(voxtral_enc_kv_physical_rows_internal() == 128);

    for (const auto & item : saved) {
        if (item.set) setenv(item.name.c_str(), item.value.c_str(), 1);
        else unsetenv(item.name.c_str());
    }
}

void test_overflow_guards() {
    voxtral_enc_kv_plan_t p;
    CHECK(!voxtral_enc_kv_plan(0, 0, CAP, WIN, p));            // n_new <= 0
    CHECK(!voxtral_enc_kv_plan(0, -3, CAP, WIN, p));
    CHECK(!voxtral_enc_kv_plan(0, MAXNEW + 1, CAP, WIN, p));   // exceeds headroom
    CHECK(!voxtral_enc_kv_plan(-1, 4, CAP, WIN, p));           // negative position
    CHECK(voxtral_enc_kv_plan(0, MAXNEW, CAP, WIN, p));        // exactly max_new OK
    CHECK(voxtral_enc_kv_plan(0, 1, CAP, WIN, p));             // single frame OK
}

void test_first_batch() {
    // At the true start the window is just the new frames, no wrap, conv from 0.
    voxtral_enc_kv_plan_t p;
    CHECK(voxtral_enc_kv_plan(0, 4, CAP, WIN, p));
    CHECK(p.win_start == 0 && p.win_end == 4 && p.win_len == 4);
    CHECK(!p.read_wraps && !p.write_wraps);
    CHECK(p.read_seg[0].slot == 0 && p.read_seg[0].len == 4);
    CHECK(p.conv_mel_start == 0);          // q_start < 2 -> true-start zero-pad
    CHECK(p.conv_mel_end == 8);
    CHECK(p.conv_slice_start == 0);
    CHECK(p.evicted == 0);
}

void test_multi_frame_and_steady_state() {
    // Single-frame, small, and max-new batches all tile correctly.
    simulate(std::vector<int32_t>(4000, 1));      // 4000 single-frame appends
    simulate(std::vector<int32_t>(1000, 4));      // steady 80 ms feeds (~4 enc/feed)
    simulate(std::vector<int32_t>(200, MAXNEW));  // max-new batches
    // Mixed / irregular batch sizes crossing several rollovers.
    std::vector<int32_t> mixed;
    for (int i = 0; i < 500; ++i) mixed.push_back(1 + (i * 37) % MAXNEW);
    simulate(mixed);
}

void test_window_boundary_positions() {
    // Stage 7/8: exact positions around 0,1,748,749,750,751 and multiples of the
    // window / capacity, each verified against the ground-truth ring after write.
    const std::vector<int64_t> heads = {
        0, 1, 2, WIN - 2, WIN - 1, WIN, WIN + 1, 2 * WIN - 1, 2 * WIN, 2 * WIN + 1,
        CAP - 1, CAP, CAP + 1, 3 * WIN, 5 * CAP + 3, 100000,
    };
    for (int64_t head : heads) {
        for (int32_t n : {1, 2, 4, 8, MAXNEW}) {
            // Rebuild a ring consistent with `head` frames already written.
            RingModel m(CAP);
            const int64_t first = std::max<int64_t>(0, head - WIN);  // oldest still-relevant
            for (int64_t a = first; a < head; ++a) m.slot[(size_t) (a % CAP)] = a;
            voxtral_enc_kv_plan_t p;
            CHECK(voxtral_enc_kv_plan(head, n, CAP, WIN, p));
            m.apply_writes(p);
            verify_plan_against_model(p, m);
            // The read window's earliest frame is exactly max(0, head-(WIN-1)).
            CHECK(p.win_start == std::max<int64_t>(0, head - (WIN - 1)));
        }
    }
}

void test_absolute_positions_after_rollover() {
    // The same physical slot holds very different absolute positions across
    // rollovers; RoPE must key on the absolute position, never the slot.
    voxtral_enc_kv_plan_t p0, p1;
    CHECK(voxtral_enc_kv_plan(0, 1, CAP, WIN, p0));
    CHECK(voxtral_enc_kv_plan(CAP, 1, CAP, WIN, p1));      // exactly one rollover later
    CHECK(p0.write_seg[0].slot == p1.write_seg[0].slot);  // same physical slot 0
    CHECK(p0.q_start != p1.q_start);                       // different absolute pos
    CHECK(p1.q_start == CAP);
    // Specific spec positions: slot(750)=750%1006, slot(1500)=1500%1006=494, etc.
    for (int64_t pos : {0, 1, 748, 749, 750, 751, 1499, 1500, 1501}) {
        voxtral_enc_kv_plan_t p;
        CHECK(voxtral_enc_kv_plan(pos, 1, CAP, WIN, p));
        CHECK(p.write_seg[0].slot == (int32_t) (pos % CAP));
    }
}

void test_wrap_segments() {
    // A batch whose new frames straddle the physical end of the ring must split
    // into two write segments; likewise a window that straddles the end.
    // Pick head so that head % CAP is within MAXNEW of CAP.
    const int64_t head = CAP - 3;   // slot 1003; a 8-frame write wraps
    RingModel m(CAP);
    const int64_t first = std::max<int64_t>(0, head - WIN);
    for (int64_t a = first; a < head; ++a) m.slot[(size_t) (a % CAP)] = a;
    voxtral_enc_kv_plan_t p;
    CHECK(voxtral_enc_kv_plan(head, 8, CAP, WIN, p));
    CHECK(p.write_wraps);
    CHECK(p.write_nseg == 2);
    CHECK(p.write_seg[0].slot == (int32_t) (head % CAP));
    CHECK(p.write_seg[0].len + p.write_seg[1].len == 8);
    CHECK(p.write_seg[1].slot == 0);
    m.apply_writes(p);
    verify_plan_against_model(p, m);
    // The window read for this batch also wraps (it ends at head+8 > CAP boundary).
    CHECK(p.read_wraps);
}

void test_mask_predicate() {
    // Sliding causal window [q-(WIN-1), q].
    CHECK(voxtral_enc_kv_mask_allows(/*kv*/5, /*q*/5, WIN));         // self
    CHECK(voxtral_enc_kv_mask_allows(0, 5, WIN));                   // past, in window
    CHECK(!voxtral_enc_kv_mask_allows(6, 5, WIN));                  // future
    CHECK(voxtral_enc_kv_mask_allows(10, 10 + (WIN - 1), WIN));     // exactly window edge
    CHECK(!voxtral_enc_kv_mask_allows(10, 10 + WIN, WIN));          // just past edge
    // Consistency with the reference run_encoder_chunk mask rule for a batch.
    const int64_t P = 2000;
    for (int32_t ql = 0; ql < 4; ++ql) {
        const int64_t q = P + ql;
        const int64_t win_start = std::max<int64_t>(0, q - (WIN - 1));
        for (int64_t kv = win_start; kv <= q; ++kv) {
            CHECK(voxtral_enc_kv_mask_allows(kv, q, WIN));
        }
        CHECK(!voxtral_enc_kv_mask_allows(win_start - 1, q, WIN));
        CHECK(!voxtral_enc_kv_mask_allows(q + 1, q, WIN));
    }
}

void test_conv_window_alignment() {
    // conv_mel_start is EVEN, so local conv1 frame conv_slice_start maps to global
    // enc frame q_start (enc e == local conv1 j + conv_mel_start/2). For q_start>=2
    // the slice starts at local frame 2 (>= 2 real-history frames -> bit-exact); at
    // the true start it slices from 0 with the intended zero-pad.
    for (int64_t P : {0, 1, 2, 3, 100, 749, 750, 4000, 100000}) {
        voxtral_enc_kv_plan_t p;
        CHECK(voxtral_enc_kv_plan(P, 8, CAP, WIN, p));
        CHECK(p.conv_mel_start % 2 == 0);                      // even alignment
        CHECK(p.conv_mel_start >= 0);
        CHECK(p.conv_mel_end == 2 * (P + 8));
        // global enc P == local conv1 frame conv_slice_start + conv_mel_start/2
        CHECK((int64_t) p.conv_slice_start + p.conv_mel_start / 2 == P);
        if (P >= 2) {
            CHECK(p.conv_slice_start == 2);                    // 2 real-history frames
        } else {
            CHECK(p.conv_mel_start == 0);                      // true start
            CHECK(p.conv_slice_start == (int32_t) P);
        }
        // The fed Mel window must cover the receptive field of the last wanted enc
        // frame: Mel index 2*(P+8)-1 <= conv_mel_end-1. (upper bound reachable)
        CHECK(p.conv_mel_end - 1 >= 2 * (P + 8) - 1);
    }
}

void test_batch_segmentation() {
    // A driver splits an oversized flush (> MAXNEW new frames) into <= MAXNEW
    // sub-batches with no lost/duplicated frame; each sub-batch plans cleanly and
    // the ground-truth ring stays consistent.
    RingModel m(CAP);
    int64_t head = 0;
    int64_t remaining = 5000;   // e.g. a 100 s synthetic flush at once
    int64_t produced = 0;
    while (remaining > 0) {
        const int32_t n = (int32_t) std::min<int64_t>(MAXNEW, remaining);
        voxtral_enc_kv_plan_t p;
        CHECK(voxtral_enc_kv_plan(head, n, CAP, WIN, p));
        m.apply_writes(p);
        verify_plan_against_model(p, m);
        head += n; produced += n; remaining -= n;
    }
    CHECK(produced == 5000);
    CHECK(head == 5000);
}

} // namespace

int main() {
    test_metadata();
    test_scheduler_configuration();
    test_overflow_guards();
    test_first_batch();
    test_multi_frame_and_steady_state();
    test_window_boundary_positions();
    test_absolute_positions_after_rollover();
    test_wrap_segments();
    test_mask_predicate();
    test_conv_window_alignment();
    test_batch_segmentation();

    std::fprintf(stderr, "voxtral_encoder_kv_unit: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}

// Model-free tests for the Session-8 decoder KV circular planner.  The Vulkan
// graph consumes this exact plan; these checks lock the risky wrap/eviction
// arithmetic independently of a model or backend.

#include "voxtral-internal.h"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

int checks = 0;
int failures = 0;

#define CHECK(expr) do { \
    ++checks; \
    if (!(expr)) { \
        ++failures; \
        std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #expr); \
    } \
} while (0)

void commit(voxtral_decoder_kv_ring & ring, const voxtral_decoder_kv_plan_t & p) {
    ring.used = p.used_after;
    ring.oldest_absolute_position = p.oldest_after;
    ring.next_absolute_position = p.next_after;
    ring.evictions += p.evicted;
    if (p.wrapped) ++ring.wraps;
}

std::vector<int64_t> logical_positions(const voxtral_decoder_kv_plan_t & p,
                                       int32_t capacity) {
    std::vector<int64_t> result;
    for (int32_t s = 0; s < p.read_nseg; ++s) {
        for (int32_t i = 0; i < p.read_seg[s].len; ++i) {
            const int32_t slot = p.read_seg[s].slot + i;
            // Recover the unique absolute position in [oldest,next) with this slot.
            for (int64_t a = p.oldest_after; a < p.next_after; ++a) {
                if (a % capacity == slot) { result.push_back(a); break; }
            }
        }
    }
    return result;
}

void test_initial_and_contiguous() {
    voxtral_decoder_kv_ring ring;
    ring.capacity = 8;
    voxtral_decoder_kv_plan_t p;
    CHECK(voxtral_decoder_kv_plan(ring, 0, 3, p));
    CHECK(p.write_nseg == 1 && p.write_seg[0].slot == 0 && p.write_seg[0].len == 3);
    CHECK(p.read_nseg == 1 && p.read_seg[0].slot == 0 && p.read_seg[0].len == 3);
    CHECK(p.used_after == 3 && p.oldest_after == 0 && p.next_after == 3);
    CHECK(p.evicted == 0 && !p.wrapped);
    commit(ring, p);
    CHECK(ring.used == 3 && ring.next_absolute_position == 3);
}

void test_straddled_write_and_logical_read() {
    voxtral_decoder_kv_ring ring;
    ring.capacity = 8;
    ring.used = 6;
    ring.oldest_absolute_position = 0;
    ring.next_absolute_position = 6;
    voxtral_decoder_kv_plan_t p;
    CHECK(voxtral_decoder_kv_plan(ring, 6, 4, p));
    CHECK(p.write_nseg == 2);
    CHECK(p.write_seg[0].slot == 6 && p.write_seg[0].len == 2);
    CHECK(p.write_seg[1].slot == 0 && p.write_seg[1].len == 2);
    CHECK(p.used_after == 8 && p.oldest_after == 2 && p.next_after == 10);
    CHECK(p.evicted == 2 && p.wrapped);
    CHECK(p.read_nseg == 2);
    CHECK(p.read_seg[0].slot == 2 && p.read_seg[0].len == 6);
    CHECK(p.read_seg[1].slot == 0 && p.read_seg[1].len == 2);
    const auto logical = logical_positions(p, 8);
    CHECK(logical.size() == 8);
    for (size_t i = 0; i < logical.size(); ++i) CHECK(logical[i] == (int64_t) i + 2);
}

void test_single_step_rollover_and_eviction() {
    voxtral_decoder_kv_ring ring;
    ring.capacity = 8;
    std::vector<int64_t> physical(8, -1);
    for (int64_t pos = 0; pos < 25; ++pos) {
        voxtral_decoder_kv_plan_t p;
        CHECK(voxtral_decoder_kv_plan(ring, pos, 1, p));
        physical[(size_t) p.write_seg[0].slot] = pos;
        commit(ring, p);
        CHECK(ring.next_absolute_position == pos + 1);
        CHECK(ring.used == (pos < 7 ? pos + 1 : 8));
        CHECK(ring.oldest_absolute_position == (pos < 8 ? 0 : pos - 7));
        const auto logical = logical_positions(p, 8);
        CHECK((int64_t) logical.size() == ring.used);
        for (size_t i = 0; i < logical.size(); ++i) {
            CHECK(logical[i] == ring.oldest_absolute_position + (int64_t) i);
            CHECK(physical[(size_t) (logical[i] % 8)] == logical[i]);
        }
    }
    CHECK(ring.wraps == 3);       // writes at absolute 8, 16, 24
    CHECK(ring.evictions == 17);  // 25 appended into capacity 8
    CHECK(ring.bytes_moved == 0 && ring.full_buffer_moves == 0);
    CHECK(ring.oldest_absolute_position == 17);
}

void test_invalid_plans() {
    voxtral_decoder_kv_ring ring;
    ring.capacity = 8;
    voxtral_decoder_kv_plan_t p;
    CHECK(!voxtral_decoder_kv_plan(ring, 1, 1, p)); // discontinuity
    CHECK(!voxtral_decoder_kv_plan(ring, 0, 0, p));
    CHECK(!voxtral_decoder_kv_plan(ring, 0, 9, p));
    ring.used = 2; // inconsistent: oldest + used != next
    CHECK(!voxtral_decoder_kv_plan(ring, 0, 1, p));
}

} // namespace

int main() {
    test_initial_and_contiguous();
    test_straddled_write_and_logical_read();
    test_single_step_rollover_and_eviction();
    test_invalid_plans();
    std::fprintf(stderr, "voxtral_decoder_kv_unit: %d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}

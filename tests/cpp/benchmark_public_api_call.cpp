#include "../../include/voxtral-stream.h"
#include "../../src/voxtral-stream.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>

namespace {

using clock_type = std::chrono::steady_clock;

template <typename Function>
double measure_ns_per_call(Function && function, uint64_t iterations) {
    volatile uint64_t status_sum = 0;
    const auto start = clock_type::now();
    for (uint64_t i = 0; i < iterations; ++i) {
        status_sum += static_cast<uint64_t>(function());
    }
    const auto finish = clock_type::now();
    if (status_sum != 0) return -1.0;
    const double elapsed_ns =
        std::chrono::duration<double, std::nano>(finish - start).count();
    return elapsed_ns / static_cast<double>(iterations);
}

double median(std::array<double, 7> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2u];
}

} // namespace

int main() {
    voxtral_context_params context_params{};
    voxtral_stream_params_internal stream_params{};
    voxtral_stream * stream = voxtral_stream_create_internal(
        nullptr, context_params, stream_params);
    if (!stream) return 1;

    constexpr uint64_t warmup_iterations = 10000;
    constexpr uint64_t measured_iterations = 1000000;
    size_t consumed = 1;
    for (uint64_t i = 0; i < warmup_iterations; ++i) {
        if (voxtral_stream_feed_pcm16_internal(stream, nullptr, 0) !=
                voxtral_status_internal::ok ||
            voxtral_stream_feed_pcm16(stream, nullptr, 0, &consumed) !=
                VOXTRAL_STATUS_OK ||
            consumed != 0) {
            voxtral_stream_destroy_internal(stream);
            return 1;
        }
    }

    std::array<double, 7> direct{};
    std::array<double, 7> public_api{};
    for (size_t repeat = 0; repeat < direct.size(); ++repeat) {
        if ((repeat & 1u) == 0) {
            direct[repeat] = measure_ns_per_call([&] {
                return voxtral_stream_feed_pcm16_internal(stream, nullptr, 0);
            }, measured_iterations);
            public_api[repeat] = measure_ns_per_call([&] {
                size_t count = 1;
                const voxtral_status status =
                    voxtral_stream_feed_pcm16(stream, nullptr, 0, &count);
                return status == VOXTRAL_STATUS_OK && count == 0
                    ? VOXTRAL_STATUS_OK
                    : VOXTRAL_STATUS_INTERNAL_ERROR;
            }, measured_iterations);
        } else {
            public_api[repeat] = measure_ns_per_call([&] {
                size_t count = 1;
                const voxtral_status status =
                    voxtral_stream_feed_pcm16(stream, nullptr, 0, &count);
                return status == VOXTRAL_STATUS_OK && count == 0
                    ? VOXTRAL_STATUS_OK
                    : VOXTRAL_STATUS_INTERNAL_ERROR;
            }, measured_iterations);
            direct[repeat] = measure_ns_per_call([&] {
                return voxtral_stream_feed_pcm16_internal(stream, nullptr, 0);
            }, measured_iterations);
        }
    }

    const double direct_median = median(direct);
    const double public_median = median(public_api);
    const double adapter_ns = std::max(0.0, public_median - direct_median);
    const bool pass = direct_median >= 0.0 && public_median >= 0.0 &&
                      adapter_ns <= 100000.0;
    std::printf(
        "{\"ok\":%s,\"iterations\":%llu,"
        "\"directMedianNs\":%.3f,\"publicMedianNs\":%.3f,"
        "\"adapterOverheadNs\":%.3f,\"adapterOverheadMs\":%.9f}\n",
        pass ? "true" : "false",
        static_cast<unsigned long long>(measured_iterations),
        direct_median, public_median, adapter_ns, adapter_ns / 1e6);

    voxtral_stream_destroy_internal(stream);
    return pass ? 0 : 1;
}

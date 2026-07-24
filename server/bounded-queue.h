#ifndef VOXTRAL_SERVER_BOUNDED_QUEUE_H
#define VOXTRAL_SERVER_BOUNDED_QUEUE_H

#include "gpu-lease.h"

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <utility>
#include <vector>

namespace voxtral::server {

class PcmQueue {
public:
    enum class PopResult {
        Audio,
        Finish,
        Cancel,
    };

    explicit PcmQueue(std::size_t capacity_samples)
        : capacity_samples_(capacity_samples) {}

    std::size_t push_some(
        const std::int16_t * samples,
        std::size_t sample_count)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (cancelled_ || finishing_ || sample_count == 0) {
            return 0;
        }
        const std::size_t available =
            capacity_samples_ > queued_samples_
                ? capacity_samples_ - queued_samples_
                : 0;
        const std::size_t accepted = std::min(available, sample_count);
        if (accepted == 0) {
            return 0;
        }
        constexpr std::size_t kWorkerChunkSamples = 1280;
        std::size_t offset = 0;
        while (offset < accepted) {
            const std::size_t count =
                std::min(kWorkerChunkSamples, accepted - offset);
            queue_.emplace_back(samples + offset, samples + offset + count);
            offset += count;
        }
        queued_samples_ += accepted;
        condition_.notify_one();
        return accepted;
    }

    PopResult pop(std::vector<std::int16_t> & output) {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] {
            return cancelled_ || !queue_.empty() || finishing_;
        });
        if (cancelled_) {
            return PopResult::Cancel;
        }
        if (!queue_.empty()) {
            output = std::move(queue_.front());
            queue_.pop_front();
            queued_samples_ -= output.size();
            return PopResult::Audio;
        }
        return PopResult::Finish;
    }

    void finish() {
        std::lock_guard<std::mutex> lock(mutex_);
        finishing_ = true;
        condition_.notify_all();
    }

    void cancel() {
        std::lock_guard<std::mutex> lock(mutex_);
        cancelled_ = true;
        queue_.clear();
        queued_samples_ = 0;
        condition_.notify_all();
    }

    std::size_t queued_samples() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queued_samples_;
    }

    std::size_t capacity_samples() const noexcept {
        return capacity_samples_;
    }

private:
    const std::size_t capacity_samples_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::deque<std::vector<std::int16_t>> queue_;
    std::size_t queued_samples_ = 0;
    bool finishing_ = false;
    bool cancelled_ = false;
};

} // namespace voxtral::server

#endif

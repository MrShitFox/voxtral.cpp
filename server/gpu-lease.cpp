#include "gpu-lease.h"

#include <utility>

namespace voxtral::server {

void Cancellation::request(CancelReason reason) noexcept {
    if (reason == CancelReason::None) {
        return;
    }
    CancelReason expected = CancelReason::None;
    reason_.compare_exchange_strong(
        expected, reason, std::memory_order_release, std::memory_order_relaxed);
}

bool Cancellation::requested() const noexcept {
    return reason() != CancelReason::None;
}

CancelReason Cancellation::reason() const noexcept {
    return reason_.load(std::memory_order_acquire);
}

GpuLease::GpuLease(
    GpuLeaseManager * manager,
    GpuMode mode,
    std::shared_ptr<Cancellation> cancellation)
    : manager_(manager),
      mode_(mode),
      cancellation_(std::move(cancellation)) {}

GpuLease::GpuLease(GpuLease && other) noexcept
    : manager_(other.manager_),
      mode_(other.mode_),
      cancellation_(std::move(other.cancellation_))
{
    other.manager_ = nullptr;
    other.mode_ = GpuMode::Free;
}

GpuLease & GpuLease::operator=(GpuLease && other) noexcept {
    if (this != &other) {
        reset();
        manager_ = other.manager_;
        mode_ = other.mode_;
        cancellation_ = std::move(other.cancellation_);
        other.manager_ = nullptr;
        other.mode_ = GpuMode::Free;
    }
    return *this;
}

GpuLease::~GpuLease() {
    reset();
}

GpuLease::operator bool() const noexcept {
    return manager_ != nullptr;
}

GpuMode GpuLease::mode() const noexcept {
    return mode_;
}

const std::shared_ptr<Cancellation> & GpuLease::cancellation() const noexcept {
    return cancellation_;
}

void GpuLease::reset() noexcept {
    if (manager_) {
        manager_->release(mode_);
        manager_ = nullptr;
        mode_ = GpuMode::Free;
        cancellation_.reset();
    }
}

GpuLease GpuLeaseManager::try_acquire(GpuMode mode) {
    if (mode != GpuMode::Batch && mode != GpuMode::Realtime) {
        return {};
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (mode_ != GpuMode::Free) {
        return {};
    }
    active_cancellation_ = std::make_shared<Cancellation>();
    mode_ = mode;
    return GpuLease(this, mode, active_cancellation_);
}

void GpuLeaseManager::begin_shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_cancellation_) {
        active_cancellation_->request(CancelReason::Shutdown);
    }
    mode_ = GpuMode::ShuttingDown;
}

GpuMode GpuLeaseManager::mode() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return mode_;
}

bool GpuLeaseManager::busy() const {
    const GpuMode current = mode();
    return current == GpuMode::Batch || current == GpuMode::Realtime;
}

bool GpuLeaseManager::shutting_down() const {
    return mode() == GpuMode::ShuttingDown;
}

std::string GpuLeaseManager::busy_mode_name() const {
    switch (mode()) {
        case GpuMode::Batch:
            return "batch";
        case GpuMode::Realtime:
            return "realtime";
        default:
            return {};
    }
}

void GpuLeaseManager::release(GpuMode mode) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (mode_ == mode) {
        mode_ = GpuMode::Free;
    }
    active_cancellation_.reset();
}

} // namespace voxtral::server

#ifndef VOXTRAL_SERVER_GPU_LEASE_H
#define VOXTRAL_SERVER_GPU_LEASE_H

#include <atomic>
#include <memory>
#include <mutex>
#include <string>

namespace voxtral::server {

enum class GpuMode {
    Free,
    Batch,
    Realtime,
    ShuttingDown,
};

enum class CancelReason {
    None,
    Client,
    Disconnect,
    Shutdown,
    Capacity,
    Protocol,
};

class Cancellation {
public:
    void request(CancelReason reason) noexcept;
    bool requested() const noexcept;
    CancelReason reason() const noexcept;

private:
    std::atomic<CancelReason> reason_{CancelReason::None};
};

class GpuLeaseManager;

class GpuLease {
public:
    GpuLease() = default;
    GpuLease(const GpuLease &) = delete;
    GpuLease & operator=(const GpuLease &) = delete;
    GpuLease(GpuLease && other) noexcept;
    GpuLease & operator=(GpuLease && other) noexcept;
    ~GpuLease();

    explicit operator bool() const noexcept;
    GpuMode mode() const noexcept;
    const std::shared_ptr<Cancellation> & cancellation() const noexcept;
    void reset() noexcept;

private:
    friend class GpuLeaseManager;
    GpuLease(
        GpuLeaseManager * manager,
        GpuMode mode,
        std::shared_ptr<Cancellation> cancellation);

    GpuLeaseManager * manager_ = nullptr;
    GpuMode mode_ = GpuMode::Free;
    std::shared_ptr<Cancellation> cancellation_;
};

class GpuLeaseManager {
public:
    GpuLease try_acquire(GpuMode mode);
    void begin_shutdown();
    GpuMode mode() const;
    bool busy() const;
    bool shutting_down() const;
    std::string busy_mode_name() const;

private:
    friend class GpuLease;
    void release(GpuMode mode) noexcept;

    mutable std::mutex mutex_;
    GpuMode mode_ = GpuMode::Free;
    std::shared_ptr<Cancellation> active_cancellation_;
};

} // namespace voxtral::server

#endif

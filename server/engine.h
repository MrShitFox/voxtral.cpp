#ifndef VOXTRAL_SERVER_ENGINE_H
#define VOXTRAL_SERVER_ENGINE_H

#include "gpu-lease.h"
#include "logger.h"

#include <voxtral-stream.h>
#include <voxtral.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace voxtral::server {

class EngineError : public std::runtime_error {
public:
    EngineError(voxtral_status status, const std::string & detail);
    voxtral_status status() const noexcept;

private:
    voxtral_status status_;
};

struct ModelDeleter {
    void operator()(voxtral_model * model) const noexcept;
};

struct ContextDeleter {
    void operator()(voxtral_context * context) const noexcept;
};

struct StreamDeleter {
    void operator()(voxtral_stream * stream) const noexcept;
};

using ModelHandle = std::unique_ptr<voxtral_model, ModelDeleter>;
using ContextHandle = std::unique_ptr<voxtral_context, ContextDeleter>;
using StreamHandle = std::unique_ptr<voxtral_stream, StreamDeleter>;

class EngineStream {
public:
    explicit EngineStream(voxtral_context * context);
    EngineStream(const EngineStream &) = delete;
    EngineStream & operator=(const EngineStream &) = delete;
    ~EngineStream();

    voxtral_status feed(
        const std::int16_t * samples,
        std::size_t sample_count,
        std::size_t * samples_consumed);
    voxtral_status poll(voxtral_event & event);
    voxtral_status finish();
    voxtral_status cancel();
    voxtral_stream_state state() const;
    voxtral_stream_metrics metrics() const;
    std::string final_text() const;
    std::string last_error() const;

private:
    StreamHandle stream_;
};

class Engine {
public:
    Engine(const std::string & model_path, Logger & logger);
    Engine(const Engine &) = delete;
    Engine & operator=(const Engine &) = delete;
    ~Engine() = default;

    std::unique_ptr<EngineStream> create_stream();
    const voxtral_capabilities & capabilities() const noexcept;
    GpuLeaseManager & leases() noexcept;
    const GpuLeaseManager & leases() const noexcept;
    std::string model_name() const;

private:
    Logger & logger_;
    ModelHandle model_;
    ContextHandle context_;
    voxtral_capabilities capabilities_{};
    GpuLeaseManager leases_;
};

std::string safe_status_message(voxtral_status status);
std::string api_error_code(voxtral_status status);
int api_http_status(voxtral_status status);

} // namespace voxtral::server

#endif

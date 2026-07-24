#include "engine.h"

#include <algorithm>
#include <utility>
#include <vector>

namespace voxtral::server {
namespace {

std::string copy_error(const voxtral_error_info & error) {
    const auto length = std::find(
        std::begin(error.message), std::end(error.message), '\0');
    return std::string(std::begin(error.message), length);
}

} // namespace

EngineError::EngineError(voxtral_status status, const std::string & detail)
    : std::runtime_error(detail),
      status_(status) {}

voxtral_status EngineError::status() const noexcept {
    return status_;
}

void ModelDeleter::operator()(voxtral_model * model) const noexcept {
    voxtral_model_destroy(model);
}

void ContextDeleter::operator()(voxtral_context * context) const noexcept {
    voxtral_context_destroy(context);
}

void StreamDeleter::operator()(voxtral_stream * stream) const noexcept {
    voxtral_stream_destroy(stream);
}

EngineStream::EngineStream(voxtral_context * context) {
    voxtral_stream * stream = nullptr;
    voxtral_stream_params parameters = voxtral_stream_default_params();
    const voxtral_status status =
        voxtral_stream_create(context, &parameters, &stream);
    if (status != VOXTRAL_STATUS_OK) {
        voxtral_error_info error{};
        error.struct_size = sizeof(error);
        error.api_version = VOXTRAL_API_VERSION;
        std::string detail = safe_status_message(status);
        if (voxtral_context_get_last_error(context, &error) ==
                VOXTRAL_STATUS_OK &&
            error.message[0] != '\0') {
            detail = copy_error(error);
        }
        throw EngineError(status, detail);
    }
    stream_.reset(stream);
}

EngineStream::~EngineStream() {
    if (stream_) {
        const voxtral_stream_state current = state();
        if (current == VOXTRAL_STREAM_CREATED ||
            current == VOXTRAL_STREAM_ACTIVE ||
            current == VOXTRAL_STREAM_FINISHING) {
            (void) cancel();
        }
    }
}

voxtral_status EngineStream::feed(
    const std::int16_t * samples,
    std::size_t sample_count,
    std::size_t * samples_consumed)
{
    return voxtral_stream_feed_pcm16(
        stream_.get(), samples, sample_count, samples_consumed);
}

voxtral_status EngineStream::poll(voxtral_event & event) {
    event = {};
    event.struct_size = sizeof(event);
    event.api_version = VOXTRAL_API_VERSION;
    return voxtral_stream_poll_event(stream_.get(), &event);
}

voxtral_status EngineStream::finish() {
    return voxtral_stream_finish(stream_.get());
}

voxtral_status EngineStream::cancel() {
    return voxtral_stream_cancel(stream_.get());
}

voxtral_stream_state EngineStream::state() const {
    return voxtral_stream_get_state(stream_.get());
}

voxtral_stream_metrics EngineStream::metrics() const {
    voxtral_stream_metrics metrics{};
    metrics.struct_size = sizeof(metrics);
    metrics.api_version = VOXTRAL_API_VERSION;
    const voxtral_status status =
        voxtral_stream_get_metrics(stream_.get(), &metrics);
    if (status != VOXTRAL_STATUS_OK) {
        throw EngineError(status, last_error());
    }
    return metrics;
}

std::string EngineStream::final_text() const {
    std::size_t required = 0;
    voxtral_status status =
        voxtral_stream_get_final_text(stream_.get(), nullptr, 0, &required);
    if (status != VOXTRAL_STATUS_OK) {
        throw EngineError(status, last_error());
    }
    if (required == 0) {
        throw EngineError(
            VOXTRAL_STATUS_INTERNAL_ERROR,
            "the final transcript size is invalid");
    }
    std::vector<char> buffer(required);
    status = voxtral_stream_get_final_text(
        stream_.get(), buffer.data(), buffer.size(), &required);
    if (status != VOXTRAL_STATUS_OK) {
        throw EngineError(status, last_error());
    }
    return std::string(buffer.data());
}

std::string EngineStream::last_error() const {
    voxtral_error_info error{};
    error.struct_size = sizeof(error);
    error.api_version = VOXTRAL_API_VERSION;
    if (voxtral_stream_get_last_error(stream_.get(), &error) !=
        VOXTRAL_STATUS_OK) {
        return "failed to query the public stream error";
    }
    return copy_error(error);
}

Engine::Engine(const std::string & model_path, Logger & logger)
    : logger_(logger)
{
    voxtral_model_params model_parameters = voxtral_model_default_params();
    model_parameters.backend = VOXTRAL_BACKEND_VULKAN;
    model_parameters.log_level = VOXTRAL_LOG_INFO;

    voxtral_model * model = nullptr;
    voxtral_status status =
        voxtral_model_load(model_path.c_str(), &model_parameters, &model);
    if (status != VOXTRAL_STATUS_OK) {
        throw EngineError(status, "model load failed: " + safe_status_message(status));
    }
    model_.reset(model);
    logger_.info("model_loaded", {{"backend", "vulkan"}});

    voxtral_context_config context_config = voxtral_context_default_config();
    voxtral_context * context = nullptr;
    status = voxtral_context_create(model_.get(), &context_config, &context);
    if (status != VOXTRAL_STATUS_OK) {
        throw EngineError(
            status, "context creation failed: " + safe_status_message(status));
    }
    context_.reset(context);

    capabilities_.struct_size = sizeof(capabilities_);
    capabilities_.api_version = VOXTRAL_API_VERSION;
    status = voxtral_context_get_capabilities(context_.get(), &capabilities_);
    if (status != VOXTRAL_STATUS_OK ||
        capabilities_.sample_rate != 16000 ||
        capabilities_.channels != 1 ||
        capabilities_.audio_format != VOXTRAL_AUDIO_PCM_S16LE ||
        capabilities_.supports_incremental == 0 ||
        capabilities_.max_active_streams_per_context != 1) {
        throw EngineError(
            status == VOXTRAL_STATUS_OK ? VOXTRAL_STATUS_MODEL_ERROR : status,
            "model does not provide the required realtime public API capabilities");
    }
    logger_.info("context_ready", {
        {"sample_rate", capabilities_.sample_rate},
        {"max_active_streams", capabilities_.max_active_streams_per_context},
    });
}

std::unique_ptr<EngineStream> Engine::create_stream() {
    return std::make_unique<EngineStream>(context_.get());
}

const voxtral_capabilities & Engine::capabilities() const noexcept {
    return capabilities_;
}

GpuLeaseManager & Engine::leases() noexcept {
    return leases_;
}

const GpuLeaseManager & Engine::leases() const noexcept {
    return leases_;
}

std::string Engine::model_name() const {
    return "voxtral-mini";
}

std::string safe_status_message(voxtral_status status) {
    switch (status) {
        case VOXTRAL_STATUS_INVALID_ARGUMENT:
            return "The inference request is invalid.";
        case VOXTRAL_STATUS_INVALID_STATE:
            return "The inference stream is in an invalid state.";
        case VOXTRAL_STATUS_OUT_OF_MEMORY:
            return "The server ran out of memory.";
        case VOXTRAL_STATUS_QUEUE_FULL:
            return "The inference output queue is full.";
        case VOXTRAL_STATUS_NOT_READY:
            return "The inference result is not ready.";
        case VOXTRAL_STATUS_CANCELLED:
            return "The inference request was cancelled.";
        case VOXTRAL_STATUS_UNSUPPORTED_AUDIO_FORMAT:
            return "The audio format is unsupported.";
        case VOXTRAL_STATUS_MODEL_ERROR:
            return "The model cannot process this request.";
        case VOXTRAL_STATUS_BACKEND_ERROR:
            return "The inference backend failed.";
        case VOXTRAL_STATUS_INTERNAL_ERROR:
        case VOXTRAL_STATUS_BUFFER_TOO_SMALL:
            return "The server encountered an internal inference error.";
        case VOXTRAL_STATUS_OK:
            return "OK";
    }
    return "The server encountered an unknown inference error.";
}

std::string api_error_code(voxtral_status status) {
    switch (status) {
        case VOXTRAL_STATUS_INVALID_ARGUMENT:
            return "invalid_argument";
        case VOXTRAL_STATUS_INVALID_STATE:
            return "invalid_state";
        case VOXTRAL_STATUS_OUT_OF_MEMORY:
            return "out_of_memory";
        case VOXTRAL_STATUS_UNSUPPORTED_AUDIO_FORMAT:
            return "unsupported_audio_format";
        case VOXTRAL_STATUS_MODEL_ERROR:
            return "model_error";
        case VOXTRAL_STATUS_BACKEND_ERROR:
            return "backend_error";
        case VOXTRAL_STATUS_CANCELLED:
            return "cancelled";
        default:
            return "internal_error";
    }
}

int api_http_status(voxtral_status status) {
    switch (status) {
        case VOXTRAL_STATUS_INVALID_ARGUMENT:
            return 400;
        case VOXTRAL_STATUS_INVALID_STATE:
            return 409;
        case VOXTRAL_STATUS_UNSUPPORTED_AUDIO_FORMAT:
            return 415;
        default:
            return 500;
    }
}

} // namespace voxtral::server

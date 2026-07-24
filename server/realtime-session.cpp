#include "realtime-session.h"

#include <boost/asio/post.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <limits>
#include <utility>

namespace voxtral::server {
namespace beast = boost::beast;
namespace http = beast::http;
namespace websocket = beast::websocket;

namespace {

constexpr std::chrono::seconds kConfigurationTimeout{5};

std::size_t realtime_capacity_samples(const ServerConfig & config) {
    constexpr std::uint64_t samples_per_ms = 16;
    const std::uint64_t samples =
        static_cast<std::uint64_t>(config.realtime_buffer_ms) *
        samples_per_ms;
    if (samples == 0 ||
        samples > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("realtime PCM buffer size is invalid");
    }
    return static_cast<std::size_t>(samples);
}

} // namespace

RealtimeSession::RealtimeSession(
    ServerApp & app,
    beast::tcp_stream stream,
    GpuLease lease,
    std::string session_id)
    : app_(app),
      websocket_(std::move(stream)),
      lease_(std::move(lease)),
      session_id_(std::move(session_id)),
      input_queue_(realtime_capacity_samples(app.config())),
      configuration_timer_(app.io_context()),
      idle_timer_(app.io_context()) {}

void RealtimeSession::run(Request request) {
    handshake_request_ = std::make_unique<Request>(std::move(request));
    beast::get_lowest_layer(websocket_).expires_after(kConfigurationTimeout);
    websocket_.set_option(websocket::stream_base::timeout{
        kConfigurationTimeout,
        std::chrono::seconds(app_.config().idle_timeout_sec),
        true,
    });
    websocket_.set_option(
        websocket::stream_base::decorator([](
            websocket::response_type & response)
        {
            response.set(http::field::server, "voxtral-server");
        }));
    websocket_.read_message_max(kMaxBinaryFrameBytes);
    websocket_.async_accept(
        *handshake_request_,
        beast::bind_front_handler(
            &RealtimeSession::on_accept, shared_from_this()));
}

void RealtimeSession::initiate_shutdown() {
    if (closed_) {
        return;
    }
    if (cancellation_) {
        cancellation_->request(CancelReason::Shutdown);
    } else if (lease_) {
        lease_.cancellation()->request(CancelReason::Shutdown);
    }
    input_queue_.cancel();
    outbound_space_.notify_all();
    (void) enqueue_from_io(
        realtime_error_json(
            "shutting_down",
            "The server is shutting down.",
            true),
        OutboundKind::Mandatory);
    request_close_after_drain(websocket::close_code::going_away);
}

void RealtimeSession::on_accept(const boost::system::error_code & error) {
    handshake_request_.reset();
    beast::get_lowest_layer(websocket_).expires_never();
    if (error) {
        app_.logger().debug(
            "websocket_upgrade_failed", {{"detail", error.message()}});
        finish_session();
        return;
    }

    app_.logger().info("realtime_accepted", {
        {"session_id", session_id_},
    });
    std::weak_ptr<RealtimeSession> weak = shared_from_this();
    websocket_.control_callback([
        weak](websocket::frame_type kind, beast::string_view)
    {
        if (kind == websocket::frame_type::ping ||
            kind == websocket::frame_type::pong) {
            if (auto self = weak.lock()) {
                self->touch_idle_timeout();
            }
        }
    });
    arm_configuration_timeout();
    read_next();
}

void RealtimeSession::arm_configuration_timeout() {
    configuration_timer_.expires_after(kConfigurationTimeout);
    configuration_timer_.async_wait([
        self = shared_from_this()](const boost::system::error_code & error)
    {
        if (!error && !self->configured_) {
            self->protocol_failure(
                "invalid_configuration",
                "session.configure was not received within five seconds.");
        }
    });
}

void RealtimeSession::touch_idle_timeout() {
    if (!configured_ || closed_ || input_ended_) {
        return;
    }
    idle_timer_.expires_after(
        std::chrono::seconds(app_.config().idle_timeout_sec));
    idle_timer_.async_wait([
        self = shared_from_this()](const boost::system::error_code & error)
    {
        if (!error && !self->closed_ && !self->input_ended_) {
            self->protocol_failure(
                "idle_timeout",
                "The realtime session exceeded its client-frame idle timeout.",
                CancelReason::Protocol);
        }
    });
}

void RealtimeSession::read_next() {
    if (closed_ || close_after_drain_ || input_ended_ ||
        read_in_progress_ || !pending_audio_.empty()) {
        return;
    }
    read_in_progress_ = true;
    websocket_.async_read(
        read_buffer_,
        beast::bind_front_handler(
            &RealtimeSession::on_read, shared_from_this()));
}

void RealtimeSession::on_read(
    const boost::system::error_code & error,
    std::size_t)
{
    read_in_progress_ = false;
    if (error) {
        if (error == websocket::error::closed) {
            if (cancellation_) {
                cancellation_->request(CancelReason::Disconnect);
            }
            input_queue_.cancel();
            app_.logger().info("realtime_cancelled", {
                {"session_id", session_id_},
                {"reason", "disconnect"},
            });
        } else if (error == websocket::error::message_too_big) {
            protocol_failure(
                "input_overflow",
                "The WebSocket message exceeds the configured frame limit.");
            return;
        } else if (error != boost::asio::error::operation_aborted) {
            if (cancellation_) {
                cancellation_->request(CancelReason::Disconnect);
            }
            input_queue_.cancel();
            app_.logger().debug("websocket_read_failed", {
                {"session_id", session_id_},
                {"detail", error.message()},
            });
        }
        finish_session();
        return;
    }

    touch_idle_timeout();
    const bool text = websocket_.got_text();
    if (text) {
        if (!valid_control_frame_size(read_buffer_.size())) {
            read_buffer_.consume(read_buffer_.size());
            protocol_failure(
                "input_overflow",
                "The JSON control frame exceeds 16 KiB.");
            return;
        }
        const std::string message = beast::buffers_to_string(read_buffer_.data());
        read_buffer_.consume(read_buffer_.size());
        if (!configured_) {
            handle_configuration(message);
        } else {
            handle_control(message);
        }
        return;
    }
    if (!configured_) {
        read_buffer_.consume(read_buffer_.size());
        protocol_failure(
            "invalid_configuration",
            "Binary audio is not accepted before session.configure.");
        return;
    }
    handle_binary();
}

void RealtimeSession::handle_configuration(std::string_view message) {
    const ConfigureParseResult parsed = parse_session_configure(message);
    if (!parsed.ok) {
        protocol_failure(parsed.code, parsed.message);
        return;
    }
    realtime_config_ = parsed.configuration;
    configured_ = true;
    configuration_timer_.cancel();
    touch_idle_timeout();
    if (!enqueue_from_io(
            session_created_json(session_id_, app_.engine().model_name()),
            OutboundKind::Mandatory)) {
        finish_session();
        return;
    }
    start_inference_worker();
    if (!closed_) {
        read_next();
    }
}

void RealtimeSession::handle_control(std::string_view message) {
    const ClientMessageParseResult parsed = parse_client_message(message);
    if (!parsed.ok) {
        protocol_failure(parsed.code, parsed.error_message);
        return;
    }
    switch (parsed.message.type) {
        case ClientMessageType::Ping:
            if (!enqueue_from_io(
                    pong_json(parsed.message.ping_id),
                    OutboundKind::Mandatory)) {
                finish_session();
                return;
            }
            read_next();
            return;
        case ClientMessageType::Finish:
            input_ended_ = true;
            idle_timer_.cancel();
            input_queue_.finish();
            return;
        case ClientMessageType::Cancel:
            input_ended_ = true;
            if (cancellation_) {
                cancellation_->request(CancelReason::Client);
            }
            input_queue_.cancel();
            outbound_space_.notify_all();
            return;
    }
}

void RealtimeSession::handle_binary() {
    const std::size_t byte_count = read_buffer_.size();
    if (!valid_binary_frame_size(byte_count)) {
        read_buffer_.consume(read_buffer_.size());
        protocol_failure(
            "invalid_audio_frame",
            "Binary audio frames must be at most 64 KiB and have an even size.");
        return;
    }
    if (byte_count == 0) {
        read_buffer_.consume(read_buffer_.size());
        read_next();
        return;
    }

    const std::string bytes = beast::buffers_to_string(read_buffer_.data());
    read_buffer_.consume(read_buffer_.size());
    pending_audio_.resize(byte_count / 2u);
    for (std::size_t i = 0; i < pending_audio_.size(); ++i) {
        const auto low = static_cast<unsigned char>(bytes[i * 2u]);
        const auto high = static_cast<unsigned char>(bytes[i * 2u + 1u]);
        pending_audio_[i] = static_cast<std::int16_t>(
            static_cast<std::uint16_t>(low) |
            (static_cast<std::uint16_t>(high) << 8u));
    }
    pending_audio_offset_ = 0;
    received_samples_.fetch_add(
        pending_audio_.size(), std::memory_order_release);
    pump_pending_audio();
}

void RealtimeSession::pump_pending_audio() {
    if (closed_ || pending_audio_.empty()) {
        read_next();
        return;
    }
    while (pending_audio_offset_ < pending_audio_.size()) {
        const std::size_t accepted = input_queue_.push_some(
            pending_audio_.data() + pending_audio_offset_,
            pending_audio_.size() - pending_audio_offset_);
        if (accepted == 0) {
            return;
        }
        pending_audio_offset_ += accepted;
    }
    pending_audio_.clear();
    pending_audio_offset_ = 0;
    read_next();
}

void RealtimeSession::notify_input_space() {
    auto self = shared_from_this();
    boost::asio::post(websocket_.get_executor(), [self] {
        if (!self->closed_) {
            self->pump_pending_audio();
        }
    });
}

void RealtimeSession::start_inference_worker() {
    auto lease_holder =
        std::make_shared<GpuLease>(std::move(lease_));
    cancellation_ = lease_holder->cancellation();
    worker_started_ = true;
    auto self = shared_from_this();
    try {
        if (!app_.start_worker([self, lease_holder] {
                self->inference_worker(lease_holder);
                lease_holder->reset();
            })) {
            worker_started_ = false;
            lease_holder->reset();
            protocol_failure(
                "server_busy",
                "The transcription engine is currently busy.");
        }
    } catch (const std::exception & error) {
        worker_started_ = false;
        lease_holder->reset();
        app_.logger().error("worker_start_failed", {{"detail", error.what()}});
        protocol_failure(
            "internal_error",
            "The server could not start the inference worker.");
    }
}

void RealtimeSession::inference_worker(std::shared_ptr<GpuLease>) {
    const auto started = std::chrono::steady_clock::now();
    try {
        auto stream = app_.engine().create_stream();
        TerminalEvents terminal;
        std::vector<std::int16_t> samples;
        for (;;) {
            const PcmQueue::PopResult pop = input_queue_.pop(samples);
            if (pop == PcmQueue::PopResult::Cancel ||
                cancellation_->requested()) {
                handle_worker_cancellation(*stream);
                return;
            }
            if (pop == PcmQueue::PopResult::Finish) {
                break;
            }
            notify_input_space();
            if (!feed_audio(*stream, samples, terminal)) {
                if (cancellation_->requested()) {
                    handle_worker_cancellation(*stream);
                }
                return;
            }
            if (!observe_lag(*stream)) {
                handle_worker_cancellation(*stream);
                return;
            }
            samples.clear();
        }

        for (;;) {
            if (cancellation_->requested()) {
                handle_worker_cancellation(*stream);
                return;
            }
            const voxtral_status status = stream->finish();
            if (status == VOXTRAL_STATUS_QUEUE_FULL) {
                if (!resume_backpressure(*stream, terminal)) {
                    return;
                }
                continue;
            }
            if (status != VOXTRAL_STATUS_OK) {
                throw EngineError(status, stream->last_error());
            }
            break;
        }
        if (!drain_events(*stream, terminal)) {
            return;
        }
        if (!terminal.final_seen || !terminal.completed_seen ||
            stream->state() != VOXTRAL_STREAM_COMPLETED) {
            throw EngineError(
                VOXTRAL_STATUS_INTERNAL_ERROR,
                "public stream omitted terminal events");
        }

        const std::string final_text = stream->final_text();
        const voxtral_stream_metrics metrics = stream->metrics();
        const double elapsed_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count();
        const double fallback_rtf =
            metrics.audio_duration_ms == 0
                ? 0.0
                : elapsed_ms / metrics.audio_duration_ms;
        if (!enqueue_from_worker(
                final_event_json(
                    terminal.final_sequence,
                    final_text,
                    terminal.final_audio_end_ms),
                OutboundKind::Mandatory) ||
            !enqueue_from_worker(
                completed_event_json(
                    terminal.completed_sequence,
                    metrics,
                    fallback_rtf),
                OutboundKind::Mandatory)) {
            if (!cancellation_->requested()) {
                worker_failure(
                    "internal_error",
                    "The final transcript exceeds the server event limit.",
                    "mandatory realtime output exceeded bounded queue limits");
            }
            return;
        }
        app_.logger().info("realtime_completed", {
            {"session_id", session_id_},
            {"duration_ms", metrics.audio_duration_ms},
            {"processing_ms", elapsed_ms},
            {"tokens", terminal.token_count},
        });
        request_close_after_drain(websocket::close_code::normal);
    } catch (const EngineError & error) {
        worker_failure(
            api_error_code(error.status()),
            safe_status_message(error.status()),
            error.what());
    } catch (const std::bad_alloc &) {
        worker_failure(
            "out_of_memory",
            "The server ran out of memory.",
            "server allocation failed");
    } catch (const std::exception & error) {
        worker_failure(
            "internal_error",
            "The server encountered an internal inference error.",
            error.what());
    }
}

bool RealtimeSession::feed_audio(
    EngineStream & stream,
    const std::vector<std::int16_t> & samples,
    TerminalEvents & terminal)
{
    std::size_t offset = 0;
    while (offset < samples.size()) {
        if (cancellation_->requested()) {
            return false;
        }
        std::size_t consumed = 0;
        const voxtral_status status = stream.feed(
            samples.data() + offset,
            samples.size() - offset,
            &consumed);
        offset += consumed;
        if (!drain_events(stream, terminal)) {
            return false;
        }
        if (status == VOXTRAL_STATUS_QUEUE_FULL) {
            if (!resume_backpressure(stream, terminal)) {
                return false;
            }
            continue;
        }
        if (status != VOXTRAL_STATUS_OK ||
            (consumed == 0 && offset < samples.size())) {
            throw EngineError(
                status == VOXTRAL_STATUS_OK
                    ? VOXTRAL_STATUS_INTERNAL_ERROR
                    : status,
                stream.last_error());
        }
    }
    return true;
}

bool RealtimeSession::resume_backpressure(
    EngineStream & stream,
    TerminalEvents & terminal)
{
    for (;;) {
        if (cancellation_->requested() ||
            !drain_events(stream, terminal)) {
            return false;
        }
        std::size_t consumed = 1;
        const voxtral_status status = stream.feed(nullptr, 0, &consumed);
        if (consumed != 0) {
            throw EngineError(
                VOXTRAL_STATUS_INTERNAL_ERROR,
                "zero-length resume consumed audio");
        }
        if (status == VOXTRAL_STATUS_QUEUE_FULL) {
            continue;
        }
        if (status != VOXTRAL_STATUS_OK) {
            throw EngineError(status, stream.last_error());
        }
        return drain_events(stream, terminal);
    }
}

bool RealtimeSession::drain_events(
    EngineStream & stream,
    TerminalEvents & terminal)
{
    for (;;) {
        if (cancellation_->requested()) {
            return false;
        }
        voxtral_event event{};
        const voxtral_status status = stream.poll(event);
        if (status == VOXTRAL_STATUS_NOT_READY) {
            return true;
        }
        if (status != VOXTRAL_STATUS_OK) {
            throw EngineError(status, stream.last_error());
        }
        switch (event.type) {
            case VOXTRAL_EVENT_TOKEN:
                ++terminal.token_count;
                if (realtime_config_.token_events &&
                    !enqueue_from_worker(
                        token_event_json(event), OutboundKind::Token)) {
                    return false;
                }
                break;
            case VOXTRAL_EVENT_PARTIAL_TEXT:
                if (realtime_config_.partial_events &&
                    !enqueue_from_worker(
                        partial_event_json(event), OutboundKind::Partial)) {
                    return false;
                }
                break;
            case VOXTRAL_EVENT_FINAL_TEXT:
                terminal.final_seen = true;
                terminal.final_sequence = event.sequence;
                terminal.final_audio_end_ms = event.audio_end_ms;
                break;
            case VOXTRAL_EVENT_COMPLETED:
                terminal.completed_seen = true;
                terminal.completed_sequence = event.sequence;
                break;
            case VOXTRAL_EVENT_ERROR:
                if (event.status == VOXTRAL_STATUS_CANCELLED &&
                    cancellation_->requested()) {
                    return false;
                }
                throw EngineError(
                    event.status,
                    std::string(event.text, event.text_length));
            case VOXTRAL_EVENT_NONE:
                throw EngineError(
                    VOXTRAL_STATUS_INTERNAL_ERROR,
                    "public stream returned an empty event");
        }
    }
}

bool RealtimeSession::observe_lag(EngineStream & stream) {
    const voxtral_stream_metrics metrics = stream.metrics();
    const std::uint64_t received =
        received_samples_.load(std::memory_order_acquire);
    const std::uint64_t unaccepted =
        received > metrics.audio_samples_accepted
            ? received - metrics.audio_samples_accepted
            : 0;
    const double input_backlog_ms =
        static_cast<double>(unaccepted) * 1000.0 / 16000.0;
    const double backlog_ms =
        std::max(input_backlog_ms, std::max(0.0, metrics.backlog_final_ms));
    const auto now = std::chrono::steady_clock::now();

    if (backlog_ms >= app_.config().realtime_hard_lag_ms) {
        cancellation_->request(CancelReason::Capacity);
        (void) enqueue_from_worker(
            realtime_error_json(
                "realtime_capacity_exceeded",
                "The server cannot process this stream within the configured "
                "realtime latency limit.",
                true),
            OutboundKind::Mandatory);
        request_close_after_drain(websocket::close_code::policy_error);
        return false;
    }

    if (backlog_ms >= app_.config().realtime_soft_lag_ms) {
        if (!warning_active_ ||
            now - last_warning_ >= std::chrono::seconds(1)) {
            if (!enqueue_from_worker(
                    warning_event_json(backlog_ms),
                    OutboundKind::Mandatory)) {
                return false;
            }
            warning_active_ = true;
            last_warning_ = now;
        }
    } else if (backlog_ms <
               app_.config().realtime_soft_lag_ms * 0.75) {
        warning_active_ = false;
    }
    return true;
}

void RealtimeSession::handle_worker_cancellation(EngineStream & stream) {
    (void) stream.cancel();
    const CancelReason reason = cancellation_->reason();
    if (reason == CancelReason::Client) {
        (void) enqueue_from_worker(
            cancelled_event_json(), OutboundKind::Mandatory);
        app_.logger().info("realtime_cancelled", {
            {"session_id", session_id_},
            {"reason", "client"},
        });
        request_close_after_drain(websocket::close_code::normal);
    } else if (reason == CancelReason::Capacity) {
        request_close_after_drain(websocket::close_code::policy_error);
    }
}

void RealtimeSession::worker_failure(
    std::string_view code,
    std::string_view safe_message,
    std::string_view detail)
{
    if (cancellation_ && cancellation_->requested()) {
        return;
    }
    app_.logger().error("realtime_failed", {
        {"session_id", session_id_},
        {"code", code},
        {"detail", detail},
    });
    (void) enqueue_from_worker(
        realtime_error_json(code, safe_message, true),
        OutboundKind::Mandatory);
    request_close_after_drain(websocket::close_code::internal_error);
}

bool RealtimeSession::enqueue_from_worker(
    std::string payload,
    OutboundKind kind)
{
    std::unique_lock<std::mutex> lock(outbound_mutex_);
    OutboundMessage message{std::move(payload), kind};
    for (;;) {
        if (outbound_closed_ ||
            (cancellation_ && cancellation_->reason() ==
                CancelReason::Disconnect)) {
            return false;
        }
        if (coalesce_partial_locked(message)) {
            lock.unlock();
            boost::asio::post(
                websocket_.get_executor(),
                [self = shared_from_this()] { self->maybe_write(); });
            return true;
        }
        if (has_queue_capacity_locked(message.payload.size())) {
            outbound_bytes_ += message.payload.size();
            outbound_.push_back(std::move(message));
            lock.unlock();
            boost::asio::post(
                websocket_.get_executor(),
                [self = shared_from_this()] { self->maybe_write(); });
            return true;
        }
        if (message.payload.size() > kMaxOutboundBytes) {
            return false;
        }
        outbound_space_.wait(lock);
    }
}

bool RealtimeSession::enqueue_from_io(
    std::string payload,
    OutboundKind kind)
{
    std::lock_guard<std::mutex> lock(outbound_mutex_);
    if (outbound_closed_) {
        return false;
    }
    OutboundMessage message{std::move(payload), kind};
    if (coalesce_partial_locked(message)) {
        boost::asio::post(
            websocket_.get_executor(),
            [self = shared_from_this()] { self->maybe_write(); });
        return true;
    }
    while (!has_queue_capacity_locked(message.payload.size()) &&
           remove_optional_partial_locked()) {
    }
    if (!has_queue_capacity_locked(message.payload.size())) {
        return false;
    }
    outbound_bytes_ += message.payload.size();
    outbound_.push_back(std::move(message));
    boost::asio::post(
        websocket_.get_executor(),
        [self = shared_from_this()] { self->maybe_write(); });
    return true;
}

bool RealtimeSession::has_queue_capacity_locked(std::size_t bytes) const {
    return bytes <= kMaxOutboundBytes &&
           outbound_.size() + (write_in_progress_ ? 1u : 0u) <
               kMaxOutboundMessages &&
           outbound_bytes_ <= kMaxOutboundBytes - bytes;
}

bool RealtimeSession::coalesce_partial_locked(OutboundMessage & message) {
    if (message.kind != OutboundKind::Partial) {
        return false;
    }
    for (std::size_t i = outbound_.size(); i > 0; --i) {
        OutboundMessage & existing = outbound_[i - 1u];
        if (existing.kind != OutboundKind::Partial) {
            continue;
        }
        const std::size_t without_existing =
            outbound_bytes_ - existing.payload.size();
        if (message.payload.size() <= kMaxOutboundBytes - without_existing) {
            outbound_bytes_ =
                without_existing + message.payload.size();
            existing = std::move(message);
            return true;
        }
        return false;
    }
    return false;
}

bool RealtimeSession::remove_optional_partial_locked() {
    for (std::size_t i = outbound_.size(); i > 0; --i) {
        if (outbound_[i - 1u].kind == OutboundKind::Partial) {
            outbound_bytes_ -= outbound_[i - 1u].payload.size();
            outbound_.erase(outbound_.begin() +
                static_cast<std::ptrdiff_t>(i - 1u));
            return true;
        }
    }
    return false;
}

void RealtimeSession::maybe_write() {
    std::unique_lock<std::mutex> lock(outbound_mutex_);
    if (closed_ || write_in_progress_) {
        return;
    }
    if (outbound_.empty()) {
        if (close_after_drain_) {
            lock.unlock();
            close_transport();
        }
        return;
    }
    write_in_progress_ = true;
    active_write_payload_ = std::make_shared<std::string>(
        std::move(outbound_.front().payload));
    outbound_.pop_front();
    websocket_.text(true);
    const auto buffer = boost::asio::buffer(*active_write_payload_);
    lock.unlock();
    websocket_.async_write(
        buffer,
        beast::bind_front_handler(
            &RealtimeSession::on_write, shared_from_this()));
}

void RealtimeSession::on_write(
    const boost::system::error_code & error,
    std::size_t)
{
    {
        std::lock_guard<std::mutex> lock(outbound_mutex_);
        if (active_write_payload_) {
            outbound_bytes_ -= active_write_payload_->size();
            active_write_payload_.reset();
        }
        write_in_progress_ = false;
        outbound_space_.notify_all();
    }
    if (error) {
        if (cancellation_) {
            cancellation_->request(CancelReason::Disconnect);
        }
        input_queue_.cancel();
        finish_session();
        return;
    }
    maybe_write();
}

void RealtimeSession::request_close_after_drain(
    websocket::close_code code)
{
    {
        std::lock_guard<std::mutex> lock(outbound_mutex_);
        bool expected = false;
        if (close_after_drain_.compare_exchange_strong(expected, true)) {
            close_code_ = code;
        }
    }
    boost::asio::post(
        websocket_.get_executor(),
        [self = shared_from_this()] { self->maybe_write(); });
}

void RealtimeSession::close_transport() {
    if (closed_ || close_started_) {
        return;
    }
    close_started_ = true;
    boost::system::error_code ignored;
    configuration_timer_.cancel();
    idle_timer_.cancel();

    if (read_in_progress_) {
        beast::get_lowest_layer(websocket_).socket().shutdown(
            boost::asio::ip::tcp::socket::shutdown_both, ignored);
        beast::get_lowest_layer(websocket_).socket().close(ignored);
        finish_session();
        return;
    }
    websocket::close_code close_code;
    {
        std::lock_guard<std::mutex> lock(outbound_mutex_);
        close_code = close_code_;
    }
    websocket_.async_close(
        close_code,
        [self = shared_from_this()](const boost::system::error_code &) {
            self->finish_session();
        });
}

void RealtimeSession::finish_session() {
    if (closed_.exchange(true)) {
        return;
    }
    boost::system::error_code ignored;
    configuration_timer_.cancel();
    idle_timer_.cancel();
    if (!cancellation_ && lease_) {
        lease_.cancellation()->request(CancelReason::Disconnect);
    }
    input_queue_.cancel();
    lease_.reset();
    {
        std::lock_guard<std::mutex> lock(outbound_mutex_);
        outbound_closed_ = true;
        for (const auto & message : outbound_) {
            outbound_bytes_ -= message.payload.size();
        }
        outbound_.clear();
        outbound_space_.notify_all();
    }
    beast::get_lowest_layer(websocket_).socket().shutdown(
        boost::asio::ip::tcp::socket::shutdown_both, ignored);
    beast::get_lowest_layer(websocket_).socket().close(ignored);
    app_.unregister_session(this);
}

void RealtimeSession::protocol_failure(
    std::string_view code,
    std::string_view message,
    CancelReason reason)
{
    if (closed_) {
        return;
    }
    input_ended_ = true;
    if (cancellation_) {
        cancellation_->request(reason);
    } else if (lease_) {
        lease_.cancellation()->request(reason);
    }
    input_queue_.cancel();
    outbound_space_.notify_all();
    if (!enqueue_from_io(
            realtime_error_json(code, message, true),
            OutboundKind::Mandatory)) {
        finish_session();
        return;
    }
    request_close_after_drain(websocket::close_code::policy_error);
}

} // namespace voxtral::server

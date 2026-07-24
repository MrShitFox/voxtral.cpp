#ifndef VOXTRAL_SERVER_REALTIME_SESSION_H
#define VOXTRAL_SERVER_REALTIME_SESSION_H

#include "bounded-queue.h"
#include "engine.h"
#include "gpu-lease.h"
#include "json-protocol.h"
#include "server-app.h"

#include <boost/asio.hpp>
#include <boost/beast.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/websocket.hpp>

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace voxtral::server {

class RealtimeSession final
    : public ManagedSession,
      public std::enable_shared_from_this<RealtimeSession> {
public:
    using Request = boost::beast::http::request<
        boost::beast::http::string_body>;

    RealtimeSession(
        ServerApp & app,
        boost::beast::tcp_stream stream,
        GpuLease lease,
        std::string session_id);

    void run(Request request);
    void initiate_shutdown() override;

private:
    enum class OutboundKind {
        Token,
        Partial,
        Mandatory,
    };

    struct OutboundMessage {
        std::string payload;
        OutboundKind kind = OutboundKind::Mandatory;
    };

    struct TerminalEvents {
        std::uint64_t final_sequence = 0;
        std::uint64_t final_audio_end_ms = 0;
        std::uint64_t completed_sequence = 0;
        bool final_seen = false;
        bool completed_seen = false;
        std::uint64_t token_count = 0;
    };

    void on_accept(const boost::system::error_code & error);
    void arm_configuration_timeout();
    void touch_idle_timeout();
    void read_next();
    void on_read(
        const boost::system::error_code & error,
        std::size_t bytes);
    void handle_configuration(std::string_view message);
    void handle_control(std::string_view message);
    void handle_binary();
    void pump_pending_audio();
    void notify_input_space();
    void start_inference_worker();
    void inference_worker(std::shared_ptr<GpuLease> lease);

    bool feed_audio(
        EngineStream & stream,
        const std::vector<std::int16_t> & samples,
        TerminalEvents & terminal);
    bool resume_backpressure(
        EngineStream & stream,
        TerminalEvents & terminal);
    bool drain_events(
        EngineStream & stream,
        TerminalEvents & terminal);
    bool observe_lag(EngineStream & stream);
    void handle_worker_cancellation(EngineStream & stream);
    void worker_failure(
        std::string_view code,
        std::string_view safe_message,
        std::string_view detail);

    bool enqueue_from_worker(std::string payload, OutboundKind kind);
    bool enqueue_from_io(std::string payload, OutboundKind kind);
    bool has_queue_capacity_locked(std::size_t bytes) const;
    bool coalesce_partial_locked(OutboundMessage & message);
    bool remove_optional_partial_locked();
    void maybe_write();
    void on_write(
        const boost::system::error_code & error,
        std::size_t bytes);
    void request_close_after_drain(
        boost::beast::websocket::close_code code);
    void close_transport();
    void finish_session();
    void protocol_failure(
        std::string_view code,
        std::string_view message,
        CancelReason reason = CancelReason::Protocol);

    ServerApp & app_;
    boost::beast::websocket::stream<boost::beast::tcp_stream> websocket_;
    boost::beast::flat_buffer read_buffer_;
    std::unique_ptr<Request> handshake_request_;
    GpuLease lease_;
    std::shared_ptr<Cancellation> cancellation_;
    std::string session_id_;
    RealtimeConfiguration realtime_config_;
    PcmQueue input_queue_;
    std::vector<std::int16_t> pending_audio_;
    std::size_t pending_audio_offset_ = 0;
    std::atomic<std::uint64_t> received_samples_{0};

    boost::asio::steady_timer configuration_timer_;
    boost::asio::steady_timer idle_timer_;

    std::mutex outbound_mutex_;
    std::condition_variable outbound_space_;
    std::deque<OutboundMessage> outbound_;
    std::shared_ptr<std::string> active_write_payload_;
    std::size_t outbound_bytes_ = 0;
    bool outbound_closed_ = false;
    bool write_in_progress_ = false;
    std::atomic<bool> close_after_drain_{false};
    boost::beast::websocket::close_code close_code_ =
        boost::beast::websocket::close_code::normal;

    bool configured_ = false;
    bool worker_started_ = false;
    bool input_ended_ = false;
    bool read_in_progress_ = false;
    std::atomic<bool> closed_{false};
    bool close_started_ = false;
    bool warning_active_ = false;
    std::chrono::steady_clock::time_point last_warning_{};
};

} // namespace voxtral::server

#endif

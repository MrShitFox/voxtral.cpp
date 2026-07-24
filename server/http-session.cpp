#include "http-session.h"

#include "auth.h"
#include "json-protocol.h"
#include "realtime-session.h"

#include <boost/beast/websocket.hpp>
#include <boost/json.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <exception>
#include <limits>
#include <string_view>
#include <utility>

namespace voxtral::server {
namespace beast = boost::beast;
namespace http = beast::http;
namespace websocket = beast::websocket;

namespace {

std::string_view header_value(
    const HttpSession::Request & request,
    http::field field)
{
    const auto it = request.find(field);
    if (it == request.end()) {
        return {};
    }
    return {it->value().data(), it->value().size()};
}

std::string_view header_value(
    const HttpSession::Request & request,
    beast::string_view name)
{
    const auto it = request.find(name);
    if (it == request.end()) {
        return {};
    }
    return {it->value().data(), it->value().size()};
}

std::string normalized_content_type(std::string_view content_type) {
    const auto semicolon = content_type.find(';');
    std::string result(content_type.substr(0, semicolon));
    while (!result.empty() &&
           std::isspace(static_cast<unsigned char>(result.back()))) {
        result.pop_back();
    }
    std::size_t start = 0;
    while (start < result.size() &&
           std::isspace(static_cast<unsigned char>(result[start]))) {
        ++start;
    }
    result.erase(0, start);
    std::transform(result.begin(), result.end(), result.begin(), [](char value) {
        return static_cast<char>(
            std::tolower(static_cast<unsigned char>(value)));
    });
    return result;
}

struct TargetParts {
    std::string path;
    std::string query;
};

TargetParts split_target(beast::string_view target) {
    const auto query = target.find('?');
    if (query == beast::string_view::npos) {
        return {std::string(target), {}};
    }
    return {
        std::string(target.substr(0, query)),
        std::string(target.substr(query + 1)),
    };
}

bool parse_response_format(
    std::string_view query,
    bool & text_response)
{
    if (query.empty() || query == "response_format=json") {
        text_response = false;
        return true;
    }
    if (query == "response_format=text") {
        text_response = true;
        return true;
    }
    return false;
}

http::status status_from_integer(int status) {
    return static_cast<http::status>(status);
}

} // namespace

HttpSession::HttpSession(ServerApp & app, tcp::socket socket)
    : app_(app),
      stream_(std::move(socket)) {}

void HttpSession::run() {
    read_request();
}

void HttpSession::initiate_shutdown() {
    if (batch_cancellation_) {
        batch_cancellation_->request(CancelReason::Shutdown);
    }
    close();
}

void HttpSession::read_request() {
    parser_ = std::make_unique<http::request_parser<http::string_body>>();
    parser_->header_limit(kMaxHeaderBytes);
    parser_->body_limit(app_.config().max_upload_bytes);
    stream_.expires_after(
        std::chrono::seconds(app_.config().idle_timeout_sec));
    http::async_read(
        stream_,
        buffer_,
        *parser_,
        beast::bind_front_handler(
            &HttpSession::on_read, shared_from_this()));
}

void HttpSession::on_read(
    const boost::system::error_code & error,
    std::size_t)
{
    if (error == http::error::body_limit) {
        send_json_error(
            http::status::payload_too_large,
            "request_too_large",
            "The request body exceeds the configured upload limit.");
        return;
    }
    if (error) {
        if (error != boost::asio::error::operation_aborted &&
            error != http::error::end_of_stream &&
            error != boost::asio::error::eof) {
            app_.logger().debug("http_read_failed", {{"detail", error.message()}});
        }
        close();
        return;
    }
    route(parser_->release());
    parser_.reset();
}

void HttpSession::route(Request request) {
    const TargetParts target = split_target(request.target());
    if (target.path == "/health") {
        if (request.method() != http::verb::get) {
            send_json_error(
                http::status::method_not_allowed,
                "method_not_allowed",
                "This endpoint only accepts GET.");
            return;
        }
        handle_health(request);
        return;
    }

    if (target.path == "/v1/audio/transcriptions") {
        if (request.method() != http::verb::post) {
            send_json_error(
                http::status::method_not_allowed,
                "method_not_allowed",
                "This endpoint only accepts POST.");
            return;
        }
        handle_batch(std::move(request));
        return;
    }

    if (target.path == "/v1/realtime/transcription") {
        if (!websocket::is_upgrade(request)) {
            send_json_error(
                http::status::bad_request,
                "invalid_upgrade",
                "This endpoint requires a WebSocket upgrade.");
            return;
        }
        handle_realtime_upgrade(std::move(request));
        return;
    }

    send_json_error(
        http::status::not_found,
        "not_found",
        "The requested endpoint does not exist.");
}

void HttpSession::handle_health(const Request & request) {
    const GpuMode mode = app_.engine().leases().mode();
    boost::json::object capabilities;
    capabilities["sample_rate"] = 16000;
    capabilities["channels"] = 1;
    capabilities["audio_format"] = "pcm_s16le";
    capabilities["max_active_streams"] = 1;

    boost::json::object root;
    root["status"] = "ok";
    root["ready"] = !app_.shutting_down();
    const bool busy = mode == GpuMode::Batch || mode == GpuMode::Realtime;
    root["busy"] = busy;
    if (mode == GpuMode::Batch) {
        root["busy_mode"] = "batch";
    } else if (mode == GpuMode::Realtime) {
        root["busy_mode"] = "realtime";
    } else {
        root["busy_mode"] = nullptr;
    }
    root["server_version"] = kServerVersion;
    root["voxtral_version"] = voxtral_version_string();
    root["voxtral_api_version"] = voxtral_api_version();
    root["model"] = app_.engine().model_name();
    root["capabilities"] = std::move(capabilities);

    Response response{http::status::ok, request.version()};
    response.set(http::field::server, "voxtral-server");
    response.set(http::field::content_type, "application/json");
    response.keep_alive(false);
    response.body() = serialize_json(root);
    response.prepare_payload();
    send_response(std::move(response));
}

void HttpSession::handle_batch(Request request) {
    if (!authorized(request)) {
        Response response{http::status::unauthorized, request.version()};
        response.set(http::field::server, "voxtral-server");
        response.set(http::field::content_type, "application/json");
        response.set(http::field::www_authenticate, "Bearer");
        response.keep_alive(false);
        response.body() = error_response_json(
            "unauthorized", "A valid Bearer token is required.");
        response.prepare_payload();
        send_response(std::move(response));
        return;
    }
    if (app_.shutting_down()) {
        send_json_error(
            http::status::service_unavailable,
            "shutting_down",
            "The server is shutting down.");
        return;
    }
    if (!valid_http_body_size(
            request.body().size(), app_.config().max_upload_bytes)) {
        send_json_error(
            http::status::payload_too_large,
            "request_too_large",
            "The request body exceeds the configured upload limit.");
        return;
    }

    const TargetParts target = split_target(request.target());
    bool text_response = false;
    if (!parse_response_format(target.query, text_response)) {
        send_json_error(
            http::status::bad_request,
            "invalid_argument",
            "response_format must be json or text.");
        return;
    }

    const std::string content_type =
        normalized_content_type(header_value(request, http::field::content_type));
    AudioParseResult parsed;
    if (content_type == "audio/wav" || content_type == "audio/x-wav") {
        parsed = parse_pcm16_wav(request.body());
    } else if (content_type == "application/octet-stream") {
        parsed = parse_raw_pcm16(
            request.body(),
            header_value(request, "X-Audio-Format"),
            header_value(request, "X-Sample-Rate"),
            header_value(request, "X-Audio-Channels"));
    } else {
        send_json_error(
            http::status::unsupported_media_type,
            "unsupported_media_type",
            "Only PCM16 WAV and declared raw PCM16 are supported.");
        return;
    }
    if (!parsed.ok) {
        const http::status status =
            parsed.code == "unsupported_audio_format"
                ? http::status::unsupported_media_type
                : http::status::bad_request;
        send_json_error(status, parsed.code, parsed.message);
        return;
    }

    GpuLease lease = app_.engine().leases().try_acquire(GpuMode::Batch);
    if (!lease) {
        send_json_error(
            http::status::service_unavailable,
            app_.shutting_down() ? "shutting_down" : "server_busy",
            app_.shutting_down()
                ? "The server is shutting down."
                : "The transcription engine is currently busy.",
            !app_.shutting_down());
        app_.logger().info("busy_rejection", {{"mode", "batch"}});
        return;
    }
    start_batch(
        std::move(parsed.audio),
        text_response,
        std::move(lease),
        random_id("req_"));
}

void HttpSession::handle_realtime_upgrade(Request request) {
    if (!authorized(request)) {
        Response response{http::status::unauthorized, request.version()};
        response.set(http::field::server, "voxtral-server");
        response.set(http::field::content_type, "application/json");
        response.set(http::field::www_authenticate, "Bearer");
        response.keep_alive(false);
        response.body() = error_response_json(
            "unauthorized", "A valid Bearer token is required.");
        response.prepare_payload();
        send_response(std::move(response));
        return;
    }
    GpuLease lease = app_.engine().leases().try_acquire(GpuMode::Realtime);
    if (!lease) {
        send_json_error(
            http::status::service_unavailable,
            app_.shutting_down() ? "shutting_down" : "server_busy",
            app_.shutting_down()
                ? "The server is shutting down."
                : "The transcription engine is currently busy.",
            !app_.shutting_down());
        app_.logger().info("busy_rejection", {{"mode", "realtime"}});
        return;
    }

    auto realtime = std::make_shared<RealtimeSession>(
        app_,
        std::move(stream_),
        std::move(lease),
        random_id("st_"));
    transferred_ = true;
    app_.unregister_session(this);
    app_.register_session(realtime);
    realtime->run(std::move(request));
}

bool HttpSession::authorized(const Request & request) const {
    return authorize_bearer(
        header_value(request, http::field::authorization),
        app_.config().api_key,
        !app_.config().no_auth);
}

void HttpSession::start_batch(
    AudioData audio,
    bool text_response,
    GpuLease lease,
    std::string request_id)
{
    app_.logger().info("batch_accepted", {
        {"request_id", request_id},
        {"duration_ms", audio.duration_ms},
    });
    batch_running_ = true;
    const std::uint64_t duration_ms = audio.duration_ms;
    auto audio_holder = std::make_shared<AudioData>(std::move(audio));
    auto lease_holder =
        std::make_shared<GpuLease>(std::move(lease));
    batch_cancellation_ = lease_holder->cancellation();
    const auto started = std::chrono::steady_clock::now();
    auto self = shared_from_this();
    try {
        if (!app_.start_worker([
                self,
                audio_holder,
                lease_holder,
                text_response,
                request_id = std::move(request_id),
                duration_ms,
                started]() mutable
            {
                BatchTranscriptionResult result = transcribe_batch(
                    self->app_.engine(),
                    audio_holder->samples,
                    lease_holder->cancellation());
                const auto finished = std::chrono::steady_clock::now();
                const double processing_ms =
                    std::chrono::duration<double, std::milli>(
                        finished - started).count();
                lease_holder->reset();
                boost::asio::post(
                    self->stream_.get_executor(),
                    [self,
                     result = std::move(result),
                     text_response,
                     request_id = std::move(request_id),
                     duration_ms,
                     processing_ms]() mutable
                    {
                        self->on_batch_result(
                            std::move(result),
                            text_response,
                            std::move(request_id),
                            duration_ms,
                            processing_ms);
                    });
            })) {
            batch_running_ = false;
            batch_cancellation_.reset();
            lease_holder->reset();
            send_json_error(
                http::status::service_unavailable,
                "server_busy",
                "The transcription engine is currently busy.",
                true);
            return;
        }
    } catch (const std::exception & error) {
        batch_running_ = false;
        batch_cancellation_.reset();
        lease_holder->reset();
        app_.logger().error("worker_start_failed", {{"detail", error.what()}});
        send_json_error(
            http::status::internal_server_error,
            "internal_error",
            "The server could not start the inference worker.");
        return;
    }
    watch_batch_disconnect();
}

void HttpSession::watch_batch_disconnect() {
    stream_.socket().async_wait(
        tcp::socket::wait_read,
        [self = shared_from_this()](
            const boost::system::error_code & error)
        {
            if (self->batch_done_ ||
                error == boost::asio::error::operation_aborted) {
                return;
            }
            if (self->batch_cancellation_) {
                self->batch_cancellation_->request(CancelReason::Disconnect);
            }
            self->app_.logger().info("batch_cancelled", {
                {"reason", "disconnect"},
            });
            self->close();
        });
}

void HttpSession::on_batch_result(
    BatchTranscriptionResult result,
    bool text_response,
    std::string request_id,
    std::uint64_t duration_ms,
    double processing_ms)
{
    batch_running_ = false;
    batch_done_ = true;
    batch_cancellation_.reset();
    boost::system::error_code ignored;
    stream_.socket().cancel(ignored);
    if (closed_ || app_.shutting_down() || result.cancelled) {
        if (result.cancelled && !closed_) {
            close();
        }
        return;
    }
    if (!result.ok) {
        app_.logger().error("batch_failed", {
            {"request_id", request_id},
            {"status", voxtral_status_string(result.status)},
            {"detail", result.detail},
        });
        send_json_error(
            status_from_integer(api_http_status(result.status)),
            api_error_code(result.status),
            safe_status_message(result.status));
        return;
    }

    app_.logger().info("batch_completed", {
        {"request_id", request_id},
        {"duration_ms", duration_ms},
        {"processing_ms", processing_ms},
        {"tokens", result.token_count},
    });

    Response response{http::status::ok, 11};
    response.set(http::field::server, "voxtral-server");
    response.keep_alive(false);
    if (text_response) {
        response.set(
            http::field::content_type,
            "text/plain; charset=utf-8");
        response.body() = std::move(result.text);
    } else {
        const double realtime_factor =
            duration_ms == 0 ? 0.0 : processing_ms / duration_ms;
        boost::json::object root;
        root["id"] = random_id("tr_");
        root["object"] = "audio.transcription";
        root["model"] = app_.engine().model_name();
        root["text"] = std::move(result.text);
        root["duration_ms"] = duration_ms;
        root["processing_ms"] = processing_ms;
        root["realtime_factor"] = realtime_factor;
        response.set(http::field::content_type, "application/json");
        response.body() = serialize_json(root);
    }
    response.prepare_payload();
    send_response(std::move(response));
}

void HttpSession::send_json_error(
    http::status status,
    std::string_view code,
    std::string_view message,
    bool retry_after)
{
    Response response{status, 11};
    response.set(http::field::server, "voxtral-server");
    response.set(http::field::content_type, "application/json");
    if (retry_after) {
        response.set(http::field::retry_after, "1");
    }
    response.keep_alive(false);
    response.body() = error_response_json(code, message);
    response.prepare_payload();
    send_response(std::move(response));
}

void HttpSession::send_response(Response response) {
    if (closed_) {
        return;
    }
    stream_.expires_after(
        std::chrono::seconds(app_.config().idle_timeout_sec));
    response_ = std::make_shared<Response>(std::move(response));
    http::async_write(
        stream_,
        *response_,
        [self = shared_from_this()](
            const boost::system::error_code &,
            std::size_t)
        {
            self->response_.reset();
            self->close();
        });
}

void HttpSession::close() {
    if (closed_ || transferred_) {
        return;
    }
    closed_ = true;
    boost::system::error_code ignored;
    stream_.socket().shutdown(tcp::socket::shutdown_both, ignored);
    stream_.socket().close(ignored);
    app_.unregister_session(this);
}

} // namespace voxtral::server

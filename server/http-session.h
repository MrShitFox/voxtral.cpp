#ifndef VOXTRAL_SERVER_HTTP_SESSION_H
#define VOXTRAL_SERVER_HTTP_SESSION_H

#include "batch-transcription.h"
#include "server-app.h"
#include "wav-reader.h"

#include <boost/asio.hpp>
#include <boost/beast.hpp>
#include <boost/beast/http.hpp>

#include <chrono>
#include <memory>
#include <string>

namespace voxtral::server {

class HttpSession final
    : public ManagedSession,
      public std::enable_shared_from_this<HttpSession> {
public:
    using tcp = boost::asio::ip::tcp;
    using Request = boost::beast::http::request<
        boost::beast::http::string_body>;
    using Response = boost::beast::http::response<
        boost::beast::http::string_body>;

    HttpSession(ServerApp & app, tcp::socket socket);
    void run();
    void initiate_shutdown() override;

private:
    void read_request();
    void on_read(const boost::system::error_code & error, std::size_t bytes);
    void route(Request request);
    void handle_health(const Request & request);
    void handle_batch(Request request);
    void handle_realtime_upgrade(Request request);

    bool authorized(const Request & request) const;
    void start_batch(
        AudioData audio,
        bool text_response,
        GpuLease lease,
        std::string request_id);
    void watch_batch_disconnect();
    void on_batch_result(
        BatchTranscriptionResult result,
        bool text_response,
        std::string request_id,
        std::uint64_t duration_ms,
        double processing_ms);

    void send_json_error(
        boost::beast::http::status status,
        std::string_view code,
        std::string_view message,
        bool retry_after = false);
    void send_response(Response response);
    void close();

    ServerApp & app_;
    boost::beast::tcp_stream stream_;
    boost::beast::flat_buffer buffer_;
    std::unique_ptr<boost::beast::http::request_parser<
        boost::beast::http::string_body>> parser_;
    std::shared_ptr<Response> response_;
    std::shared_ptr<Cancellation> batch_cancellation_;
    bool batch_running_ = false;
    bool batch_done_ = false;
    bool closed_ = false;
    bool transferred_ = false;
};

} // namespace voxtral::server

#endif

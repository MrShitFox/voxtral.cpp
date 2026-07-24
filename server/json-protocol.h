#ifndef VOXTRAL_SERVER_JSON_PROTOCOL_H
#define VOXTRAL_SERVER_JSON_PROTOCOL_H

#include <voxtral-stream.h>

#include <boost/json.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace voxtral::server {

constexpr std::size_t kMaxHeaderBytes = 32u * 1024u;
constexpr std::size_t kMaxControlFrameBytes = 16u * 1024u;
constexpr std::size_t kMaxBinaryFrameBytes = 64u * 1024u;
constexpr std::size_t kMaxOutboundMessages = 256u;
constexpr std::size_t kMaxOutboundBytes = 4u * 1024u * 1024u;

struct RealtimeConfiguration {
    bool token_events = false;
    bool partial_events = true;
};

struct ConfigureParseResult {
    bool ok = false;
    RealtimeConfiguration configuration;
    std::string code;
    std::string message;
};

enum class ClientMessageType {
    Finish,
    Cancel,
    Ping,
};

struct ClientMessage {
    ClientMessageType type = ClientMessageType::Finish;
    std::string ping_id;
};

struct ClientMessageParseResult {
    bool ok = false;
    ClientMessage message;
    std::string code;
    std::string error_message;
};

bool valid_http_body_size(std::uint64_t size, std::uint64_t maximum);
bool valid_control_frame_size(std::size_t size);
bool valid_binary_frame_size(std::size_t size);

ConfigureParseResult parse_session_configure(std::string_view json);
ClientMessageParseResult parse_client_message(std::string_view json);

std::string serialize_json(const boost::json::value & value);
boost::json::object error_object(
    std::string_view code,
    std::string_view message);
std::string error_response_json(
    std::string_view code,
    std::string_view message);
std::string realtime_error_json(
    std::string_view code,
    std::string_view message,
    bool fatal);
std::string random_id(std::string_view prefix);

std::string session_created_json(
    std::string_view session_id,
    std::string_view model_name);
std::string pong_json(std::string_view id);
std::string token_event_json(const voxtral_event & event);
std::string partial_event_json(const voxtral_event & event);
std::string final_event_json(
    std::uint64_t sequence,
    std::string_view text,
    std::uint64_t audio_end_ms);
std::string completed_event_json(
    std::uint64_t sequence,
    const voxtral_stream_metrics & metrics,
    double fallback_realtime_factor);
std::string warning_event_json(double backlog_ms);
std::string cancelled_event_json();

} // namespace voxtral::server

#endif

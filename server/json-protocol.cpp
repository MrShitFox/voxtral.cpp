#include "json-protocol.h"

#include <array>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>

namespace voxtral::server {
namespace {

boost::json::object audio_description() {
    boost::json::object audio;
    audio["format"] = "pcm_s16le";
    audio["sample_rate"] = 16000;
    audio["channels"] = 1;
    return audio;
}

ConfigureParseResult configure_failure(
    std::string code,
    std::string message)
{
    ConfigureParseResult result;
    result.code = std::move(code);
    result.message = std::move(message);
    return result;
}

ClientMessageParseResult message_failure(
    std::string code,
    std::string message)
{
    ClientMessageParseResult result;
    result.code = std::move(code);
    result.error_message = std::move(message);
    return result;
}

bool read_bool(
    const boost::json::object & object,
    std::string_view key,
    bool default_value,
    bool & output)
{
    const auto * value = object.if_contains(key);
    if (!value) {
        output = default_value;
        return true;
    }
    if (!value->is_bool()) {
        return false;
    }
    output = value->as_bool();
    return true;
}

} // namespace

bool valid_http_body_size(std::uint64_t size, std::uint64_t maximum) {
    return maximum > 0 && size <= maximum;
}

bool valid_control_frame_size(std::size_t size) {
    return size > 0 && size <= kMaxControlFrameBytes;
}

bool valid_binary_frame_size(std::size_t size) {
    return size <= kMaxBinaryFrameBytes && (size & 1u) == 0;
}

ConfigureParseResult parse_session_configure(std::string_view json) {
    boost::system::error_code error;
    const boost::json::value value = boost::json::parse(json, error);
    if (error || !value.is_object()) {
        return configure_failure("invalid_json", "The control message is not valid JSON.");
    }
    const auto & root = value.as_object();
    const auto * type = root.if_contains("type");
    if (!type || !type->is_string() ||
        type->as_string() != "session.configure") {
        return configure_failure(
            "invalid_configuration",
            "The first message must be session.configure.");
    }
    const auto * audio_value = root.if_contains("audio");
    if (!audio_value || !audio_value->is_object()) {
        return configure_failure(
            "invalid_configuration",
            "session.configure must contain an audio object.");
    }
    const auto & audio = audio_value->as_object();
    const auto * format = audio.if_contains("format");
    const auto * sample_rate = audio.if_contains("sample_rate");
    const auto * channels = audio.if_contains("channels");
    if (!format || !format->is_string() ||
        format->as_string() != "pcm_s16le" ||
        !sample_rate || !sample_rate->is_int64() ||
        sample_rate->as_int64() != 16000 ||
        !channels || !channels->is_int64() ||
        channels->as_int64() != 1) {
        return configure_failure(
            "unsupported_audio_format",
            "Realtime audio must be pcm_s16le, 16000 Hz, and one channel.");
    }
    if (const auto * language = root.if_contains("language")) {
        if (!language->is_string() || language->as_string() != "auto") {
            return configure_failure(
                "invalid_configuration",
                "language must be absent or set to auto.");
        }
    }

    RealtimeConfiguration configuration;
    if (const auto * events_value = root.if_contains("events")) {
        if (!events_value->is_object()) {
            return configure_failure(
                "invalid_configuration",
                "events must be a JSON object.");
        }
        const auto & events = events_value->as_object();
        if (!read_bool(events, "token", false, configuration.token_events) ||
            !read_bool(events, "partial", true, configuration.partial_events)) {
            return configure_failure(
                "invalid_configuration",
                "events.token and events.partial must be booleans.");
        }
    }

    ConfigureParseResult result;
    result.ok = true;
    result.configuration = configuration;
    return result;
}

ClientMessageParseResult parse_client_message(std::string_view json) {
    boost::system::error_code error;
    const boost::json::value value = boost::json::parse(json, error);
    if (error || !value.is_object()) {
        return message_failure("invalid_json", "The control message is not valid JSON.");
    }
    const auto & root = value.as_object();
    const auto * type_value = root.if_contains("type");
    if (!type_value || !type_value->is_string()) {
        return message_failure(
            "unsupported_message",
            "Unsupported client message type.");
    }
    const auto & type = type_value->as_string();
    ClientMessageParseResult result;
    if (type == "input_audio.end") {
        result.ok = true;
        result.message.type = ClientMessageType::Finish;
        return result;
    }
    if (type == "session.cancel") {
        result.ok = true;
        result.message.type = ClientMessageType::Cancel;
        return result;
    }
    if (type == "ping") {
        const auto * id = root.if_contains("id");
        if (!id || !id->is_string()) {
            return message_failure(
                "invalid_configuration",
                "ping.id must be a string.");
        }
        result.ok = true;
        result.message.type = ClientMessageType::Ping;
        result.message.ping_id.assign(id->as_string().data(), id->as_string().size());
        return result;
    }
    return message_failure(
        "unsupported_message",
        "Unsupported client message type.");
}

std::string serialize_json(const boost::json::value & value) {
    return boost::json::serialize(value);
}

boost::json::object error_object(
    std::string_view code,
    std::string_view message)
{
    boost::json::object error;
    error["code"] = code;
    error["message"] = message;
    return error;
}

std::string error_response_json(
    std::string_view code,
    std::string_view message)
{
    boost::json::object root;
    root["error"] = error_object(code, message);
    return serialize_json(root);
}

std::string realtime_error_json(
    std::string_view code,
    std::string_view message,
    bool fatal)
{
    boost::json::object root;
    root["type"] = "error";
    root["code"] = code;
    root["message"] = message;
    root["fatal"] = fatal;
    return serialize_json(root);
}

std::string random_id(std::string_view prefix) {
    std::array<unsigned char, 16> bytes{};
    bool filled = false;
    std::ifstream random("/dev/urandom", std::ios::binary);
    if (random) {
        random.read(
            reinterpret_cast<char *>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
        filled = random.good();
    }
    if (!filled) {
        std::random_device device;
        for (auto & byte : bytes) {
            byte = static_cast<unsigned char>(device());
        }
    }

    std::ostringstream output;
    output << prefix << std::hex << std::setfill('0');
    for (const unsigned char byte : bytes) {
        output << std::setw(2) << static_cast<unsigned int>(byte);
    }
    return output.str();
}

std::string session_created_json(
    std::string_view session_id,
    std::string_view model_name)
{
    boost::json::object root;
    root["type"] = "session.created";
    root["session_id"] = session_id;
    root["protocol_version"] = 1;
    root["model"] = model_name;
    root["audio"] = audio_description();
    return serialize_json(root);
}

std::string pong_json(std::string_view id) {
    boost::json::object root;
    root["type"] = "pong";
    root["id"] = id;
    return serialize_json(root);
}

std::string token_event_json(const voxtral_event & event) {
    boost::json::object root;
    root["type"] = "transcript.token";
    root["sequence"] = event.sequence;
    root["token_id"] = event.token_id;
    root["text"] = std::string_view(event.text, event.text_length);
    root["audio_end_ms"] = event.audio_end_ms;
    return serialize_json(root);
}

std::string partial_event_json(const voxtral_event & event) {
    boost::json::object root;
    root["type"] = "transcript.partial";
    root["sequence"] = event.sequence;
    root["text"] = std::string_view(event.text, event.text_length);
    root["audio_end_ms"] = event.audio_end_ms;
    return serialize_json(root);
}

std::string final_event_json(
    std::uint64_t sequence,
    std::string_view text,
    std::uint64_t audio_end_ms)
{
    boost::json::object root;
    root["type"] = "transcript.final";
    root["sequence"] = sequence;
    root["text"] = text;
    root["audio_end_ms"] = audio_end_ms;
    return serialize_json(root);
}

std::string completed_event_json(
    std::uint64_t sequence,
    const voxtral_stream_metrics & metrics,
    double fallback_realtime_factor)
{
    boost::json::object processing;
    processing["realtime_factor"] =
        metrics.pipeline_rtf > 0.0 ? metrics.pipeline_rtf :
                                     fallback_realtime_factor;
    processing["backlog_final_ms"] = metrics.backlog_final_ms;
    processing["backlog_slope_ms_per_s"] = metrics.backlog_slope_ms_per_s;

    boost::json::object root;
    root["type"] = "session.completed";
    root["sequence"] = sequence;
    root["audio_duration_ms"] = metrics.audio_duration_ms;
    root["processing"] = std::move(processing);
    return serialize_json(root);
}

std::string warning_event_json(double backlog_ms) {
    boost::json::object root;
    root["type"] = "session.warning";
    root["code"] = "processing_lag";
    root["backlog_ms"] = backlog_ms;
    root["message"] = "Transcription is falling behind real-time audio.";
    return serialize_json(root);
}

std::string cancelled_event_json() {
    boost::json::object root;
    root["type"] = "session.cancelled";
    return serialize_json(root);
}

} // namespace voxtral::server

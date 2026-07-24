#include "auth.h"
#include "gpu-lease.h"
#include "json-protocol.h"
#include "server-config.h"
#include "wav-reader.h"

#include <boost/json.hpp>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace voxtral::server;

namespace {

void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void append_u16(std::string & bytes, std::uint16_t value) {
    bytes.push_back(static_cast<char>(value & 0xffu));
    bytes.push_back(static_cast<char>((value >> 8u) & 0xffu));
}

void append_u32(std::string & bytes, std::uint32_t value) {
    bytes.push_back(static_cast<char>(value & 0xffu));
    bytes.push_back(static_cast<char>((value >> 8u) & 0xffu));
    bytes.push_back(static_cast<char>((value >> 16u) & 0xffu));
    bytes.push_back(static_cast<char>((value >> 24u) & 0xffu));
}

std::string make_wav(
    std::uint16_t format = 1,
    std::uint16_t channels = 1,
    std::uint32_t sample_rate = 16000)
{
    std::string chunks;
    chunks += "JUNK";
    append_u32(chunks, 3);
    chunks += "abc";
    chunks.push_back('\0');

    chunks += "fmt ";
    append_u32(chunks, 16);
    append_u16(chunks, format);
    append_u16(chunks, channels);
    append_u32(chunks, sample_rate);
    append_u32(chunks, sample_rate * channels * 2u);
    append_u16(chunks, static_cast<std::uint16_t>(channels * 2u));
    append_u16(chunks, 16);

    chunks += "data";
    append_u32(chunks, 4);
    append_u16(chunks, 0x1234);
    append_u16(chunks, 0xfedc);

    std::string wav = "RIFF";
    append_u32(wav, static_cast<std::uint32_t>(chunks.size() + 4u));
    wav += "WAVE";
    wav += chunks;
    return wav;
}

void test_config() {
    Environment environment{
        {"VOXTRAL_SERVER_MODEL", "/env/model.gguf"},
        {"VOXTRAL_SERVER_PORT", "9000"},
        {"VOXTRAL_SERVER_API_KEY", "environment-secret"},
    };
    const ParsedConfig parsed = parse_config(
        {"voxtral-server", "--model", "/cli/model.gguf", "--port", "8123"},
        environment);
    require(parsed.config.model_path == "/cli/model.gguf", "CLI model precedence");
    require(parsed.config.port == 8123, "CLI port precedence");
    require(parsed.config.api_key == "environment-secret", "environment key");
    require(parsed.config.max_upload_bytes == 512u * kMebibyte, "upload default");
    require(parsed.config.realtime_buffer_ms == 1000, "buffer default");

    const ParsedConfig loopback = parse_config(
        {"voxtral-server", "--model", "model.gguf", "--no-auth"},
        {});
    require(loopback.config.no_auth, "loopback no-auth");

    bool rejected = false;
    try {
        (void) parse_config(
            {"voxtral-server", "--model", "model.gguf", "--listen", "0.0.0.0",
             "--no-auth"},
            {});
    } catch (const ConfigError &) {
        rejected = true;
    }
    require(rejected, "non-loopback no-auth must fail");

    const ParsedConfig insecure = parse_config(
        {"voxtral-server", "--model", "model.gguf", "--listen", "0.0.0.0",
         "--no-auth", "--allow-insecure-no-auth"},
        {});
    require(insecure.config.allow_insecure_no_auth, "explicit insecure override");

    const std::filesystem::path key_path =
        std::filesystem::temp_directory_path() / random_id("voxtral-key-");
    {
        std::ofstream key_file(key_path, std::ios::binary);
        key_file << "file-secret\r\n";
    }
    Environment file_environment{
        {"VOXTRAL_SERVER_MODEL", "model.gguf"},
        {"VOXTRAL_SERVER_API_KEY_FILE", key_path.string()},
    };
    const ParsedConfig from_file =
        parse_config({"voxtral-server"}, file_environment);
    std::filesystem::remove(key_path);
    require(from_file.config.api_key == "file-secret", "API key file");
}

void test_auth() {
    require(
        authorize_bearer("Bearer secret", "secret", true),
        "valid bearer");
    require(
        !authorize_bearer("bearer secret", "secret", true),
        "scheme is case-sensitive");
    require(
        !authorize_bearer("Bearer ", "secret", true),
        "empty token");
    require(
        !authorize_bearer("Bearer secrex", "secret", true),
        "wrong token");
    require(
        authorize_bearer({}, {}, false),
        "disabled authentication");
    require(constant_time_equal("abc", "abc"), "constant-time equality");
    require(!constant_time_equal("abc", "abcd"), "length mismatch");
}

void test_wav_and_raw() {
    const AudioParseResult wav = parse_pcm16_wav(make_wav());
    require(wav.ok, "valid chunked WAV");
    require(wav.audio.samples.size() == 2, "WAV sample count");
    require(wav.audio.samples[0] == 0x1234, "WAV little endian");
    require(wav.audio.samples[1] == static_cast<std::int16_t>(0xfedc), "WAV signed");

    require(
        !parse_pcm16_wav(make_wav(3)).ok,
        "compressed WAV rejected");
    require(
        !parse_pcm16_wav(make_wav(1, 2)).ok,
        "stereo WAV rejected");
    std::string truncated = make_wav();
    truncated.pop_back();
    require(!parse_pcm16_wav(truncated).ok, "truncated WAV rejected");

    const std::string pcm{"\x01\x00\xff\xff", 4};
    const AudioParseResult raw =
        parse_raw_pcm16(pcm, "pcm_s16le", "16000", "1");
    require(raw.ok && raw.audio.samples.size() == 2, "valid raw PCM");
    require(
        !parse_raw_pcm16(std::string_view(pcm).substr(0, 3),
                         "pcm_s16le", "16000", "1").ok,
        "odd raw PCM rejected");
    require(
        !parse_raw_pcm16(pcm, "pcm_s16le", "48000", "1").ok,
        "wrong raw rate rejected");
}

void test_json_protocol() {
    const std::string configure = R"({
        "type":"session.configure",
        "audio":{"format":"pcm_s16le","sample_rate":16000,"channels":1},
        "events":{"token":true,"partial":false}
    })";
    const ConfigureParseResult parsed = parse_session_configure(configure);
    require(parsed.ok, "valid session.configure");
    require(parsed.configuration.token_events, "token event config");
    require(!parsed.configuration.partial_events, "partial event config");
    require(
        !parse_session_configure(R"({"type":"ping","id":"1"})").ok,
        "configure must be first");

    const ClientMessageParseResult ping =
        parse_client_message(R"({"type":"ping","id":"42"})");
    require(ping.ok && ping.message.ping_id == "42", "application ping");
    require(
        !parse_client_message(R"({"type":"unknown"})").ok,
        "unsupported control message");

    const auto error = boost::json::parse(
        error_response_json("server_busy", "busy")).as_object();
    require(error.contains("error"), "HTTP error envelope");
    const std::string first = random_id("req_");
    const std::string second = random_id("req_");
    require(first.rfind("req_", 0) == 0 && first != second, "random IDs");
}

void test_lease_and_limits() {
    GpuLeaseManager manager;
    GpuLease batch = manager.try_acquire(GpuMode::Batch);
    require(static_cast<bool>(batch), "batch lease acquired");
    require(manager.busy_mode_name() == "batch", "batch mode visible");
    require(
        !manager.try_acquire(GpuMode::Realtime),
        "second lease rejected immediately");
    const auto cancellation = batch.cancellation();
    batch.reset();
    require(!manager.busy(), "lease released");
    GpuLease realtime = manager.try_acquire(GpuMode::Realtime);
    require(static_cast<bool>(realtime), "realtime lease after release");
    manager.begin_shutdown();
    require(cancellation->reason() == CancelReason::None, "old token isolated");
    require(
        realtime.cancellation()->reason() == CancelReason::Shutdown,
        "active lease cancelled on shutdown");
    require(
        !manager.try_acquire(GpuMode::Batch),
        "lease rejected during shutdown");

    require(valid_http_body_size(10, 10), "body at limit");
    require(!valid_http_body_size(11, 10), "body over limit");
    require(valid_control_frame_size(kMaxControlFrameBytes), "control at limit");
    require(!valid_control_frame_size(kMaxControlFrameBytes + 1), "control over");
    require(valid_binary_frame_size(0), "zero binary frame is a no-op");
    require(valid_binary_frame_size(kMaxBinaryFrameBytes), "binary at limit");
    require(!valid_binary_frame_size(3), "odd binary frame");
}

} // namespace

int main() {
    try {
        test_config();
        test_auth();
        test_wav_and_raw();
        test_json_protocol();
        test_lease_and_limits();
        std::cout << "voxtral_server_unit: PASS\n";
        return EXIT_SUCCESS;
    } catch (const std::exception & error) {
        std::cerr << "voxtral_server_unit: FAIL: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}

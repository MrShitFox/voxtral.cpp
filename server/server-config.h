#ifndef VOXTRAL_SERVER_CONFIG_H
#define VOXTRAL_SERVER_CONFIG_H

#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace voxtral::server {

constexpr std::uint64_t kMebibyte = 1024u * 1024u;

struct ServerConfig {
    std::string model_path;
    std::string listen_address = "127.0.0.1";
    std::uint16_t port = 8080;
    std::string api_key;
    std::string api_key_file;
    bool no_auth = false;
    bool allow_insecure_no_auth = false;
    std::uint64_t max_upload_bytes = 512u * kMebibyte;
    std::uint32_t realtime_soft_lag_ms = 1000;
    std::uint32_t realtime_hard_lag_ms = 5000;
    std::uint32_t realtime_buffer_ms = 1000;
    std::uint32_t idle_timeout_sec = 60;
};

enum class ConfigAction {
    Run,
    Help,
    Version,
};

struct ParsedConfig {
    ServerConfig config;
    ConfigAction action = ConfigAction::Run;
};

class ConfigError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

using Environment = std::map<std::string, std::string>;

Environment process_environment();
ParsedConfig parse_config(
    const std::vector<std::string> & arguments,
    const Environment & environment);
bool is_loopback_address(const std::string & address);
std::string server_usage(const std::string & executable);

} // namespace voxtral::server

#endif

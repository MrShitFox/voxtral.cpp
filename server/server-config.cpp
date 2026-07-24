#include "server-config.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>

namespace voxtral::server {
namespace {

constexpr std::uint64_t kMaxApiKeyBytes = 4096;

const char * const kEnvironmentNames[] = {
    "VOXTRAL_SERVER_MODEL",
    "VOXTRAL_SERVER_LISTEN",
    "VOXTRAL_SERVER_PORT",
    "VOXTRAL_SERVER_API_KEY",
    "VOXTRAL_SERVER_API_KEY_FILE",
    "VOXTRAL_SERVER_MAX_UPLOAD_MIB",
    "VOXTRAL_SERVER_REALTIME_SOFT_LAG_MS",
    "VOXTRAL_SERVER_REALTIME_HARD_LAG_MS",
    "VOXTRAL_SERVER_REALTIME_BUFFER_MS",
    "VOXTRAL_SERVER_IDLE_TIMEOUT_SEC",
};

std::uint64_t parse_unsigned(
    const std::string & value,
    const std::string & name,
    std::uint64_t minimum,
    std::uint64_t maximum)
{
    if (value.empty() || value.front() == '-' || value.front() == '+') {
        throw ConfigError(name + " must be an unsigned integer");
    }

    errno = 0;
    char * end = nullptr;
    const unsigned long long parsed = std::strtoull(value.c_str(), &end, 10);
    if (errno == ERANGE || end == value.c_str() || *end != '\0' ||
        parsed < minimum || parsed > maximum) {
        std::ostringstream message;
        message << name << " must be between " << minimum << " and " << maximum;
        throw ConfigError(message.str());
    }
    return static_cast<std::uint64_t>(parsed);
}

std::string read_api_key_file(const std::string & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw ConfigError("cannot open API key file");
    }

    std::string key(
        (std::istreambuf_iterator<char>(input)),
        std::istreambuf_iterator<char>());
    if (input.bad() || key.size() > kMaxApiKeyBytes) {
        throw ConfigError("API key file is invalid or too large");
    }

    while (!key.empty() && (key.back() == '\n' || key.back() == '\r')) {
        key.pop_back();
    }
    if (key.empty() || key.find('\0') != std::string::npos) {
        throw ConfigError("API key file must contain a non-empty text token");
    }
    return key;
}

void apply_environment(ServerConfig & config, const Environment & environment) {
    const auto get = [&environment](const char * name) -> const std::string * {
        const auto it = environment.find(name);
        return it == environment.end() ? nullptr : &it->second;
    };

    if (const auto * value = get("VOXTRAL_SERVER_MODEL")) {
        config.model_path = *value;
    }
    if (const auto * value = get("VOXTRAL_SERVER_LISTEN")) {
        config.listen_address = *value;
    }
    if (const auto * value = get("VOXTRAL_SERVER_PORT")) {
        config.port = static_cast<std::uint16_t>(
            parse_unsigned(*value, "VOXTRAL_SERVER_PORT", 1, 65535));
    }
    if (const auto * value = get("VOXTRAL_SERVER_API_KEY")) {
        config.api_key = *value;
        config.api_key_file.clear();
    } else if (const auto * value = get("VOXTRAL_SERVER_API_KEY_FILE")) {
        config.api_key_file = *value;
    }
    if (const auto * value = get("VOXTRAL_SERVER_MAX_UPLOAD_MIB")) {
        const std::uint64_t mib = parse_unsigned(
            *value,
            "VOXTRAL_SERVER_MAX_UPLOAD_MIB",
            1,
            std::numeric_limits<std::uint64_t>::max() / kMebibyte);
        config.max_upload_bytes = mib * kMebibyte;
    }
    if (const auto * value = get("VOXTRAL_SERVER_REALTIME_SOFT_LAG_MS")) {
        config.realtime_soft_lag_ms = static_cast<std::uint32_t>(
            parse_unsigned(
                *value,
                "VOXTRAL_SERVER_REALTIME_SOFT_LAG_MS",
                1,
                std::numeric_limits<std::uint32_t>::max()));
    }
    if (const auto * value = get("VOXTRAL_SERVER_REALTIME_HARD_LAG_MS")) {
        config.realtime_hard_lag_ms = static_cast<std::uint32_t>(
            parse_unsigned(
                *value,
                "VOXTRAL_SERVER_REALTIME_HARD_LAG_MS",
                1,
                std::numeric_limits<std::uint32_t>::max()));
    }
    if (const auto * value = get("VOXTRAL_SERVER_REALTIME_BUFFER_MS")) {
        config.realtime_buffer_ms = static_cast<std::uint32_t>(
            parse_unsigned(
                *value,
                "VOXTRAL_SERVER_REALTIME_BUFFER_MS",
                1,
                std::numeric_limits<std::uint32_t>::max()));
    }
    if (const auto * value = get("VOXTRAL_SERVER_IDLE_TIMEOUT_SEC")) {
        config.idle_timeout_sec = static_cast<std::uint32_t>(
            parse_unsigned(
                *value,
                "VOXTRAL_SERVER_IDLE_TIMEOUT_SEC",
                1,
                std::numeric_limits<std::uint32_t>::max()));
    }
}

std::string option_value(
    const std::vector<std::string> & arguments,
    std::size_t & index,
    const std::string & option)
{
    if (index + 1 >= arguments.size()) {
        throw ConfigError(option + " requires a value");
    }
    ++index;
    if (arguments[index].empty()) {
        throw ConfigError(option + " requires a non-empty value");
    }
    return arguments[index];
}

void validate_config(ServerConfig & config) {
    if (config.model_path.empty()) {
        throw ConfigError(
            "model path is required (--model or VOXTRAL_SERVER_MODEL)");
    }
    if (config.listen_address.empty()) {
        throw ConfigError("listen address must not be empty");
    }
    if (config.realtime_soft_lag_ms > config.realtime_hard_lag_ms) {
        throw ConfigError("realtime soft lag must not exceed hard lag");
    }

    if (config.no_auth) {
        config.api_key.clear();
        config.api_key_file.clear();
        if (!is_loopback_address(config.listen_address) &&
            !config.allow_insecure_no_auth) {
            throw ConfigError(
                "--no-auth is only allowed on a loopback bind; "
                "use --allow-insecure-no-auth to override");
        }
        return;
    }

    if (!config.api_key_file.empty()) {
        config.api_key = read_api_key_file(config.api_key_file);
    }
    if (config.api_key.empty()) {
        throw ConfigError(
            "authentication is required; configure an API key or use --no-auth");
    }
    if (config.api_key.size() > kMaxApiKeyBytes ||
        config.api_key.find('\0') != std::string::npos) {
        throw ConfigError("API key is invalid or too large");
    }
}

} // namespace

Environment process_environment() {
    Environment result;
    for (const char * name : kEnvironmentNames) {
        if (const char * value = std::getenv(name)) {
            result.emplace(name, value);
        }
    }
    return result;
}

ParsedConfig parse_config(
    const std::vector<std::string> & arguments,
    const Environment & environment)
{
    ParsedConfig parsed;
    apply_environment(parsed.config, environment);

    for (std::size_t i = 1; i < arguments.size(); ++i) {
        const std::string & argument = arguments[i];
        if (argument == "--help") {
            parsed.action = ConfigAction::Help;
            return parsed;
        }
        if (argument == "--version") {
            parsed.action = ConfigAction::Version;
            return parsed;
        }
        if (argument == "--model") {
            parsed.config.model_path = option_value(arguments, i, argument);
        } else if (argument == "--listen") {
            parsed.config.listen_address = option_value(arguments, i, argument);
        } else if (argument == "--port") {
            parsed.config.port = static_cast<std::uint16_t>(parse_unsigned(
                option_value(arguments, i, argument), argument, 1, 65535));
        } else if (argument == "--api-key-file") {
            parsed.config.api_key_file = option_value(arguments, i, argument);
            parsed.config.api_key.clear();
            parsed.config.no_auth = false;
        } else if (argument == "--no-auth") {
            parsed.config.no_auth = true;
        } else if (argument == "--allow-insecure-no-auth") {
            parsed.config.allow_insecure_no_auth = true;
        } else if (argument == "--max-upload-mib") {
            const auto mib = parse_unsigned(
                option_value(arguments, i, argument),
                argument,
                1,
                std::numeric_limits<std::uint64_t>::max() / kMebibyte);
            parsed.config.max_upload_bytes = mib * kMebibyte;
        } else if (argument == "--realtime-soft-lag-ms") {
            parsed.config.realtime_soft_lag_ms =
                static_cast<std::uint32_t>(parse_unsigned(
                    option_value(arguments, i, argument),
                    argument,
                    1,
                    std::numeric_limits<std::uint32_t>::max()));
        } else if (argument == "--realtime-hard-lag-ms") {
            parsed.config.realtime_hard_lag_ms =
                static_cast<std::uint32_t>(parse_unsigned(
                    option_value(arguments, i, argument),
                    argument,
                    1,
                    std::numeric_limits<std::uint32_t>::max()));
        } else if (argument == "--realtime-buffer-ms") {
            parsed.config.realtime_buffer_ms =
                static_cast<std::uint32_t>(parse_unsigned(
                    option_value(arguments, i, argument),
                    argument,
                    1,
                    std::numeric_limits<std::uint32_t>::max()));
        } else if (argument == "--idle-timeout-sec") {
            parsed.config.idle_timeout_sec =
                static_cast<std::uint32_t>(parse_unsigned(
                    option_value(arguments, i, argument),
                    argument,
                    1,
                    std::numeric_limits<std::uint32_t>::max()));
        } else {
            throw ConfigError("unknown option: " + argument);
        }
    }

    validate_config(parsed.config);
    return parsed;
}

bool is_loopback_address(const std::string & address) {
    if (address == "localhost" || address == "::1") {
        return true;
    }
    constexpr const char prefix[] = "127.";
    return address.compare(0, sizeof(prefix) - 1, prefix) == 0;
}

std::string server_usage(const std::string & executable) {
    std::ostringstream output;
    output
        << "Usage: " << executable << " --model PATH [options]\n\n"
        << "Options:\n"
        << "  --model PATH                    Realtime GGUF model\n"
        << "  --listen ADDRESS                Bind address (default 127.0.0.1)\n"
        << "  --port PORT                     TCP port (default 8080)\n"
        << "  --api-key-file PATH             Read Bearer token from a file\n"
        << "  --no-auth                       Disable auth on loopback only\n"
        << "  --allow-insecure-no-auth        Permit unauthenticated non-loopback bind\n"
        << "  --max-upload-mib N              Batch body limit (default 512)\n"
        << "  --realtime-soft-lag-ms N        Lag warning threshold (default 1000)\n"
        << "  --realtime-hard-lag-ms N        Fatal lag threshold (default 5000)\n"
        << "  --realtime-buffer-ms N          PCM queue capacity (default 1000)\n"
        << "  --idle-timeout-sec N            Client-frame idle timeout (default 60)\n"
        << "  --help                          Show this help\n"
        << "  --version                       Show server version\n\n"
        << "Environment equivalents use the VOXTRAL_SERVER_* names documented in "
           "docs/server-api.md.\n";
    return output.str();
}

} // namespace voxtral::server

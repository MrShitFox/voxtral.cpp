#include "logger.h"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace voxtral::server {
namespace {

const char * level_name(LogLevel level) {
    switch (level) {
        case LogLevel::Error:
            return "error";
        case LogLevel::Warn:
            return "warn";
        case LogLevel::Info:
            return "info";
        case LogLevel::Debug:
            return "debug";
    }
    return "info";
}

std::string timestamp() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm utc{};
#if defined(_WIN32)
    gmtime_s(&utc, &time);
#else
    gmtime_r(&time, &utc);
#endif
    std::ostringstream output;
    output << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    return output.str();
}

} // namespace

Logger::Logger(LogLevel threshold) : threshold_(threshold) {}

void Logger::log(
    LogLevel level,
    std::string_view event,
    boost::json::object fields)
{
    if (static_cast<int>(level) > static_cast<int>(threshold_)) {
        return;
    }
    boost::json::object root;
    root["timestamp"] = timestamp();
    root["level"] = level_name(level);
    root["event"] = event;
    for (auto & field : fields) {
        root[field.key()] = std::move(field.value());
    }
    std::lock_guard<std::mutex> lock(mutex_);
    std::cerr << boost::json::serialize(root) << '\n';
}

void Logger::error(std::string_view event, boost::json::object fields) {
    log(LogLevel::Error, event, std::move(fields));
}

void Logger::warn(std::string_view event, boost::json::object fields) {
    log(LogLevel::Warn, event, std::move(fields));
}

void Logger::info(std::string_view event, boost::json::object fields) {
    log(LogLevel::Info, event, std::move(fields));
}

void Logger::debug(std::string_view event, boost::json::object fields) {
    log(LogLevel::Debug, event, std::move(fields));
}

} // namespace voxtral::server

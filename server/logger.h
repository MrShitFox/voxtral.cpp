#ifndef VOXTRAL_SERVER_LOGGER_H
#define VOXTRAL_SERVER_LOGGER_H

#include <boost/json.hpp>

#include <mutex>
#include <string_view>

namespace voxtral::server {

enum class LogLevel {
    Error = 0,
    Warn = 1,
    Info = 2,
    Debug = 3,
};

class Logger {
public:
    explicit Logger(LogLevel threshold = LogLevel::Info);

    void log(
        LogLevel level,
        std::string_view event,
        boost::json::object fields = {});
    void error(std::string_view event, boost::json::object fields = {});
    void warn(std::string_view event, boost::json::object fields = {});
    void info(std::string_view event, boost::json::object fields = {});
    void debug(std::string_view event, boost::json::object fields = {});

private:
    LogLevel threshold_;
    std::mutex mutex_;
};

} // namespace voxtral::server

#endif

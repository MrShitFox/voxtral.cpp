#include "engine.h"
#include "logger.h"
#include "server-app.h"
#include "server-config.h"

#include <boost/asio.hpp>
#include <boost/json.hpp>

#include <exception>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    using namespace voxtral::server;

    Logger logger;
    try {
        std::vector<std::string> arguments;
        arguments.reserve(static_cast<std::size_t>(argc));
        for (int i = 0; i < argc; ++i) {
            arguments.emplace_back(argv[i]);
        }
        const ParsedConfig parsed =
            parse_config(arguments, process_environment());
        if (parsed.action == ConfigAction::Help) {
            std::cout << server_usage(
                arguments.empty() ? "voxtral-server" : arguments.front());
            return 0;
        }
        if (parsed.action == ConfigAction::Version) {
            std::cout << "voxtral-server " << kServerVersion
                      << " (libvoxtral " << voxtral_version_string()
                      << ", API " << voxtral_api_version() << ")\n";
            return 0;
        }

        logger.info("startup", {
            {"server_version", kServerVersion},
            {"voxtral_version", voxtral_version_string()},
            {"voxtral_api_version", voxtral_api_version()},
        });
        Engine engine(parsed.config.model_path, logger);
        boost::asio::io_context io_context(1);
        ServerApp app(io_context, engine, parsed.config, logger);
        app.start();
        io_context.run();
        app.join_worker();
        return 0;
    } catch (const ConfigError & error) {
        logger.error("startup_error", {
            {"code", "invalid_configuration"},
            {"message", error.what()},
        });
        return 2;
    } catch (const EngineError & error) {
        logger.error("startup_error", {
            {"code", api_error_code(error.status())},
            {"status", voxtral_status_string(error.status())},
            {"message", error.what()},
        });
        return 1;
    } catch (const std::exception & error) {
        logger.error("startup_error", {
            {"code", "internal_error"},
            {"message", error.what()},
        });
        return 1;
    }
}

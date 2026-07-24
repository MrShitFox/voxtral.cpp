#include "server-app.h"

#include "http-session.h"

#include <boost/asio/ip/address.hpp>

#include <csignal>
#include <exception>
#include <utility>
#include <vector>

namespace voxtral::server {

WorkerSlot::WorkerSlot(Completion completion)
    : completion_(std::move(completion)) {}

WorkerSlot::~WorkerSlot() {
    join();
}

bool WorkerSlot::start(std::function<void()> task) {
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true)) {
        return false;
    }
    if (thread_.joinable()) {
        thread_.join();
    }
    try {
        thread_ = std::thread([this, task = std::move(task)]() mutable {
            try {
                task();
            } catch (...) {
                // Session workers contain their own exception mapping. Keep the
                // slot releasable even if a programming error escapes.
            }
            running_.store(false, std::memory_order_release);
            completion_();
        });
    } catch (...) {
        running_.store(false, std::memory_order_release);
        throw;
    }
    return true;
}

bool WorkerSlot::running() const noexcept {
    return running_.load(std::memory_order_acquire);
}

void WorkerSlot::join() {
    if (thread_.joinable()) {
        thread_.join();
    }
}

ServerApp::ServerApp(
    boost::asio::io_context & io_context,
    Engine & engine,
    const ServerConfig & config,
    Logger & logger)
    : io_context_(io_context),
      engine_(engine),
      config_(config),
      logger_(logger),
      acceptor_(io_context),
      signals_(io_context, SIGINT, SIGTERM),
      shutdown_timer_(io_context),
      worker_([this] {
          boost::asio::post(io_context_, [this] { on_worker_finished(); });
      })
{
    boost::system::error_code error;
    const std::string bind_address =
        config.listen_address == "localhost"
            ? std::string("127.0.0.1")
            : config.listen_address;
    const auto address = boost::asio::ip::make_address(bind_address, error);
    if (error) {
        throw std::runtime_error("invalid listen address");
    }
    const tcp::endpoint endpoint(address, config.port);
    acceptor_.open(endpoint.protocol(), error);
    if (error) {
        throw std::runtime_error("failed to open the TCP acceptor");
    }
    acceptor_.set_option(boost::asio::socket_base::reuse_address(true), error);
    if (error) {
        throw std::runtime_error("failed to configure the TCP acceptor");
    }
    acceptor_.bind(endpoint, error);
    if (error) {
        throw std::runtime_error("failed to bind the listen address");
    }
    acceptor_.listen(boost::asio::socket_base::max_listen_connections, error);
    if (error) {
        throw std::runtime_error("failed to listen on the configured address");
    }
}

ServerApp::~ServerApp() {
    engine_.leases().begin_shutdown();
    boost::system::error_code ignored;
    acceptor_.close(ignored);
    signals_.cancel(ignored);
    shutdown_timer_.cancel();
    worker_.join();
}

void ServerApp::start() {
    if (started_) {
        return;
    }
    started_ = true;
    signals_.async_wait([this](
        const boost::system::error_code & error,
        int signal_number)
    {
        if (!error) {
            initiate_shutdown(signal_number);
        }
    });
    accept_next();
    logger_.info("server_listening", {
        {"listen", config_.listen_address},
        {"port", config_.port},
        {"auth", config_.no_auth ? "disabled" : "bearer"},
    });
    if (config_.no_auth && !is_loopback_address(config_.listen_address)) {
        logger_.warn("insecure_no_auth", {
            {"message", "authentication is disabled on a non-loopback bind"},
        });
    }
}

void ServerApp::initiate_shutdown(int signal_number) {
    if (shutting_down_) {
        return;
    }
    shutting_down_ = true;
    logger_.info("shutdown_started", {{"signal", signal_number}});
    engine_.leases().begin_shutdown();

    boost::system::error_code ignored;
    acceptor_.cancel(ignored);
    acceptor_.close(ignored);
    signals_.cancel(ignored);

    std::vector<std::shared_ptr<ManagedSession>> live_sessions;
    live_sessions.reserve(sessions_.size());
    for (auto it = sessions_.begin(); it != sessions_.end();) {
        if (auto session = it->second.lock()) {
            live_sessions.push_back(std::move(session));
            ++it;
        } else {
            it = sessions_.erase(it);
        }
    }
    for (const auto & session : live_sessions) {
        session->initiate_shutdown();
    }

    shutdown_timer_.expires_after(kShutdownTimeout);
    shutdown_timer_.async_wait([this](const boost::system::error_code & error) {
        if (!error) {
            force_shutdown();
        }
    });
    maybe_finish_shutdown();
}

bool ServerApp::start_worker(std::function<void()> task) {
    return worker_.start(std::move(task));
}

void ServerApp::join_worker() {
    worker_.join();
}

void ServerApp::register_session(
    const std::shared_ptr<ManagedSession> & session)
{
    sessions_[session.get()] = session;
}

void ServerApp::unregister_session(ManagedSession * session) {
    sessions_.erase(session);
    maybe_finish_shutdown();
}

boost::asio::io_context & ServerApp::io_context() noexcept {
    return io_context_;
}

Engine & ServerApp::engine() noexcept {
    return engine_;
}

const ServerConfig & ServerApp::config() const noexcept {
    return config_;
}

Logger & ServerApp::logger() noexcept {
    return logger_;
}

bool ServerApp::shutting_down() const noexcept {
    return shutting_down_;
}

void ServerApp::accept_next() {
    if (shutting_down_) {
        return;
    }
    acceptor_.async_accept(
        boost::asio::make_strand(io_context_),
        [this](
            const boost::system::error_code & error,
            tcp::socket socket)
        {
            if (!error) {
                auto session =
                    std::make_shared<HttpSession>(*this, std::move(socket));
                register_session(session);
                session->run();
            } else if (!shutting_down_ &&
                       error != boost::asio::error::operation_aborted) {
                logger_.error("accept_failed", {{"detail", error.message()}});
            }
            accept_next();
        });
}

void ServerApp::on_worker_finished() {
    maybe_finish_shutdown();
}

void ServerApp::maybe_finish_shutdown() {
    if (!shutting_down_) {
        return;
    }
    for (auto it = sessions_.begin(); it != sessions_.end();) {
        if (it->second.expired()) {
            it = sessions_.erase(it);
        } else {
            ++it;
        }
    }
    if (!worker_.running() && sessions_.empty()) {
        shutdown_timer_.cancel();
        logger_.info("shutdown_complete");
        io_context_.stop();
    }
}

void ServerApp::force_shutdown() {
    logger_.warn("shutdown_timeout", {
        {"timeout_seconds", kShutdownTimeout.count()},
    });
    std::vector<std::shared_ptr<ManagedSession>> live_sessions;
    for (auto & entry : sessions_) {
        if (auto session = entry.second.lock()) {
            live_sessions.push_back(std::move(session));
        }
    }
    for (const auto & session : live_sessions) {
        session->initiate_shutdown();
    }
    io_context_.stop();
}

} // namespace voxtral::server

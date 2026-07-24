#ifndef VOXTRAL_SERVER_APP_H
#define VOXTRAL_SERVER_APP_H

#include "engine.h"
#include "logger.h"
#include "server-config.h"

#include <boost/asio.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <functional>
#include <memory>
#include <thread>
#include <unordered_map>

namespace voxtral::server {

constexpr const char * kServerVersion = "1.0.0";
constexpr std::chrono::seconds kShutdownTimeout{10};

class ManagedSession {
public:
    virtual ~ManagedSession() = default;
    virtual void initiate_shutdown() = 0;
};

class WorkerSlot {
public:
    using Completion = std::function<void()>;

    explicit WorkerSlot(Completion completion);
    WorkerSlot(const WorkerSlot &) = delete;
    WorkerSlot & operator=(const WorkerSlot &) = delete;
    ~WorkerSlot();

    bool start(std::function<void()> task);
    bool running() const noexcept;
    void join();

private:
    Completion completion_;
    std::thread thread_;
    std::atomic<bool> running_{false};
};

class ServerApp {
public:
    using tcp = boost::asio::ip::tcp;

    ServerApp(
        boost::asio::io_context & io_context,
        Engine & engine,
        const ServerConfig & config,
        Logger & logger);
    ServerApp(const ServerApp &) = delete;
    ServerApp & operator=(const ServerApp &) = delete;
    ~ServerApp();

    void start();
    void initiate_shutdown(int signal_number = 0);
    bool start_worker(std::function<void()> task);
    void join_worker();

    void register_session(const std::shared_ptr<ManagedSession> & session);
    void unregister_session(ManagedSession * session);

    boost::asio::io_context & io_context() noexcept;
    Engine & engine() noexcept;
    const ServerConfig & config() const noexcept;
    Logger & logger() noexcept;
    bool shutting_down() const noexcept;

private:
    void accept_next();
    void on_worker_finished();
    void maybe_finish_shutdown();
    void force_shutdown();

    boost::asio::io_context & io_context_;
    Engine & engine_;
    const ServerConfig & config_;
    Logger & logger_;
    tcp::acceptor acceptor_;
    boost::asio::signal_set signals_;
    boost::asio::steady_timer shutdown_timer_;
    WorkerSlot worker_;
    std::unordered_map<ManagedSession *, std::weak_ptr<ManagedSession>> sessions_;
    bool started_ = false;
    bool shutting_down_ = false;
};

} // namespace voxtral::server

#endif

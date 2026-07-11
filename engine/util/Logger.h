/*
 * Engine-layer logger.
 * No Qt dependency, no direct stderr output — all logs go through callback.
 * Python bindings register a callback to forward logs to Python logging.
 */
#pragma once

#include <chrono>
#include <ctime>
#include <functional>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>

namespace rehab {

class Logger {
 public:
  /// Callback: level in {"INFO","WARN","ERROR","PERF"}, message is log body.
  using LogCallback = std::function<void(const std::string& level,
                                         const std::string& message)>;

  static void info(const std::string& msg) { log("INFO", msg); }
  static void warn(const std::string& msg) { log("WARN", msg); }
  static void error(const std::string& msg) { log("ERROR", msg); }
  static void performance(const std::string& msg) { log("PERF", msg); }

  /// Set the log callback. Thread-safe. Pass nullptr to disable.
  static void setCallback(LogCallback callback) {
    std::lock_guard<std::mutex> lock(stateMutex());
    callbackSlot() = std::move(callback);
  }

 private:
  static void log(const char* level, const std::string& msg) {
    LogCallback cb;
    {
      std::lock_guard<std::mutex> lock(stateMutex());
      cb = callbackSlot();
    }
    if (cb) {
      cb(level, msg);
    }
  }

  static std::mutex& stateMutex() {
    static std::mutex mtx;
    return mtx;
  }

  static LogCallback& callbackSlot() {
    static LogCallback cb;
    return cb;
  }
};

}  // namespace rehab
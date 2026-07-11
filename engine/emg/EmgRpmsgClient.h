#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>

#include "engine/common/Config.h"
#include "engine/emg/EmgTypes.h"

namespace rehab {

class EmgRpmsgClient {
 public:
  using FeatureCallback = std::function<void(const EmgFeatureFrame&)>;
  using StatusCallback = std::function<void(const std::string&)>;

  EmgRpmsgClient() = default;
  ~EmgRpmsgClient();

  void configure(const EmgConfig& config);
  void setOnFeature(FeatureCallback callback);
  void setOnStatus(StatusCallback callback);

  bool connect();
  void close();
  bool isConnected() const { return connected_.load(); }

  bool sendConfig();
  bool sendRawChunk(const EmgRawChunk& chunk);

 private:
  void readLoop();
  void emitStatus(const std::string& status);
  bool writePacket(const void* data, std::size_t size);
  bool parseFeaturePacket(const uint8_t* data, std::size_t size, EmgFeatureFrame* outFrame);

  EmgConfig config_;
  std::atomic<bool> connected_{false};
  std::atomic<bool> readerRunning_{false};
  std::thread reader_;
  int ctrlFd_{-1};
  int dataFd_{-1};
  std::mutex ioMutex_;
  std::mutex callbackMutex_;
  FeatureCallback featureCallback_;
  StatusCallback statusCallback_;
};

}  // namespace rehab

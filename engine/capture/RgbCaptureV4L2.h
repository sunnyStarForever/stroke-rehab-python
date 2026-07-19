#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>

#include "engine/common/Config.h"
#include "engine/common/FrameEnvelope.h"
#include "engine/sync/TimestampNormalizer.h"

namespace rehab {

class RgbCaptureV4L2 {
 public:
  using FrameCallback = std::function<void(FrameEnvelope)>;
  using StatusCallback = std::function<void(const std::string&)>;

  ~RgbCaptureV4L2();

  bool start(const DeviceConfig& config, FrameCallback callback);
  void stop();
  void setOnStatus(StatusCallback callback);
  bool isRunning() const { return running_.load(); }

 private:
  void run();
  void runFallback();
  void emitStatus(const std::string& status);

  DeviceConfig config_;
  FrameCallback callback_;
  StatusCallback statusCallback_;
  std::mutex statusMutex_;
  std::atomic<bool> running_{false};
  TimestampNormalizer tsNormalizer_;
  uint64_t lastV4l2SyncTsNs_{0};
  std::thread worker_;
};

}  // namespace rehab

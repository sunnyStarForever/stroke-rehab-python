#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <thread>

#include "engine/common/Config.h"
#include "engine/common/FrameEnvelope.h"
#include "engine/sync/TimestampNormalizer.h"

namespace rehab {

class DepthCaptureOpenNI {
 public:
  using FrameCallback = std::function<void(FrameEnvelope)>;

  ~DepthCaptureOpenNI();

  bool start(const DeviceConfig& config, FrameCallback callback);
  void stop();
  void setQueueDropCounter(std::function<uint64_t()> counter);

  bool isRunning() const { return running_.load(); }
  bool hardwareD2CActive() const { return hardwareD2CActive_.load(); }

 private:
  void run();
  void runFallback();

  DeviceConfig config_;
  FrameCallback callback_;
  std::function<uint64_t()> queueDropCounter_;

  std::atomic<bool> running_{false};
  std::atomic<bool> hardwareD2CActive_{false};
  TimestampNormalizer tsNormalizer_;
  std::thread worker_;
};

}  // namespace rehab

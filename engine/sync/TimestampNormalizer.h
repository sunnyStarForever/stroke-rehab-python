/*
 * 模块作用：
 * 本文件声明时间戳归一化器。采集层把不同设备来源的时间信息统一写入
 * FrameEnvelope，后续同步层只需要读取 hostTsNs/deviceTsUs。
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>

#include "engine/common/FrameEnvelope.h"

namespace rehab {

/*
 * TimestampNormalizer
 * 职责：
 * 将采集线程测得的 host_ts_ns 和设备原始 device_ts_us 写入 FrameEnvelope。
 * host_ts_ns 用于软件同步，device_ts_us 用于调试设备侧抖动。
 */
class TimestampNormalizer {
 public:
  explicit TimestampNormalizer(std::size_t windowSize = 60);

  void reset();
  void stampHostFallback(FrameEnvelope& frame, uint64_t arrivalTsNs,
                         uint64_t deviceTsUs, const std::string& reason);
  void stampNativeMonotonic(FrameEnvelope& frame, uint64_t arrivalTsNs,
                            uint64_t mappedDeviceTsNs,
                            uint64_t deviceTsUs = 0);
  void stampDeviceMicroseconds(FrameEnvelope& frame, uint64_t arrivalTsNs,
                               uint64_t deviceTsUs);

 private:
  void resetMapping(const std::string& reason);

  std::size_t windowSize_{60};
  std::deque<int64_t> offsetsNs_;
  int64_t estimatedOffsetNs_{0};
  uint64_t lastDeviceTsUs_{0};
  uint64_t lastArrivalTsNs_{0};
  uint64_t resetCount_{0};
  bool initialized_{false};
  std::string lastResetReason_;
};

}  // namespace rehab

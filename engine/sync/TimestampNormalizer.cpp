#include "engine/sync/TimestampNormalizer.h"

#include <algorithm>
#include <vector>

namespace rehab {

namespace {
constexpr uint64_t kDiscontinuityNs = 1000000000ULL;
constexpr int64_t kMaxOffsetStepNs = 200000LL;
}

TimestampNormalizer::TimestampNormalizer(std::size_t windowSize)
    : windowSize_(std::max<std::size_t>(5, windowSize)) {}

void TimestampNormalizer::reset() {
  offsetsNs_.clear();
  estimatedOffsetNs_ = 0;
  lastDeviceTsUs_ = 0;
  lastArrivalTsNs_ = 0;
  initialized_ = false;
  lastResetReason_.clear();
}

void TimestampNormalizer::resetMapping(const std::string& reason) {
  reset();
  ++resetCount_;
  lastResetReason_ = reason;
}

void TimestampNormalizer::stampHostFallback(FrameEnvelope& frame,
                                            uint64_t arrivalTsNs,
                                            uint64_t deviceTsUs,
                                            const std::string& reason) {
  frame.hostTsNs = arrivalTsNs;
  frame.arrivalTsNs = arrivalTsNs;
  frame.syncTsNs = arrivalTsNs;
  frame.deviceTsUs = deviceTsUs;
  frame.deviceTimeUnit = "us";
  frame.clockQuality = "host_fallback";
  frame.clockReason = reason;
  frame.clockResetCount = resetCount_;
}

void TimestampNormalizer::stampNativeMonotonic(FrameEnvelope& frame,
                                               uint64_t arrivalTsNs,
                                               uint64_t mappedDeviceTsNs,
                                               uint64_t deviceTsUs) {
  if (mappedDeviceTsNs == 0) {
    stampHostFallback(frame, arrivalTsNs, deviceTsUs,
                      "invalid_native_monotonic");
    return;
  }
  frame.hostTsNs = arrivalTsNs;
  frame.arrivalTsNs = arrivalTsNs;
  frame.syncTsNs = mappedDeviceTsNs;
  frame.deviceTsUs = deviceTsUs;
  frame.deviceTimeUnit = "us";
  frame.clockQuality = "native_monotonic";
  frame.clockReason.clear();
  frame.clockResetCount = resetCount_;
}

void TimestampNormalizer::stampDeviceMicroseconds(FrameEnvelope& frame,
                                                  uint64_t arrivalTsNs,
                                                  uint64_t deviceTsUs) {
  if (deviceTsUs == 0) {
    stampHostFallback(frame, arrivalTsNs, 0, "missing_device_timestamp");
    return;
  }

  if (lastDeviceTsUs_ > 0) {
    const bool backwards = deviceTsUs <= lastDeviceTsUs_;
    const uint64_t deviceDeltaNs = backwards ? 0 :
        (deviceTsUs - lastDeviceTsUs_) * 1000ULL;
    const uint64_t arrivalDeltaNs = arrivalTsNs > lastArrivalTsNs_ ?
        arrivalTsNs - lastArrivalTsNs_ : 0;
    const uint64_t deltaError = deviceDeltaNs > arrivalDeltaNs ?
        deviceDeltaNs - arrivalDeltaNs : arrivalDeltaNs - deviceDeltaNs;
    if (backwards || deltaError > kDiscontinuityNs) {
      resetMapping(backwards ? "device_timestamp_backwards" :
                               "device_timestamp_jump");
    }
  }

  const int64_t observedOffset = static_cast<int64_t>(arrivalTsNs) -
      static_cast<int64_t>(deviceTsUs * 1000ULL);
  offsetsNs_.push_back(observedOffset);
  while (offsetsNs_.size() > windowSize_) offsetsNs_.pop_front();
  std::vector<int64_t> sorted(offsetsNs_.begin(), offsetsNs_.end());
  std::sort(sorted.begin(), sorted.end());
  const int64_t candidate = sorted[(sorted.size() - 1) / 10];
  if (!initialized_) {
    estimatedOffsetNs_ = candidate;
    initialized_ = true;
  } else {
    const int64_t error = candidate - estimatedOffsetNs_;
    estimatedOffsetNs_ += std::max(-kMaxOffsetStepNs,
                                   std::min(kMaxOffsetStepNs, error));
  }

  const int64_t mapped = static_cast<int64_t>(deviceTsUs * 1000ULL) +
                         estimatedOffsetNs_;
  frame.hostTsNs = arrivalTsNs;
  frame.arrivalTsNs = arrivalTsNs;
  frame.syncTsNs = mapped > 0 ? static_cast<uint64_t>(mapped) : arrivalTsNs;
  frame.deviceTsUs = deviceTsUs;
  frame.deviceTimeUnit = "us";
  frame.clockQuality = mapped > 0 ? "normalized_device" : "host_fallback";
  frame.clockReason = mapped > 0 ? lastResetReason_ : "invalid_mapped_time";
  frame.clockResetCount = resetCount_;
  lastDeviceTsUs_ = deviceTsUs;
  lastArrivalTsNs_ = arrivalTsNs;
}

}  // namespace rehab

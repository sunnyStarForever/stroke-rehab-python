#pragma once

#include <array>
#include <string>
#include <unordered_map>

#include "engine/pose/Rehab22Types.h"
#include "engine/pose/SkeletonTypes.h"

namespace rehab {

class EMASkeletonFilter {
 public:
  struct Options {
    float alphaGood{0.65f};
    float alphaLowConfidence{0.35f};
    float alphaRecovered{0.45f};
    float alphaInvalid{0.0f};

    float maxZJumpM{0.45f};
    float maxJointSpeedMps{2.5f};

    bool holdLastWhenInvalid{true};
    int maxHoldFrames{5};
  };

  EMASkeletonFilter();
  explicit EMASkeletonFilter(Options options);

  void setOptions(const Options& options) { options_ = options; }
  void reset(const std::string& reason);

  Skeleton3D filter(const Skeleton3D& raw, double dtSeconds);
  Rehab22Skeleton3D filter(
      const Rehab22Skeleton3D& raw,
      double dtSeconds,
      std::array<JointEmaDebugInfo, kRehab22JointCount>* debug = nullptr);

 private:
  Options options_;
  Rehab22Skeleton3D last_;
  bool hasLast_{false};
  std::unordered_map<std::string, int> invalidHoldCount_;
  uint64_t lastLogNs_{0};
};

}  // namespace rehab

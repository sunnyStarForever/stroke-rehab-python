#include "engine/pose/EMASkeletonFilter.h"

#include <algorithm>
#include <cmath>
#include <sstream>

#include "engine/pose/JointNameMapper.h"
#include "engine/util/Logger.h"
#include "engine/common/Timestamp.h"

namespace rehab {

namespace {

float distance3d(const Keypoint3D& a, const Keypoint3D& b) {
  const float dx = a.x - b.x;
  const float dy = a.y - b.y;
  const float dz = a.z - b.z;
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

Keypoint3D lerpJoint(const Keypoint3D& current,
                     const Keypoint3D& previous,
                     float alpha) {
  Keypoint3D out = current;
  out.x = alpha * current.x + (1.0f - alpha) * previous.x;
  out.y = alpha * current.y + (1.0f - alpha) * previous.y;
  out.z = alpha * current.z + (1.0f - alpha) * previous.z;
  out.score = alpha * current.score + (1.0f - alpha) * previous.score;
  out.valid = true;
  return out;
}

}  // namespace

EMASkeletonFilter::EMASkeletonFilter() : options_() {}

EMASkeletonFilter::EMASkeletonFilter(Options options)
    : options_(options) {}

void EMASkeletonFilter::reset(const std::string& reason) {
  last_ = {};
  hasLast_ = false;
  invalidHoldCount_.clear();
  Logger::info("[EMA] reset reason=" + reason);
}

Skeleton3D EMASkeletonFilter::filter(const Skeleton3D& raw,
                                     double /*dtSeconds*/) {
  return raw;
}

Rehab22Skeleton3D EMASkeletonFilter::filter(
    const Rehab22Skeleton3D& raw,
    double dtSeconds,
    std::array<JointEmaDebugInfo, kRehab22JointCount>* debug) {
  Rehab22Skeleton3D output{};
  std::array<JointEmaDebugInfo, kRehab22JointCount> localDebug{};

  if (!hasLast_) {
    for (std::size_t i = 0; i < kRehab22JointCount; ++i) {
      output[i] = raw[i];
      localDebug[i].alpha = raw[i].valid ? 1.0f : 0.0f;
      localDebug[i].reason = raw[i].valid ? "ema_init" : "raw_invalid";
      if (raw[i].valid) {
        invalidHoldCount_[canonicalJointName(rehab22JointName(i))] = 0;
      }
    }
    last_ = output;
    hasLast_ = true;
    if (debug != nullptr) {
      *debug = localDebug;
    }
    return output;
  }

  const double safeDt = dtSeconds > 0.0 ? dtSeconds : 1.0 / 30.0;
  const uint64_t nowNs = monotonicRawNowNs();
  const bool canLog =
      lastLogNs_ == 0 || nowNs - lastLogNs_ >= 1000000000ULL;

  for (std::size_t i = 0; i < kRehab22JointCount; ++i) {
    const std::string canonical = canonicalJointName(rehab22JointName(i));
    const Keypoint3D& current = raw[i];
    const Keypoint3D& previous = last_[i];
    int& holdCount = invalidHoldCount_[canonical];

    if (!current.valid) {
      if (options_.holdLastWhenInvalid && previous.valid &&
          holdCount < options_.maxHoldFrames) {
        output[i] = previous;
        output[i].valid = true;
        ++holdCount;
        localDebug[i].alpha = options_.alphaInvalid;
        localDebug[i].reason = "ema_hold_invalid";
        localDebug[i].invalidHoldCount = holdCount;
        if (canLog) {
          std::ostringstream oss;
          oss << "[EMA] joint=" << canonical << " invalid hold="
              << holdCount << "/" << options_.maxHoldFrames;
          Logger::info(oss.str());
        }
      } else {
        output[i] = current;
        output[i].valid = false;
        localDebug[i].alpha = options_.alphaInvalid;
        localDebug[i].reason = "ema_hold_expired";
        localDebug[i].invalidHoldCount = holdCount;
        if (canLog && previous.valid) {
          Logger::info("[EMA] joint=" + canonical +
                       " hold expired, set invalid");
        }
      }
      continue;
    }

    holdCount = 0;
    float alpha = options_.alphaGood;
    std::string reason = "ema_good";
    if (current.score < 0.30f) {
      alpha = options_.alphaLowConfidence;
      reason = "ema_low_confidence";
    }
    if (current.foregroundRecovered ||
        current.sampleMethod == "LimbInwardSearch") {
      alpha = options_.alphaRecovered;
      reason = "ema_recovered";
    }

    if (previous.valid) {
      const float zJump = std::abs(current.z - previous.z);
      const float speed = distance3d(current, previous) /
                          static_cast<float>(safeDt);
      if (zJump > options_.maxZJumpM) {
        alpha = std::min(alpha, 0.20f);
        reason = "ema_low_alpha_z_jump";
        if (canLog) {
          std::ostringstream oss;
          oss << "[EMA] joint=" << canonical << " z_jump=" << zJump
              << " use low alpha=" << alpha;
          Logger::info(oss.str());
        }
      } else if (speed > options_.maxJointSpeedMps) {
        alpha = std::min(alpha, 0.20f);
        reason = "ema_low_alpha_speed_jump";
      }
      output[i] = lerpJoint(current, previous, alpha);
    } else {
      output[i] = current;
      alpha = 1.0f;
      reason = "ema_init_joint";
    }

    localDebug[i].alpha = alpha;
    localDebug[i].reason = reason;
    localDebug[i].invalidHoldCount = holdCount;
  }

  if (canLog) {
    lastLogNs_ = nowNs;
  }
  last_ = output;
  if (debug != nullptr) {
    *debug = localDebug;
  }
  return output;
}

}  // namespace rehab

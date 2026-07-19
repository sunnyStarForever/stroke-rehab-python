#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include <opencv2/core.hpp>

#include "engine/emg/EmgTypes.h"
#include "engine/pose/Rehab22Types.h"

namespace rehab {

enum class FrameSource {
  Rgb,
  Depth
};

struct FrameEnvelope {
  FrameSource source{FrameSource::Rgb};
  // Compatibility alias for arrivalTsNs. Synchronizers use syncTsNs.
  uint64_t hostTsNs{0};
  uint64_t deviceTsUs{0};
  uint64_t arrivalTsNs{0};
  uint64_t syncTsNs{0};
  uint64_t frameId{0};

  std::string deviceTimeUnit{"us"};
  std::string clockQuality{"host_fallback"};
  std::string clockReason;
  uint64_t clockResetCount{0};

  int width{0};
  int height{0};

  cv::Mat image;

  float depthUnitToMeter{0.001f};
  std::string pixelFormatName;

  bool valid() const {
    return width > 0 && height > 0 && !image.empty();
  }
};

struct SyncedFramePair {
  FrameEnvelope rgb;
  FrameEnvelope depth;
  int64_t deltaNs{0};
};

struct AlignedFrameSet {
  SyncedFramePair pair;
  cv::Mat alignedDepth;
  bool hardwareD2CUsed{false};
  uint64_t pairId{0};
  Rehab22PoseResult rehabPose;
  double rgbFps{0.0};
  double depthFps{0.0};
  double pairFps{0.0};
  double poseFps{0.0};
  double yoloMs{0.0};
  double poseMs{0.0};
  double recordWriteMs{0.0};
  std::size_t queueLength{0};
  uint64_t droppedPairs{0};
  int poseInterval{0};
  std::string bboxMode{"full_fallback"};
  double deltaMs{0.0};
  bool recording{false};
  bool skeletonRecording{false};
  uint64_t skeletonSavedFrames{0};
  uint64_t rgbRecordedFrames{0};
  uint64_t depthRecordedFrames{0};
  std::optional<EmgFeatureFrame> emgFeature;
  std::string emgStatus;
};

}  // namespace rehab

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

#include "engine/pose/Halpe26Types.h"

namespace rehab {

struct Keypoint3D {
  float x{0.0f};
  float y{0.0f};
  float z{0.0f};
  float score{0.0f};
  bool valid{false};

  float u{0.0f};
  float v{0.0f};
  float rawPoseScore{0.0f};
  uint16_t sampledDepthMm{0};
  std::string sampleMethod;
  std::string sampleReason;
  bool rejectedAsBackground{false};
  bool edgeAmbiguous{false};
  bool foregroundRecovered{false};
};

enum class Rehab22Joint : std::size_t {
  Waist = 0,
  Spine = 1,
  Chest = 2,
  Neck = 3,
  Head = 4,
  HeadTip = 5,
  LeftCollar = 6,
  LeftUpperArm = 7,
  LeftForearm = 8,
  LeftHand = 9,
  RightCollar = 10,
  RightUpperArm = 11,
  RightForearm = 12,
  RightHand = 13,
  LeftUpperLeg = 14,
  LeftLowerLeg = 15,
  LeftFoot = 16,
  LeftToes = 17,
  RightUpperLeg = 18,
  RightLowerLeg = 19,
  RightFoot = 20,
  RightToes = 21,
};

constexpr std::size_t kRehab22JointCount = 22;

using Rehab22Skeleton2D = std::array<Keypoint2D, kRehab22JointCount>;
using Rehab22Skeleton3D = std::array<Keypoint3D, kRehab22JointCount>;
using Rehab22DepthSamples = std::array<float, kRehab22JointCount>;

struct JointDepthDebugInfo {
  std::string canonicalName;
  uint16_t rawDepthSinglePointMm{0};
  uint16_t sampledDepthMm{0};
  bool depthValid{false};
  std::string sampleMethod;
  std::string sampleReason;
  float bodyDepthRefMm{0.0f};
  float backgroundDepthRefMm{0.0f};
  bool rejectedAsBackground{false};
  bool edgeAmbiguous{false};
  bool foregroundRecovered{false};
  int foregroundPixelCount{0};
  int rejectedBackgroundCount{0};
  int validPixelCount{0};
  int usedRadius{0};
};

struct JointEmaDebugInfo {
  float alpha{0.0f};
  std::string reason;
  int invalidHoldCount{0};
};

enum class Pose2DSource {
  None,
  CurrentFrame,
  Reused,
};

inline const char* pose2DSourceName(Pose2DSource source) {
  switch (source) {
    case Pose2DSource::CurrentFrame:
      return "current";
    case Pose2DSource::Reused:
      return "reused";
    case Pose2DSource::None:
    default:
      return "none";
  }
}

struct Rehab22PoseResult {
  uint64_t hostTsNs{0};
  bool modelLoaded{false};
  bool intrinsicsValid{false};
  bool hasValid2D{false};
  bool hasValid3D{false};
  Halpe26Skeleton2D halpe26{};
  Rehab22Skeleton2D joints2d{};
  Rehab22Skeleton3D joints3d{};
  Rehab22Skeleton3D rawJoints3d{};
  Rehab22Skeleton3D emaJoints3d{};
  Rehab22DepthSamples depthMeters{};
  std::array<int, kRehab22JointCount> depthValidCounts{};
  std::array<JointDepthDebugInfo, kRehab22JointCount> depthDebug{};
  std::array<JointEmaDebugInfo, kRehab22JointCount> emaDebug{};
  int depthInvalidCount{0};
  int depthJumpRejectedCount{0};
  Pose2DSource poseSource{Pose2DSource::None};
  bool poseReusedFor2D{false};
  uint64_t poseFrameId{0};
  double poseAgeMs{0.0};
};

inline const char* rehab22JointName(std::size_t idx) {
  static constexpr const char* kNames[kRehab22JointCount] = {
      "Waist",       "Spine",        "Chest",      "Neck",
      "Head",        "HeadTip",      "LeftCollar", "LeftUpperArm",
      "LeftForearm", "LeftHand",     "RightCollar", "RightUpperArm",
      "RightForearm", "RightHand",   "LeftUpperLeg", "LeftLowerLeg",
      "LeftFoot",    "LeftToes",     "RightUpperLeg", "RightLowerLeg",
      "RightFoot",   "RightToes"};
  return (idx < kRehab22JointCount) ? kNames[idx] : "Unknown";
}

inline const std::array<std::pair<int, int>, 21>& rehab22BonePairs() {
  static const std::array<std::pair<int, int>, 21> kBones = {{
      {0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5},
      {3, 6}, {6, 7}, {7, 8}, {8, 9},
      {3, 10}, {10, 11}, {11, 12}, {12, 13},
      {0, 14}, {14, 15}, {15, 16}, {16, 17},
      {0, 18}, {18, 19}, {19, 20}, {20, 21},
  }};
  return kBones;
}

}  // namespace rehab

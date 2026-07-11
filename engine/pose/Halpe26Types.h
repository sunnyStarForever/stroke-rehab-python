#pragma once

#include <array>
#include <cstddef>

namespace rehab {

struct Keypoint2D {
  float x{0.0f};
  float y{0.0f};
  float score{0.0f};
  float rawScore{0.0f};
  bool valid{false};
};

enum class Halpe26Joint : std::size_t {
  Nose = 0,
  LeftEye = 1,
  RightEye = 2,
  LeftEar = 3,
  RightEar = 4,
  LeftShoulder = 5,
  RightShoulder = 6,
  LeftElbow = 7,
  RightElbow = 8,
  LeftWrist = 9,
  RightWrist = 10,
  LeftHip = 11,
  RightHip = 12,
  LeftKnee = 13,
  RightKnee = 14,
  LeftAnkle = 15,
  RightAnkle = 16,
  Head = 17,
  Neck = 18,
  Hip = 19,
  LeftBigToe = 20,
  RightBigToe = 21,
  LeftSmallToe = 22,
  RightSmallToe = 23,
  LeftHeel = 24,
  RightHeel = 25,
};

constexpr std::size_t kHalpe26JointCount = 26;
using Halpe26Skeleton2D = std::array<Keypoint2D, kHalpe26JointCount>;

}  // namespace rehab

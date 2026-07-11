/*
 * 模块作用：
 * 本文件实现 Halpe26 -> Rehab22 映射。它不改变原始推理算法，
 * 只把通用关键点整理成康复采集需要的 22 点骨架。
 */
#include "engine/pose/Halpe26ToRehab22Mapper.h"

namespace rehab {

namespace {

Keypoint2D joint(const Halpe26Skeleton2D& halpe26, Halpe26Joint index) {
  return halpe26[static_cast<std::size_t>(index)];
}

}  // namespace

Keypoint2D Halpe26ToRehab22Mapper::midpoint(const Keypoint2D& a,
                                            const Keypoint2D& b) {
  return interpolate(a, b, 0.5f);
}

Keypoint2D Halpe26ToRehab22Mapper::interpolate(const Keypoint2D& a,
                                               const Keypoint2D& b,
                                               float t) {
  Keypoint2D out;
  if (!a.valid || !b.valid) {
    return out;
  }
  out.x = a.x * (1.0f - t) + b.x * t;
  out.y = a.y * (1.0f - t) + b.y * t;
  out.score = a.score * (1.0f - t) + b.score * t;
  out.rawScore = a.rawScore * (1.0f - t) + b.rawScore * t;
  out.valid = true;
  return out;
}

Keypoint2D Halpe26ToRehab22Mapper::fallback(const Keypoint2D& primary,
                                            const Keypoint2D& secondary) {
  return primary.valid ? primary : secondary;
}

Rehab22Skeleton2D Halpe26ToRehab22Mapper::map(const Halpe26Skeleton2D& halpe26) const {
  /*
   * map()
   * 输入：RTMPose Halpe26 关键点。
   * 输出：Rehab22 2D 关键点。
   * 说明：Rehab22 中部分点不是模型直接输出，需要通过两侧关节插值或备用点推断。
   */
  Rehab22Skeleton2D out{};

  const Keypoint2D neck = joint(halpe26, Halpe26Joint::Neck);
  const Keypoint2D nose = joint(halpe26, Halpe26Joint::Nose);
  const Keypoint2D head = joint(halpe26, Halpe26Joint::Head);
  const Keypoint2D hip = joint(halpe26, Halpe26Joint::Hip);
  const Keypoint2D leftShoulder = joint(halpe26, Halpe26Joint::LeftShoulder);
  const Keypoint2D rightShoulder = joint(halpe26, Halpe26Joint::RightShoulder);
  const Keypoint2D leftHip = joint(halpe26, Halpe26Joint::LeftHip);
  const Keypoint2D rightHip = joint(halpe26, Halpe26Joint::RightHip);

  const Keypoint2D chest = midpoint(leftShoulder, rightShoulder);
  const Keypoint2D waist = fallback(hip, midpoint(leftHip, rightHip));

  out[static_cast<std::size_t>(Rehab22Joint::Waist)] = waist;
  out[static_cast<std::size_t>(Rehab22Joint::Chest)] = chest;
  out[static_cast<std::size_t>(Rehab22Joint::Neck)] = neck;
  out[static_cast<std::size_t>(Rehab22Joint::Head)] = nose;

  // 脊柱点由腰部和胸口中点推断，用于形成稳定的躯干链。
  out[static_cast<std::size_t>(Rehab22Joint::Spine)] = interpolate(waist, chest, 0.5f);
  out[static_cast<std::size_t>(Rehab22Joint::HeadTip)] = head;

  // 锁骨点由颈部向肩部插值得到，补足 Rehab22 对肩颈连接的表达。
  out[static_cast<std::size_t>(Rehab22Joint::LeftCollar)] =
      interpolate(neck, leftShoulder, 0.35f);
  out[static_cast<std::size_t>(Rehab22Joint::RightCollar)] =
      interpolate(neck, rightShoulder, 0.35f);

  out[static_cast<std::size_t>(Rehab22Joint::LeftUpperArm)] = leftShoulder;
  out[static_cast<std::size_t>(Rehab22Joint::LeftForearm)] =
      joint(halpe26, Halpe26Joint::LeftElbow);
  out[static_cast<std::size_t>(Rehab22Joint::LeftHand)] =
      joint(halpe26, Halpe26Joint::LeftWrist);

  out[static_cast<std::size_t>(Rehab22Joint::RightUpperArm)] = rightShoulder;
  out[static_cast<std::size_t>(Rehab22Joint::RightForearm)] =
      joint(halpe26, Halpe26Joint::RightElbow);
  out[static_cast<std::size_t>(Rehab22Joint::RightHand)] =
      joint(halpe26, Halpe26Joint::RightWrist);

  out[static_cast<std::size_t>(Rehab22Joint::LeftUpperLeg)] = leftHip;
  out[static_cast<std::size_t>(Rehab22Joint::LeftLowerLeg)] =
      joint(halpe26, Halpe26Joint::LeftKnee);
  out[static_cast<std::size_t>(Rehab22Joint::RightUpperLeg)] = rightHip;
  out[static_cast<std::size_t>(Rehab22Joint::RightLowerLeg)] =
      joint(halpe26, Halpe26Joint::RightKnee);

  const Keypoint2D leftAnkle = joint(halpe26, Halpe26Joint::LeftAnkle);
  const Keypoint2D rightAnkle = joint(halpe26, Halpe26Joint::RightAnkle);
  const Keypoint2D leftHeel = joint(halpe26, Halpe26Joint::LeftHeel);
  const Keypoint2D rightHeel = joint(halpe26, Halpe26Joint::RightHeel);

  // 足部中心优先使用踝和脚跟的中点，脚跟缺失时退回踝点。
  out[static_cast<std::size_t>(Rehab22Joint::LeftFoot)] =
      fallback(midpoint(leftAnkle, leftHeel), leftAnkle);
  out[static_cast<std::size_t>(Rehab22Joint::RightFoot)] =
      fallback(midpoint(rightAnkle, rightHeel), rightAnkle);

  const Keypoint2D leftBigToe = joint(halpe26, Halpe26Joint::LeftBigToe);
  const Keypoint2D leftSmallToe = joint(halpe26, Halpe26Joint::LeftSmallToe);
  const Keypoint2D rightBigToe = joint(halpe26, Halpe26Joint::RightBigToe);
  const Keypoint2D rightSmallToe = joint(halpe26, Halpe26Joint::RightSmallToe);

  out[static_cast<std::size_t>(Rehab22Joint::LeftToes)] =
      midpoint(leftBigToe, leftSmallToe);
  out[static_cast<std::size_t>(Rehab22Joint::RightToes)] =
      midpoint(rightBigToe, rightSmallToe);

  return out;
}

}  // namespace rehab

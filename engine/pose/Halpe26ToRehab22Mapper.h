/*
 * 模块作用：
 * 本文件声明 Halpe26 到 Rehab22 的骨架映射。
 * RTMPose 输出的是通用人体关键点，Rehab22 是本项目康复记录和显示统一使用的关节集合。
 */
#pragma once

#include "engine/pose/Halpe26Types.h"
#include "engine/pose/Rehab22Types.h"

namespace rehab {

/*
 * Halpe26ToRehab22Mapper
 * 职责：
 * 1. 直接复用 Halpe26 中存在的肩、肘、腕、髋、膝、踝等点；
 * 2. 通过中点或插值构造脊柱、胸口、锁骨、足部中心等 Rehab22 点；
 * 3. 在主流程中把模型输出转换成稳定的康复骨架拓扑。
 */
class Halpe26ToRehab22Mapper {
 public:
  Rehab22Skeleton2D map(const Halpe26Skeleton2D& halpe26) const;

 private:
  static Keypoint2D midpoint(const Keypoint2D& a, const Keypoint2D& b);
  static Keypoint2D interpolate(const Keypoint2D& a, const Keypoint2D& b, float t);
  static Keypoint2D fallback(const Keypoint2D& primary, const Keypoint2D& secondary);
};

}  // namespace rehab

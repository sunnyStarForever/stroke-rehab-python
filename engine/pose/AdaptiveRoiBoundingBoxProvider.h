/*
 * 模块作用：
 * 本文件声明自适应 ROI 提供器。它让 YOLO 不必每帧运行，
 * 而是在检测框、上一帧关键点和周期性重检之间平衡精度与速度。
 */
#pragma once

#include <memory>
#include <string>

#include <opencv2/core.hpp>

#include "engine/pose/PersonDetectorOrt.h"
#include "engine/pose/PoseEstimator.h"

namespace rehab {

struct AdaptiveRoiConfig {
  int detectorInterval{30};       // 每隔多少帧强制跑一次 YOLO 校正 ROI
  float roiMarginRatio{0.20f};    // ROI 外扩比例，避免肢体贴边被裁掉
  float minTrackMeanScore{0.25f}; // 关键点平均分低于该值认为跟踪变弱
  int minTrackValidPoints{6};     // 有效关键点太少时触发重新检测
  int maxConsecutiveMisses{3};    // 连续弱跟踪次数超过阈值后强制 YOLO
  float motionTriggerRatio{0.35f}; // 人体中心移动超过 ROI 尺寸比例后重检
};

/*
 * AdaptiveRoiBoundingBoxProvider
 * 职责：
 * 1. 首帧或失效时调用 YOLO 获取人体框；
 * 2. 之后用姿态关键点反推人体框，减少 YOLO 运行频率；
 * 3. 低分、丢点或大幅移动时重新触发 YOLO，防止 ROI 漂移。
 */
class AdaptiveRoiBoundingBoxProvider : public BoundingBoxProvider {
 public:
  AdaptiveRoiBoundingBoxProvider(std::shared_ptr<PersonDetectorOrt> detector,
                                 AdaptiveRoiConfig config);

  BoundingBox2D getPrimaryBox(const cv::Mat& bgr) override;
  void updateFromPose(const BoundingBox2D& usedBox,
                      const Halpe26Skeleton2D& keypoints) override;
  void reset() override;
  std::string debugState() const override;

 private:
  BoundingBox2D fullImageBox(const cv::Size& imageSize) const;
  BoundingBox2D expandAndClip(const BoundingBox2D& box,
                              const cv::Size& imageSize) const;
  BoundingBox2D poseBoxFromKeypoints(const Halpe26Skeleton2D& keypoints,
                                     float* meanScore,
                                     int* validCount) const;

  std::shared_ptr<PersonDetectorOrt> detector_;
  AdaptiveRoiConfig config_;

  BoundingBox2D trackedBox_{};
  int frameCounter_{0};
  int missCounter_{0};
  bool forceRedetect_{true};
  cv::Size lastImageSize_{0, 0};
  std::string lastState_{"detect"};
};

}  // namespace rehab

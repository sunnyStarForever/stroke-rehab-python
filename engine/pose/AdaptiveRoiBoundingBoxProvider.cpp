/*
 * 模块作用：
 * 本文件实现自适应人体 ROI。ROI 越小，RTMPose 输入越聚焦、推理越省；
 * 但 ROI 漂移会造成关键点丢失，因此需要周期性 YOLO 和质量检查。
 */
#include <cmath>
#include <limits>
#include <utility>

#include "engine/pose/AdaptiveRoiBoundingBoxProvider.h"

namespace rehab {

namespace {

float maxFloat(float a, float b) {
  return a > b ? a : b;
}

float minFloat(float a, float b) {
  return a < b ? a : b;
}

float clampFloat(float value, float low, float high) {
  if (value < low) {
    return low;
  }
  if (value > high) {
    return high;
  }
  return value;
}

}  // namespace

AdaptiveRoiBoundingBoxProvider::AdaptiveRoiBoundingBoxProvider(
    std::shared_ptr<PersonDetectorOrt> detector, AdaptiveRoiConfig config)
    : detector_(std::move(detector)), config_(config) {}

BoundingBox2D AdaptiveRoiBoundingBoxProvider::fullImageBox(
    const cv::Size& imageSize) const {
  BoundingBox2D box;
  if (imageSize.width <= 0 || imageSize.height <= 0) {
    return box;
  }
  box.x = 0.0f;
  box.y = 0.0f;
  box.w = static_cast<float>(imageSize.width);
  box.h = static_cast<float>(imageSize.height);
  box.score = 1.0f;
  box.valid = box.w > 1.0f && box.h > 1.0f;
  return box;
}

BoundingBox2D AdaptiveRoiBoundingBoxProvider::expandAndClip(
    const BoundingBox2D& box, const cv::Size& imageSize) const {
  // ROI 外扩后再裁剪到图像边界，避免手脚在运动中跑出 RTMPose 裁剪区域。
  if (!box.valid || imageSize.width <= 0 || imageSize.height <= 0) {
    return {};
  }

  const float margin = maxFloat(0.0f, config_.roiMarginRatio);
  const float expandX = box.w * margin;
  const float expandY = box.h * margin;

  float x1 = box.x - expandX;
  float y1 = box.y - expandY;
  float x2 = box.x + box.w + expandX;
  float y2 = box.y + box.h + expandY;

  x1 = clampFloat(x1, 0.0f, static_cast<float>(imageSize.width - 1));
  y1 = clampFloat(y1, 0.0f, static_cast<float>(imageSize.height - 1));
  x2 = clampFloat(x2, 1.0f, static_cast<float>(imageSize.width));
  y2 = clampFloat(y2, 1.0f, static_cast<float>(imageSize.height));

  BoundingBox2D out;
  out.x = x1;
  out.y = y1;
  out.w = x2 - x1;
  out.h = y2 - y1;
  out.score = box.score;
  out.valid = out.w > 1.0f && out.h > 1.0f;
  return out;
}

BoundingBox2D AdaptiveRoiBoundingBoxProvider::poseBoxFromKeypoints(
    const Halpe26Skeleton2D& keypoints, float* meanScore, int* validCount) const {
  // 用当前关键点包围盒估计下一帧 ROI，降低 YOLO 调用频率。
  if (meanScore != nullptr) {
    *meanScore = 0.0f;
  }
  if (validCount != nullptr) {
    *validCount = 0;
  }

  float minX = std::numeric_limits<float>::infinity();
  float minY = std::numeric_limits<float>::infinity();
  float maxX = -std::numeric_limits<float>::infinity();
  float maxY = -std::numeric_limits<float>::infinity();
  float scoreSum = 0.0f;
  int count = 0;

  for (const Keypoint2D& point : keypoints) {
    if (!point.valid) {
      continue;
    }
    minX = minFloat(minX, point.x);
    minY = minFloat(minY, point.y);
    maxX = maxFloat(maxX, point.x);
    maxY = maxFloat(maxY, point.y);
    scoreSum += point.score;
    ++count;
  }

  if (validCount != nullptr) {
    *validCount = count;
  }
  if (meanScore != nullptr) {
    *meanScore = count > 0 ? (scoreSum / static_cast<float>(count)) : 0.0f;
  }

  BoundingBox2D box;
  if (count <= 0) {
    return box;
  }
  box.x = minX;
  box.y = minY;
  box.w = maxX - minX;
  box.h = maxY - minY;
  box.score = count > 0 ? (scoreSum / static_cast<float>(count)) : 0.0f;
  box.valid = box.w > 1.0f && box.h > 1.0f;
  return box;
}

BoundingBox2D AdaptiveRoiBoundingBoxProvider::getPrimaryBox(const cv::Mat& bgr) {
  /*
   * getPrimaryBox()
   * 首帧、周期帧或跟踪失效时跑 YOLO；否则直接返回 trackedBox_。
   * 这样 RTMPose 可以低频结合检测，高频保持 ROI 近似稳定。
   */
  if (bgr.empty()) {
    lastState_ = "full_fallback";
    return {};
  }

  lastImageSize_ = bgr.size();
  ++frameCounter_;

  const int interval = std::max(1, config_.detectorInterval);
  const bool periodicDetect = ((frameCounter_ - 1) % interval) == 0;
  const bool shouldDetect =
      forceRedetect_ || !trackedBox_.valid || frameCounter_ <= 1 || periodicDetect;

  if (shouldDetect && detector_ && detector_->isInitialized()) {
    BoundingBox2D detected = detector_->detectLargestPerson(bgr);
    if (detected.valid) {
      trackedBox_ = expandAndClip(detected, lastImageSize_);
      forceRedetect_ = false;
      lastState_ = "detect";
      return trackedBox_;
    }
    if (trackedBox_.valid) {
      lastState_ = "track_fallback";
      return trackedBox_;
    }
    lastState_ = "no_person";
    return {};
  }

  if (trackedBox_.valid) {
    lastState_ = shouldDetect ? "track_fallback" : "track";
    return trackedBox_;
  }

  lastState_ = "no_person";
  return {};
}

void AdaptiveRoiBoundingBoxProvider::updateFromPose(
    const BoundingBox2D& usedBox, const Halpe26Skeleton2D& keypoints) {
  // RTMPose 输出质量会反馈给 ROI：分数低或点太少表示框可能偏了，需要重新检测。
  float meanScore = 0.0f;
  int validCount = 0;
  BoundingBox2D poseBox = poseBoxFromKeypoints(keypoints, &meanScore, &validCount);

  const bool poseWeak = meanScore < config_.minTrackMeanScore ||
                        validCount < config_.minTrackValidPoints;
  if (poseWeak) {
    ++missCounter_;
  }
  if (missCounter_ >= config_.maxConsecutiveMisses) {
    forceRedetect_ = true;
  }

  if (!poseBox.valid) {
    forceRedetect_ = true;
    return;
  }

  const float poseCx = poseBox.x + poseBox.w * 0.5f;
  const float poseCy = poseBox.y + poseBox.h * 0.5f;
  const float usedCx = usedBox.x + usedBox.w * 0.5f;
  const float usedCy = usedBox.y + usedBox.h * 0.5f;
  const float dx = poseCx - usedCx;
  const float dy = poseCy - usedCy;
  const float moveDistance = std::sqrt(dx * dx + dy * dy);
  const float moveThreshold = config_.motionTriggerRatio *
                              maxFloat(maxFloat(usedBox.w, usedBox.h), 1.0f);
  if (moveDistance > moveThreshold) {
    forceRedetect_ = true;
  }

  if (!poseWeak) {
    missCounter_ = 0;
    trackedBox_ = expandAndClip(poseBox, lastImageSize_);
  }
}

void AdaptiveRoiBoundingBoxProvider::reset() {
  trackedBox_ = {};
  frameCounter_ = 0;
  missCounter_ = 0;
  forceRedetect_ = true;
  lastImageSize_ = cv::Size(0, 0);
  lastState_ = "detect";
}

std::string AdaptiveRoiBoundingBoxProvider::debugState() const {
  return lastState_;
}

}  // namespace rehab

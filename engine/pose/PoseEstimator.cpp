#include "engine/pose/PoseEstimator.h"

namespace rehab {

PoseEstimator::~PoseEstimator() = default;

BoundingBox2D FullImageBoundingBoxProvider::getPrimaryBox(const cv::Mat& bgr) {
  BoundingBox2D box;
  if (bgr.empty()) {
    return box;
  }
  box.x = 0.0f;
  box.y = 0.0f;
  box.w = static_cast<float>(bgr.cols);
  box.h = static_cast<float>(bgr.rows);
  box.score = 1.0f;
  box.valid = box.w > 1.0f && box.h > 1.0f;
  return box;
}

void FullImageBoundingBoxProvider::updateFromPose(
    const BoundingBox2D& /*usedBox*/, const Halpe26Skeleton2D& /*keypoints*/) {}

void FullImageBoundingBoxProvider::reset() {}

std::string FullImageBoundingBoxProvider::debugState() const {
  return "full_fallback";
}

}  // namespace rehab

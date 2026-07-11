#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "engine/pose/Rehab22Types.h"
#include "engine/pose/SkeletonTypes.h"

namespace rehab {

enum class DepthSampleMethod {
  SinglePoint,
  MedianWindow,
  ForegroundWindow,
  ForegroundPercentile,
  LimbInwardSearch,
  FailedOutOfBounds,
  FailedNoValidDepth,
  FailedBackgroundHit
};

std::string depthSampleMethodToString(DepthSampleMethod method);

struct DepthSampleContext {
  float bodyDepthRefMm{0.0f};
  float backgroundDepthRefMm{0.0f};
  bool hasBodyDepthRef{false};
  bool hasBackgroundDepthRef{false};
};

struct DepthSampleResult {
  bool valid{false};
  uint16_t depthRawMm{0};
  float depthMeters{0.0f};

  int u{0};
  int v{0};

  int usedRadius{0};
  int validPixelCount{0};
  int foregroundPixelCount{0};
  int rejectedBackgroundCount{0};

  float bodyDepthRefMm{0.0f};
  float backgroundDepthRefMm{0.0f};
  float depthBeforeRejectMm{0.0f};

  bool rejectedAsBackground{false};
  bool edgeAmbiguous{false};
  bool foregroundRecovered{false};

  DepthSampleMethod method{DepthSampleMethod::FailedNoValidDepth};
  std::string reason;
};

struct DepthSamplingStats {
  std::array<int, kRehab22JointCount> validCounts{};
  int validJoints{0};
  int invalidJoints{0};
};

class DepthSampler {
 public:
  struct Options {
    int minDepthMm{300};
    int maxDepthMm{5000};

    int bodyDepthBandMm{700};
    int edgeBodyDepthBandMm{900};
    int backgroundRejectMarginMm{500};
    int backgroundMatchBandMm{300};

    float foregroundPercentile{0.20f};
    int minForegroundPixels{3};

    int hipRadius{2};
    int kneeRadius{3};
    int ankleRadius{4};
    int toeRadius{5};
    int wristRadius{4};
    int defaultRadius{2};

    bool limbInwardSearchEnabled{true};
    int limbInwardSteps{10};
    int limbInwardStepPx{3};
    int limbInwardRadius{3};
  };

  DepthSampler();
  explicit DepthSampler(Options options);
  void setOptions(const Options& options) { options_ = options; }

  DepthSampleContext estimateDepthContext(
      const cv::Mat& alignedDepth,
      const std::vector<Joint2D>& joints2d,
      float depthUnitToMeter,
      const cv::Rect& personRoi = cv::Rect()) const;

  DepthSampleResult sampleJoint(const cv::Mat& alignedDepth,
                                const Joint2D& joint,
                                const Joint2D* parentJoint,
                                const DepthSampleContext& context,
                                float depthUnitToMeter) const;

  Rehab22DepthSamples sample(const cv::Mat& alignedDepth,
                             const Rehab22Skeleton2D& joints2d,
                             float depthUnitToMeter,
                             int windowSize = 7,
                             DepthSamplingStats* stats = nullptr) const;

  float sampleSingle(const cv::Mat& alignedDepth,
                     float x,
                     float y,
                     float depthUnitToMeter,
                     int windowSize = 7,
                     int* validCount = nullptr) const;

 private:
  DepthSampleResult sampleForegroundJointDepth(
      const cv::Mat& alignedDepth,
      float u,
      float v,
      const std::string& jointName,
      const DepthSampleContext& context,
      float depthUnitToMeter) const;

  DepthSampleResult sampleWithLimbInwardFallback(
      const cv::Mat& alignedDepth,
      float jointU,
      float jointV,
      float parentU,
      float parentV,
      const std::string& jointName,
      const DepthSampleContext& context,
      float depthUnitToMeter) const;

  float estimateBackgroundDepthMm(const cv::Mat& alignedDepth,
                                  const cv::Rect& personRoi) const;
  float estimateBackgroundDepthMm(const cv::Mat& alignedDepth,
                                  const cv::Rect& personRoi,
                                  float depthUnitToMeter) const;

  int radiusForJoint(const std::string& canonicalName) const;
  bool isEdgeJoint(const std::string& canonicalName) const;
  bool isLowerBodyOrHandEdgeJoint(const std::string& canonicalName) const;

  DepthSampleResult sampleForegroundAt(const cv::Mat& alignedDepth,
                                       float u,
                                       float v,
                                       const std::string& jointName,
                                       const DepthSampleContext& context,
                                       float depthUnitToMeter,
                                       int radiusOverride) const;

  uint16_t rawDepthToMm(uint16_t raw, float depthUnitToMeter) const;
  static int normalizeWindowSize(int requested);

  Options options_;
};

}  // namespace rehab

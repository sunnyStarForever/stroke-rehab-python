#include "engine/pose/DepthSampler.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <sstream>

#include "engine/pose/JointNameMapper.h"
#include "engine/util/Logger.h"

namespace rehab {

namespace {

bool contains(const std::string& value, const std::string& needle) {
  return value.find(needle) != std::string::npos;
}

uint16_t medianOf(std::vector<uint16_t>* values) {
  if (values == nullptr || values->empty()) {
    return 0;
  }
  const std::size_t mid = values->size() / 2;
  std::nth_element(values->begin(),
                   values->begin() + static_cast<std::ptrdiff_t>(mid),
                   values->end());
  return (*values)[mid];
}

uint16_t percentileOf(std::vector<uint16_t>* values, float percentile) {
  if (values == nullptr || values->empty()) {
    return 0;
  }
  std::sort(values->begin(), values->end());
  const float clamped = std::clamp(percentile, 0.0f, 1.0f);
  const std::size_t idx = static_cast<std::size_t>(
      std::round(clamped * static_cast<float>(values->size() - 1)));
  return (*values)[std::min(idx, values->size() - 1)];
}

bool isTorsoContextJoint(const std::string& canonical) {
  return canonical == "waist" || canonical == "spine" ||
         canonical == "chest" || canonical == "neck" ||
         canonical == "left_hip" || canonical == "right_hip" ||
         canonical == "pelvis";
}

}  // namespace

std::string depthSampleMethodToString(DepthSampleMethod method) {
  switch (method) {
    case DepthSampleMethod::SinglePoint:
      return "SinglePoint";
    case DepthSampleMethod::MedianWindow:
      return "MedianWindow";
    case DepthSampleMethod::ForegroundWindow:
      return "ForegroundWindow";
    case DepthSampleMethod::ForegroundPercentile:
      return "ForegroundPercentile";
    case DepthSampleMethod::LimbInwardSearch:
      return "LimbInwardSearch";
    case DepthSampleMethod::FailedOutOfBounds:
      return "FailedOutOfBounds";
    case DepthSampleMethod::FailedBackgroundHit:
      return "FailedBackgroundHit";
    case DepthSampleMethod::FailedNoValidDepth:
    default:
      return "FailedNoValidDepth";
  }
}

DepthSampler::DepthSampler() : options_() {}

DepthSampler::DepthSampler(Options options) : options_(options) {}

int DepthSampler::normalizeWindowSize(int requested) {
  int window = requested;
  if (window < 3) {
    window = 3;
  }
  if (window % 2 == 0) {
    ++window;
  }
  return window;
}

uint16_t DepthSampler::rawDepthToMm(uint16_t raw,
                                    float depthUnitToMeter) const {
  if (raw == 0 || depthUnitToMeter <= 0.0f) {
    return 0;
  }
  return static_cast<uint16_t>(
      std::round(static_cast<float>(raw) * depthUnitToMeter * 1000.0f));
}

DepthSampleContext DepthSampler::estimateDepthContext(
    const cv::Mat& alignedDepth,
    const std::vector<Joint2D>& joints2d,
    float depthUnitToMeter,
    const cv::Rect& personRoi) const {
  DepthSampleContext context;
  if (alignedDepth.empty() || alignedDepth.type() != CV_16UC1) {
    return context;
  }

  std::vector<uint16_t> torsoDepths;
  for (const Joint2D& joint : joints2d) {
    if (!joint.valid) {
      continue;
    }
    const std::string canonical = canonicalJointName(joint.name);
    if (!isTorsoContextJoint(canonical)) {
      continue;
    }
    int validCount = 0;
    const float depth = sampleSingle(alignedDepth, joint.x, joint.y,
                                     depthUnitToMeter, 5, &validCount);
    if (depth > 0.0f && validCount > 0) {
      torsoDepths.push_back(static_cast<uint16_t>(std::round(depth * 1000.0f)));
    }
  }

  if (torsoDepths.size() >= 3) {
    context.bodyDepthRefMm = static_cast<float>(medianOf(&torsoDepths));
    context.hasBodyDepthRef = true;
  }

  const float background =
      estimateBackgroundDepthMm(alignedDepth, personRoi, depthUnitToMeter);
  if (background > 0.0f) {
    context.backgroundDepthRefMm = background;
    context.hasBackgroundDepthRef = true;
  }
  return context;
}

float DepthSampler::estimateBackgroundDepthMm(
    const cv::Mat& alignedDepth,
    const cv::Rect& personRoi) const {
  return estimateBackgroundDepthMm(alignedDepth, personRoi, 0.001f);
}

float DepthSampler::estimateBackgroundDepthMm(
    const cv::Mat& alignedDepth,
    const cv::Rect& personRoi,
    float depthUnitToMeter) const {
  if (alignedDepth.empty() || alignedDepth.type() != CV_16UC1) {
    return 0.0f;
  }

  const cv::Rect imageRect(0, 0, alignedDepth.cols, alignedDepth.rows);
  const cv::Rect roi = personRoi.area() > 0 ? (personRoi & imageRect)
                                            : cv::Rect();
  std::vector<uint16_t> samples;
  const int step = 16;
  for (int y = 0; y < alignedDepth.rows; y += step) {
    const uint16_t* row = alignedDepth.ptr<uint16_t>(y);
    for (int x = 0; x < alignedDepth.cols; x += step) {
      if (roi.area() > 0) {
        if (roi.contains(cv::Point(x, y))) {
          continue;
        }
      } else {
        const bool border = x < alignedDepth.cols / 6 ||
                            x > alignedDepth.cols * 5 / 6 ||
                            y < alignedDepth.rows / 5;
        if (!border) {
          continue;
        }
      }
      const uint16_t depthMm = rawDepthToMm(row[x], depthUnitToMeter);
      if (depthMm >= options_.minDepthMm && depthMm <= options_.maxDepthMm) {
        samples.push_back(depthMm);
      }
    }
  }
  if (samples.size() < 10) {
    return 0.0f;
  }
  return static_cast<float>(medianOf(&samples));
}

DepthSampleResult DepthSampler::sampleJoint(
    const cv::Mat& alignedDepth,
    const Joint2D& joint,
    const Joint2D* parentJoint,
    const DepthSampleContext& context,
    float depthUnitToMeter) const {
  const std::string canonical = canonicalJointName(joint.name);
  if (!joint.valid) {
    DepthSampleResult result;
    result.u = static_cast<int>(std::round(joint.x));
    result.v = static_cast<int>(std::round(joint.y));
    result.method = DepthSampleMethod::FailedNoValidDepth;
    result.reason = "joint_invalid";
    return result;
  }

  const bool needsInward =
      canonical == "left_knee" || canonical == "right_knee" ||
      canonical == "left_ankle" || canonical == "right_ankle" ||
      canonical == "left_toe" || canonical == "right_toe" ||
      canonical == "left_wrist" || canonical == "right_wrist" ||
      canonical == "left_hand" || canonical == "right_hand";

  if (needsInward && parentJoint != nullptr && parentJoint->valid) {
    return sampleWithLimbInwardFallback(alignedDepth, joint.x, joint.y,
                                        parentJoint->x, parentJoint->y,
                                        joint.name, context,
                                        depthUnitToMeter);
  }

  return sampleForegroundJointDepth(alignedDepth, joint.x, joint.y, joint.name,
                                    context, depthUnitToMeter);
}

DepthSampleResult DepthSampler::sampleForegroundJointDepth(
    const cv::Mat& alignedDepth,
    float u,
    float v,
    const std::string& jointName,
    const DepthSampleContext& context,
    float depthUnitToMeter) const {
  return sampleForegroundAt(alignedDepth, u, v, jointName, context,
                            depthUnitToMeter, -1);
}

DepthSampleResult DepthSampler::sampleForegroundAt(
    const cv::Mat& alignedDepth,
    float u,
    float v,
    const std::string& jointName,
    const DepthSampleContext& context,
    float depthUnitToMeter,
    int radiusOverride) const {
  DepthSampleResult result;
  result.u = static_cast<int>(std::round(u));
  result.v = static_cast<int>(std::round(v));
  result.bodyDepthRefMm = context.bodyDepthRefMm;
  result.backgroundDepthRefMm = context.backgroundDepthRefMm;

  if (alignedDepth.empty() || alignedDepth.type() != CV_16UC1 ||
      depthUnitToMeter <= 0.0f) {
    result.method = DepthSampleMethod::FailedNoValidDepth;
    result.reason = "depth_image_invalid";
    return result;
  }
  if (result.u < 0 || result.u >= alignedDepth.cols || result.v < 0 ||
      result.v >= alignedDepth.rows) {
    result.method = DepthSampleMethod::FailedOutOfBounds;
    result.reason = "joint_out_of_bounds";
    return result;
  }

  const std::string canonical = canonicalJointName(jointName);
  const int radius =
      radiusOverride >= 0 ? radiusOverride : radiusForJoint(canonical);
  result.usedRadius = radius;

  const uint16_t centerRaw =
      alignedDepth.at<uint16_t>(result.v, result.u);
  result.depthBeforeRejectMm =
      static_cast<float>(rawDepthToMm(centerRaw, depthUnitToMeter));

  std::vector<uint16_t> foregroundValues;
  foregroundValues.reserve(static_cast<std::size_t>((radius * 2 + 1) *
                                                    (radius * 2 + 1)));
  const int xMin = std::max(0, result.u - radius);
  const int xMax = std::min(alignedDepth.cols - 1, result.u + radius);
  const int yMin = std::max(0, result.v - radius);
  const int yMax = std::min(alignedDepth.rows - 1, result.v + radius);
  const float bodyBand = isEdgeJoint(canonical)
                             ? static_cast<float>(options_.edgeBodyDepthBandMm)
                             : static_cast<float>(options_.bodyDepthBandMm);

  for (int y0 = yMin; y0 <= yMax; ++y0) {
    const uint16_t* row = alignedDepth.ptr<uint16_t>(y0);
    for (int x0 = xMin; x0 <= xMax; ++x0) {
      const uint16_t raw = row[x0];
      const uint16_t depthMm = rawDepthToMm(raw, depthUnitToMeter);
      if (depthMm == 0 || depthMm < options_.minDepthMm ||
          depthMm > options_.maxDepthMm) {
        continue;
      }
      ++result.validPixelCount;

      if (context.hasBackgroundDepthRef && context.hasBodyDepthRef &&
          std::abs(static_cast<float>(depthMm) -
                   context.backgroundDepthRefMm) <
              static_cast<float>(options_.backgroundMatchBandMm) &&
          static_cast<float>(depthMm) >
              context.bodyDepthRefMm +
                  static_cast<float>(options_.backgroundRejectMarginMm)) {
        ++result.rejectedBackgroundCount;
        result.rejectedAsBackground = true;
        continue;
      }

      if (context.hasBodyDepthRef &&
          std::abs(static_cast<float>(depthMm) - context.bodyDepthRefMm) >
              bodyBand) {
        continue;
      }

      foregroundValues.push_back(depthMm);
    }
  }

  result.foregroundPixelCount = static_cast<int>(foregroundValues.size());
  if (result.foregroundPixelCount < options_.minForegroundPixels) {
    result.valid = false;
    result.edgeAmbiguous = isEdgeJoint(canonical);
    result.method = result.rejectedBackgroundCount > 0
                        ? DepthSampleMethod::FailedBackgroundHit
                        : DepthSampleMethod::FailedNoValidDepth;
    result.reason = result.rejectedBackgroundCount > 0
                        ? "background_rejected"
                        : "not_enough_foreground_pixels";
    return result;
  }

  const bool usePercentile = isLowerBodyOrHandEdgeJoint(canonical);
  const uint16_t depthMm = usePercentile
                               ? percentileOf(&foregroundValues,
                                              options_.foregroundPercentile)
                               : medianOf(&foregroundValues);
  result.valid = depthMm > 0;
  result.depthRawMm = depthMm;
  result.depthMeters = static_cast<float>(depthMm) / 1000.0f;
  result.method = usePercentile ? DepthSampleMethod::ForegroundPercentile
                                : DepthSampleMethod::ForegroundWindow;
  result.reason = "foreground_window";
  return result;
}

DepthSampleResult DepthSampler::sampleWithLimbInwardFallback(
    const cv::Mat& alignedDepth,
    float jointU,
    float jointV,
    float parentU,
    float parentV,
    const std::string& jointName,
    const DepthSampleContext& context,
    float depthUnitToMeter) const {
  DepthSampleResult result = sampleForegroundJointDepth(
      alignedDepth, jointU, jointV, jointName, context, depthUnitToMeter);
  if (result.valid || !options_.limbInwardSearchEnabled) {
    return result;
  }

  const float dx = parentU - jointU;
  const float dy = parentV - jointV;
  const float len = std::sqrt(dx * dx + dy * dy);
  if (len <= 1.0f) {
    result.reason = "limb_inward_no_parent_direction";
    return result;
  }

  const float dirX = dx / len;
  const float dirY = dy / len;
  for (int step = 1; step <= options_.limbInwardSteps; ++step) {
    const float candidateU =
        jointU + dirX * static_cast<float>(step * options_.limbInwardStepPx);
    const float candidateV =
        jointV + dirY * static_cast<float>(step * options_.limbInwardStepPx);
    DepthSampleResult candidate = sampleForegroundAt(
        alignedDepth, candidateU, candidateV, jointName, context,
        depthUnitToMeter, options_.limbInwardRadius);
    if (candidate.valid) {
      candidate.method = DepthSampleMethod::LimbInwardSearch;
      candidate.foregroundRecovered = true;
      candidate.reason = "limb_inward_recovered";
      std::ostringstream oss;
      oss << "[DEPTH SAMPLE] joint=" << canonicalJointName(jointName)
          << " method=LimbInwardSearch depth=" << candidate.depthRawMm
          << " body_ref=" << candidate.bodyDepthRefMm
          << " bg_ref=" << candidate.backgroundDepthRefMm
          << " reason=" << candidate.reason;
      Logger::info(oss.str());
      return candidate;
    }
  }

  result.reason = "limb_inward_failed";
  return result;
}

int DepthSampler::radiusForJoint(const std::string& canonicalName) const {
  if (canonicalName == "waist" || canonicalName == "chest" ||
      canonicalName == "spine" || canonicalName == "neck" ||
      contains(canonicalName, "hip")) {
    return options_.hipRadius;
  }
  if (contains(canonicalName, "knee")) {
    return options_.kneeRadius;
  }
  if (contains(canonicalName, "ankle")) {
    return options_.ankleRadius;
  }
  if (contains(canonicalName, "toe")) {
    return options_.toeRadius;
  }
  if (contains(canonicalName, "wrist") || contains(canonicalName, "hand")) {
    return options_.wristRadius;
  }
  return options_.defaultRadius;
}

bool DepthSampler::isEdgeJoint(const std::string& canonicalName) const {
  return contains(canonicalName, "knee") || contains(canonicalName, "ankle") ||
         contains(canonicalName, "toe") || contains(canonicalName, "wrist") ||
         contains(canonicalName, "hand");
}

bool DepthSampler::isLowerBodyOrHandEdgeJoint(
    const std::string& canonicalName) const {
  return contains(canonicalName, "ankle") || contains(canonicalName, "toe") ||
         contains(canonicalName, "wrist") || contains(canonicalName, "hand");
}

float DepthSampler::sampleSingle(const cv::Mat& alignedDepth,
                                 float x,
                                 float y,
                                 float depthUnitToMeter,
                                 int windowSize,
                                 int* validCount) const {
  if (validCount != nullptr) {
    *validCount = 0;
  }
  if (alignedDepth.empty() || alignedDepth.type() != CV_16UC1 ||
      depthUnitToMeter <= 0.0f) {
    return 0.0f;
  }

  const int u = static_cast<int>(std::round(x));
  const int v = static_cast<int>(std::round(y));
  if (u < 0 || u >= alignedDepth.cols || v < 0 || v >= alignedDepth.rows) {
    return 0.0f;
  }

  const int radius = normalizeWindowSize(windowSize) / 2;
  std::vector<uint16_t> values;
  for (int yy = std::max(0, v - radius);
       yy <= std::min(alignedDepth.rows - 1, v + radius); ++yy) {
    const uint16_t* row = alignedDepth.ptr<uint16_t>(yy);
    for (int xx = std::max(0, u - radius);
         xx <= std::min(alignedDepth.cols - 1, u + radius); ++xx) {
      const uint16_t depthMm = rawDepthToMm(row[xx], depthUnitToMeter);
      if (depthMm >= options_.minDepthMm && depthMm <= options_.maxDepthMm) {
        values.push_back(depthMm);
      }
    }
  }
  if (validCount != nullptr) {
    *validCount = static_cast<int>(values.size());
  }
  if (values.empty()) {
    return 0.0f;
  }
  return static_cast<float>(medianOf(&values)) / 1000.0f;
}

Rehab22DepthSamples DepthSampler::sample(const cv::Mat& alignedDepth,
                                         const Rehab22Skeleton2D& joints2d,
                                         float depthUnitToMeter,
                                         int windowSize,
                                         DepthSamplingStats* stats) const {
  Rehab22DepthSamples depths{};
  DepthSamplingStats localStats;
  for (std::size_t i = 0; i < kRehab22JointCount; ++i) {
    if (!joints2d[i].valid || joints2d[i].score < 0.30f) {
      ++localStats.invalidJoints;
      continue;
    }
    int validCount = 0;
    depths[i] = sampleSingle(alignedDepth, joints2d[i].x, joints2d[i].y,
                             depthUnitToMeter, windowSize, &validCount);
    localStats.validCounts[i] = validCount;
    if (depths[i] > 0.0f) {
      ++localStats.validJoints;
    } else {
      ++localStats.invalidJoints;
    }
  }
  if (stats != nullptr) {
    *stats = localStats;
  }
  return depths;
}

}  // namespace rehab

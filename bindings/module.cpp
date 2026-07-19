/*
 * pybind11 bindings for the stroke-rehab C++ engine.
 *
 * Two build modes:
 *   STROKE_ENGINE_STUB=1 → Config + Logger + basic types only
 *   STROKE_ENGINE_STUB=0 → Full engine (capture/sync/align/pose/EMG)
 */

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine/common/Config.h"
#include "engine/util/Logger.h"

#ifndef STROKE_ENGINE_STUB
#include <opencv2/imgcodecs.hpp>
#include "engine/capture/DepthCaptureOpenNI.h"
#include "engine/capture/RgbCaptureV4L2.h"
#include "engine/common/FrameEnvelope.h"
#ifndef STROKE_HARDWARE_ONLY
#include "engine/emg/EmgRpmsgClient.h"
#include "engine/emg/EmgTypes.h"
#include "engine/pose/AdaptiveRoiBoundingBoxProvider.h"
#include "engine/pose/DepthSampler.h"
#include "engine/pose/EMASkeletonFilter.h"
#include "engine/pose/Halpe26ToRehab22Mapper.h"
#include "engine/pose/Halpe26Types.h"
#include "engine/pose/JointProjector3D.h"
#include "engine/pose/PersonDetectorOrt.h"
#include "engine/pose/PoseEstimatorRTMPoseOrt.h"
#include "engine/pose/Rehab22Types.h"
#include "engine/pose/SkeletonSmoother.h"
#include "engine/sync/SyncManager.h"
#endif
#endif

namespace py = pybind11;
using namespace rehab;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

template <typename Func>
static Func wrapCallback(py::object obj) {
  if (obj.is_none()) return nullptr;
  auto fn = obj.cast<py::function>();
  return [fn](auto&&... args) {
    py::gil_scoped_acquire gil;
    fn(std::forward<decltype(args)>(args)...);
  };
}

// ---------------------------------------------------------------------------
// Config struct bindings (available in both modes)
// ---------------------------------------------------------------------------

static void registerConfigBindings(py::module_& m) {
  py::class_<DeviceConfig>(m, "DeviceConfig")
      .def(py::init<>())
      .def_readwrite("openni_device_uri", &DeviceConfig::openniDeviceUri)
      .def_readwrite("rgb_device_path", &DeviceConfig::rgbDevicePath)
      .def_readwrite("rgb_pixel_format", &DeviceConfig::rgbPixelFormat)
      .def_readwrite("rgb_device_index", &DeviceConfig::rgbDeviceIndex)
      .def_readwrite("rgb_width", &DeviceConfig::rgbWidth)
      .def_readwrite("rgb_height", &DeviceConfig::rgbHeight)
      .def_readwrite("rgb_fps", &DeviceConfig::rgbFps)
      .def_readwrite("mirror_rgb_at_capture", &DeviceConfig::mirrorRgbAtCapture)
      .def_readwrite("depth_pixel_format", &DeviceConfig::depthPixelFormat)
      .def_readwrite("depth_width", &DeviceConfig::depthWidth)
      .def_readwrite("depth_height", &DeviceConfig::depthHeight)
      .def_readwrite("depth_fps", &DeviceConfig::depthFps)
      .def_readwrite("enable_hardware_d2c", &DeviceConfig::enableHardwareD2C)
      .def_readwrite("enable_openni_color_stream_for_debug",
                     &DeviceConfig::enableOpenNIColorStreamForDebug)
      .def_readwrite("enable_openni_depth_color_sync",
                     &DeviceConfig::enableOpenNIDepthColorSync)
      .def_readwrite("latest_queue_size", &DeviceConfig::latestQueueSize)
      .def_readwrite("raw_perf_log_interval_sec",
                     &DeviceConfig::rawPerfLogIntervalSec)
      .def_readwrite("enable_cpu_affinity", &DeviceConfig::enableCpuAffinity)
      .def_readwrite("rgb_capture_cpu", &DeviceConfig::rgbCaptureCpu)
      .def_readwrite("depth_capture_cpu", &DeviceConfig::depthCaptureCpu);

  py::class_<SyncConfig>(m, "SyncConfig")
      .def(py::init<>())
      .def_readwrite("match_threshold_ns", &SyncConfig::matchThresholdNs)
      .def_readwrite("queue_size", &SyncConfig::queueSize);

  py::class_<PoseConfig>(m, "PoseConfig")
      .def(py::init<>())
      .def_readwrite("enable_pose", &PoseConfig::enablePose)
      .def_readwrite("enable_adaptive_roi", &PoseConfig::enableAdaptiveRoi)
      .def_readwrite("model_path", &PoseConfig::modelPath)
      .def_readwrite("detector_model_path", &PoseConfig::detectorModelPath)
      .def_readwrite("pipeline_json_path", &PoseConfig::pipelineJsonPath)
      .def_readwrite("detail_json_path", &PoseConfig::detailJsonPath)
      .def_readwrite("deploy_json_path", &PoseConfig::deployJsonPath)
      .def_readwrite("min_score", &PoseConfig::minScore)
      .def_readwrite("max_pair_queue", &PoseConfig::maxPairQueue)
      .def_readwrite("pose_interval", &PoseConfig::poseInterval)
      .def_readwrite("enable_pose_reuse", &PoseConfig::enablePoseReuse)
      .def_readwrite("enable_cpu_affinity", &PoseConfig::enableCpuAffinity)
      .def_readwrite("detector_interval", &PoseConfig::detectorInterval)
      .def_readwrite("roi_margin_ratio", &PoseConfig::roiMarginRatio)
      .def_readwrite("min_track_mean_score", &PoseConfig::minTrackMeanScore)
      .def_readwrite("min_track_valid_points", &PoseConfig::minTrackValidPoints)
      .def_readwrite("max_consecutive_misses", &PoseConfig::maxConsecutiveMisses)
      .def_readwrite("motion_trigger_ratio", &PoseConfig::motionTriggerRatio)
      .def_readwrite("detector_input_size", &PoseConfig::detectorInputSize)
      .def_readwrite("detector_conf_threshold", &PoseConfig::detectorConfThreshold)
      .def_readwrite("detector_nms_threshold", &PoseConfig::detectorNmsThreshold)
      .def_readwrite("enable_smoothing", &PoseConfig::enableSmoothing)
      .def_readwrite("smoothing_alpha", &PoseConfig::smoothingAlpha);

  py::class_<DepthSamplerConfig>(m, "DepthSamplerConfig")
      .def(py::init<>())
      .def_readwrite("min_depth_mm", &DepthSamplerConfig::minDepthMm)
      .def_readwrite("max_depth_mm", &DepthSamplerConfig::maxDepthMm)
      .def_readwrite("body_depth_band_mm", &DepthSamplerConfig::bodyDepthBandMm)
      .def_readwrite("edge_body_depth_band_mm", &DepthSamplerConfig::edgeBodyDepthBandMm)
      .def_readwrite("background_reject_margin_mm", &DepthSamplerConfig::backgroundRejectMarginMm)
      .def_readwrite("background_match_band_mm", &DepthSamplerConfig::backgroundMatchBandMm)
      .def_readwrite("foreground_percentile", &DepthSamplerConfig::foregroundPercentile)
      .def_readwrite("min_foreground_pixels", &DepthSamplerConfig::minForegroundPixels)
      .def_readwrite("hip_radius", &DepthSamplerConfig::hipRadius)
      .def_readwrite("knee_radius", &DepthSamplerConfig::kneeRadius)
      .def_readwrite("ankle_radius", &DepthSamplerConfig::ankleRadius)
      .def_readwrite("toe_radius", &DepthSamplerConfig::toeRadius)
      .def_readwrite("wrist_radius", &DepthSamplerConfig::wristRadius)
      .def_readwrite("default_radius", &DepthSamplerConfig::defaultRadius)
      .def_readwrite("limb_inward_search_enabled", &DepthSamplerConfig::limbInwardSearchEnabled)
      .def_readwrite("limb_inward_steps", &DepthSamplerConfig::limbInwardSteps)
      .def_readwrite("limb_inward_step_px", &DepthSamplerConfig::limbInwardStepPx)
      .def_readwrite("limb_inward_radius", &DepthSamplerConfig::limbInwardRadius);

  py::class_<SkeletonFilterConfig>(m, "SkeletonFilterConfig")
      .def(py::init<>())
      .def_readwrite("mode", &SkeletonFilterConfig::mode)
      .def_readwrite("alpha_good", &SkeletonFilterConfig::alphaGood)
      .def_readwrite("alpha_low_confidence", &SkeletonFilterConfig::alphaLowConfidence)
      .def_readwrite("alpha_recovered", &SkeletonFilterConfig::alphaRecovered)
      .def_readwrite("alpha_invalid", &SkeletonFilterConfig::alphaInvalid)
      .def_readwrite("max_z_jump_m", &SkeletonFilterConfig::maxZJumpM)
      .def_readwrite("max_joint_speed_mps", &SkeletonFilterConfig::maxJointSpeedMps)
      .def_readwrite("hold_last_when_invalid", &SkeletonFilterConfig::holdLastWhenInvalid)
      .def_readwrite("max_hold_frames", &SkeletonFilterConfig::maxHoldFrames);

  py::class_<DebugConfig>(m, "DebugConfig")
      .def(py::init<>())
      .def_readwrite("save_depth_sampling_overlay", &DebugConfig::saveDepthSamplingOverlay)
      .def_readwrite("save_skeleton_raw_csv", &DebugConfig::saveSkeletonRawCsv)
      .def_readwrite("save_skeleton_ema_csv", &DebugConfig::saveSkeletonEmaCsv);

  py::class_<EmgConfig>(m, "EmgConfig")
      .def(py::init<>())
      .def_readwrite("enabled", &EmgConfig::enabled)
      .def_readwrite("mode", &EmgConfig::mode)
      .def_readwrite("capture_backend", &EmgConfig::captureBackend)
      .def_readwrite("serial_device", &EmgConfig::serialDevice)
      .def_readwrite("serial_baud_rate", &EmgConfig::serialBaudRate)
      .def_readwrite("ble_name_prefix", &EmgConfig::bleNamePrefix)
      .def_readwrite("ble_address", &EmgConfig::bleAddress)
      .def_readwrite("sample_rate_hz", &EmgConfig::sampleRateHz)
      .def_readwrite("channel_count", &EmgConfig::channelCount)
      .def_readwrite("raw_chunk_samples", &EmgConfig::rawChunkSamples)
      .def_readwrite("rpmsg_enabled", &EmgConfig::rpmsgEnabled)
      .def_readwrite("rpmsg_ctrl_device", &EmgConfig::rpmsgCtrlDevice)
      .def_readwrite("rpmsg_data_device", &EmgConfig::rpmsgDataDevice)
      .def_readwrite("rpmsg_endpoint_name", &EmgConfig::rpmsgEndpointName)
      .def_readwrite("rpmsg_poll_timeout_ms", &EmgConfig::rpmsgPollTimeoutMs)
      .def_readwrite("active_threshold", &EmgConfig::activeThreshold)
      .def_readwrite("noise_threshold", &EmgConfig::noiseThreshold);

  py::class_<PipelineConfig>(m, "PipelineConfig")
      .def(py::init<>())
      .def_readwrite("device", &PipelineConfig::device)
      .def_readwrite("sync", &PipelineConfig::sync)
      .def_readwrite("pose", &PipelineConfig::pose)
      .def_readwrite("depth_sampler", &PipelineConfig::depthSampler)
      .def_readwrite("skeleton_filter", &PipelineConfig::skeletonFilter)
      .def_readwrite("debug", &PipelineConfig::debug)
      .def_readwrite("emg", &PipelineConfig::emg)
      .def_readwrite("calibration_file", &PipelineConfig::calibrationFile)
      .def_readwrite("record_pairs", &PipelineConfig::recordPairs)
      .def_readwrite("record_path", &PipelineConfig::recordPath);
}

// ---------------------------------------------------------------------------
// Logger bindings (available in both modes)
// ---------------------------------------------------------------------------

static void registerLoggerBindings(py::module_& m) {
  auto logger = m.def_submodule("logger", "Engine logger (callback-based)");
  logger.def(
      "set_callback",
      [](py::object callback) {
        if (callback.is_none()) {
          Logger::setCallback(nullptr);
        } else {
          auto fn = callback.cast<py::function>();
          Logger::setCallback(
              [fn](const std::string& level, const std::string& msg) {
                py::gil_scoped_acquire gil;
                fn(level, msg);
              });
        }
      },
      py::arg("callback"),
      "Set a logging callback: callback(level: str, message: str).\n"
      "Pass None to disable.");
}

// ============================================================================
// Full-mode bindings (excluded in stub build)
// ============================================================================
#ifndef STROKE_ENGINE_STUB

// Capture callbacks transfer Python-owned C-contiguous arrays. The one memcpy
// decouples their lifetime from V4L2/OpenNI buffers before those are reused.
static py::array copyFrameToArray(const FrameEnvelope& env) {
  if (env.source == FrameSource::Rgb) {
    if (env.image.type() != CV_8UC3) {
      throw std::runtime_error("RGB callback requires CV_8UC3 BGR image");
    }
    py::array_t<uint8_t> output({env.height, env.width, 3});
    auto* dst = output.mutable_data();
    const std::size_t rowBytes = static_cast<std::size_t>(env.width) * 3U;
    for (int row = 0; row < env.height; ++row) {
      std::memcpy(dst + static_cast<std::size_t>(row) * rowBytes,
                  env.image.ptr(row), rowBytes);
    }
    return output;
  }
  if (env.image.type() != CV_16UC1) {
    throw std::runtime_error("Depth callback requires CV_16UC1 image");
  }
  py::array_t<uint16_t> output({env.height, env.width});
  auto* dst = reinterpret_cast<uint8_t*>(output.mutable_data());
  const std::size_t rowBytes = static_cast<std::size_t>(env.width) *
                               sizeof(uint16_t);
  for (int row = 0; row < env.height; ++row) {
    std::memcpy(dst + static_cast<std::size_t>(row) * rowBytes,
                env.image.ptr(row), rowBytes);
  }
  return output;
}

static void dispatchArrayFrame(const py::function& fn, FrameEnvelope env,
                               const char* source) {
  py::gil_scoped_acquire gil;
  try {
    py::array image = copyFrameToArray(env);
    fn(image, env.width, env.height, env.syncTsNs, env.frameId,
       env.deviceTsUs, env.depthUnitToMeter, env.pixelFormatName, source,
       env.arrivalTsNs, env.deviceTimeUnit, env.clockQuality,
       env.clockReason, env.clockResetCount);
  } catch (py::error_already_set& error) {
    error.discard_as_unraisable(source);
  }
}

static cv::Mat bgrArrayView(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& image) {
  const auto info = image.request();
  if (info.ndim != 3 || info.shape[2] != 3) {
    throw std::invalid_argument("BGR array must have shape (height, width, 3)");
  }
  return cv::Mat(static_cast<int>(info.shape[0]),
                 static_cast<int>(info.shape[1]), CV_8UC3, info.ptr);
}

static cv::Mat depthArrayView(
    py::array_t<uint16_t, py::array::c_style | py::array::forcecast> image) {
  py::buffer_info info = image.request();
  if (info.ndim != 2) {
    throw std::invalid_argument("Depth array must have shape (height, width)");
  }
  return cv::Mat(static_cast<int>(info.shape[0]),
                 static_cast<int>(info.shape[1]), CV_16UC1, info.ptr);
}

static void registerCaptureBindings(py::module_& m) {
  py::class_<RgbCaptureV4L2>(m, "RgbCaptureV4L2")
      .def(py::init<>())
      .def("start",
           [](RgbCaptureV4L2& self, const DeviceConfig& config,
              py::object callback) {
             if (callback.is_none()) return self.start(config, nullptr);
             auto pyCb = callback.cast<py::function>();
             return self.start(config,
                 [pyCb](FrameEnvelope env) {
                   if (env.image.empty()) return;
                   dispatchArrayFrame(pyCb, std::move(env), "rgb");
                 });
           },
           py::arg("config"), py::arg("frame_callback"))
      .def("stop", &RgbCaptureV4L2::stop,
           py::call_guard<py::gil_scoped_release>())
      .def("is_running", &RgbCaptureV4L2::isRunning)
      .def("set_on_status",
           [](RgbCaptureV4L2& self, py::object callback) {
             self.setOnStatus(
                 wrapCallback<RgbCaptureV4L2::StatusCallback>(callback));
           },
           py::arg("callback"));

  py::class_<DepthCaptureOpenNI>(m, "DepthCaptureOpenNI")
      .def(py::init<>())
      .def("start",
           [](DepthCaptureOpenNI& self, const DeviceConfig& config,
              py::object callback) {
             if (callback.is_none()) return self.start(config, nullptr);
             auto pyCb = callback.cast<py::function>();
             return self.start(config,
                 [pyCb](FrameEnvelope env) {
                   if (env.image.empty()) return;
                   dispatchArrayFrame(pyCb, std::move(env), "depth");
                 });
           },
           py::arg("config"), py::arg("frame_callback"))
      .def("stop", &DepthCaptureOpenNI::stop,
           py::call_guard<py::gil_scoped_release>())
      .def("is_running", &DepthCaptureOpenNI::isRunning)
      .def("real_depth_active", &DepthCaptureOpenNI::realDepthActive)
      .def("hardware_d2c_active", &DepthCaptureOpenNI::hardwareD2CActive);
}

#ifndef STROKE_HARDWARE_ONLY
static void registerSyncBindings(py::module_& m) {
  py::class_<SyncManager>(m, "SyncManager")
      .def(py::init<SyncConfig>(), py::arg("config"))
      .def("set_on_pair_ready",
           [](SyncManager& self, py::object callback) {
             self.setOnPairReady(
                 wrapCallback<SyncManager::PairCallback>(callback));
           },
           py::arg("callback"))
      .def("push_frame", &SyncManager::pushFrame, py::arg("frame"))
      .def("clear", &SyncManager::clear);
}

// ============================================================================
// SyncedCapture — RGB-driven RGB+Depth pipeline
// ============================================================================
//
// RGB frames pace the output at 30 fps.  Depth frames are cached and
// attached to the next RGB frame.  No SyncManager pair-matching bottleneck:
// every RGB frame is delivered immediately even when depth is warming up
// or USB drops a frame, so the UI never flickers.
//
//   RgbCaptureV4L2 ──► every frame encoded + output immediately
//                                          │
//   DepthCaptureOpenNI ──► cached (mutex) ─┘  attached if available

namespace {

using SyncedPairCb = std::function<void(
    FrameEnvelope, std::optional<FrameEnvelope>, int64_t)>;

static SyncedPairCb makeSyncedPairCb(py::object obj) {
  if (obj.is_none()) return nullptr;
  auto fn = obj.cast<py::function>();
  return [fn](FrameEnvelope rgb, std::optional<FrameEnvelope> depth,
              int64_t delta) {
    py::gil_scoped_acquire gil;
    try {
      py::array rgbArray = copyFrameToArray(rgb);
      py::object depthArray = py::none();
      if (depth) depthArray = copyFrameToArray(*depth);
      fn(rgbArray, depthArray, rgb.width, rgb.height,
         depth ? depth->width : 0, depth ? depth->height : 0,
         rgb.syncTsNs, depth ? depth->syncTsNs : 0, delta);
    }
    catch (py::error_already_set&) {}
  };
}

static std::function<void(const std::string&)> makeStatusCb(py::object obj) {
  if (obj.is_none()) return nullptr;
  auto fn = obj.cast<py::function>();
  return [fn](const std::string& msg) {
    py::gil_scoped_acquire gil;
    try { fn(msg); } catch (py::error_already_set&) {}
  };
}

}  // anonymous namespace

class SyncedCapture {
 public:
  SyncedCapture() = default;
  ~SyncedCapture() { stop(); }

  bool start(const DeviceConfig& config, py::object pair_callback) {
    if (running_) return true;

    auto pyCb = makeSyncedPairCb(pair_callback);
    if (!pyCb) return false;

    // ── RGB capture (pacing source) ──
    rgbCapture_ = std::make_unique<RgbCaptureV4L2>();
    if (statusCb_) rgbCapture_->setOnStatus(makeStatusCb(statusCb_));

    if (!rgbCapture_->start(config, [this, pyCb](FrameEnvelope rgbEnv) {
          if (!rgbEnv.valid()) return;

          // ── Attach latest depth frame (best-effort) ──
          std::optional<FrameEnvelope> depth;
          int64_t delta = 0;
          {
            std::lock_guard<std::mutex> lock(depthMutex_);
            if (depthCached_ && depthCached_->valid()) {
              depth = *depthCached_;
              delta = static_cast<int64_t>(rgbEnv.syncTsNs)
                    - static_cast<int64_t>(depthCached_->syncTsNs);
            }
          }
          pyCb(std::move(rgbEnv), std::move(depth), delta);
        })) {
      Logger::warn("SyncedCapture: RGB camera start failed");
    }

    // ── Depth capture (cache only) ──
    depthCapture_ = std::make_unique<DepthCaptureOpenNI>();
    if (!depthCapture_->start(config, [this](FrameEnvelope depthEnv) {
          if (!depthEnv.valid()) return;
          std::lock_guard<std::mutex> lock(depthMutex_);
          depthCached_ = std::make_unique<FrameEnvelope>(std::move(depthEnv));
        })) {
      Logger::warn("SyncedCapture: Depth camera start failed");
    }

    running_ = true;
    return true;
  }

  void stop() {
    if (!running_) return;
    running_ = false;

    if (rgbCapture_) rgbCapture_->stop();
    if (depthCapture_) depthCapture_->stop();

    rgbCapture_.reset();
    depthCapture_.reset();
    {
      std::lock_guard<std::mutex> lock(depthMutex_);
      depthCached_.reset();
    }
  }

  bool is_running() const { return running_; }

  bool hardware_d2c_active() const {
    return depthCapture_ && depthCapture_->hardwareD2CActive();
  }

  void set_on_status(py::object callback) {
    statusCb_ = callback;
    if (rgbCapture_ && !callback.is_none())
      rgbCapture_->setOnStatus(makeStatusCb(callback));
  }

 private:
  std::unique_ptr<RgbCaptureV4L2> rgbCapture_;
  std::unique_ptr<DepthCaptureOpenNI> depthCapture_;
  std::mutex depthMutex_;
  std::unique_ptr<FrameEnvelope> depthCached_;
  py::object statusCb_;
  bool running_{false};
};

static void registerSyncedCaptureBinding(py::module_& m) {
  py::class_<SyncedCapture>(m, "SyncedCapture")
      .def(py::init<>())
      .def("start", &SyncedCapture::start,
           py::arg("config"), py::arg("pair_callback"))
      .def("stop", &SyncedCapture::stop,
           py::call_guard<py::gil_scoped_release>())
      .def("is_running", &SyncedCapture::is_running)
      .def("hardware_d2c_active", &SyncedCapture::hardware_d2c_active)
      .def("set_on_status",
           [](SyncedCapture& self, py::object cb) { self.set_on_status(cb); },
           py::arg("callback"));
}

static void registerPoseBindings(py::module_& m) {
  py::class_<PoseEstimatorConfig>(m, "PoseEstimatorConfig")
      .def(py::init<>())
      .def_readwrite("model_path", &PoseEstimatorConfig::modelPath)
      .def_readwrite("pipeline_json_path", &PoseEstimatorConfig::pipelineJsonPath)
      .def_readwrite("detail_json_path", &PoseEstimatorConfig::detailJsonPath)
      .def_readwrite("deploy_json_path", &PoseEstimatorConfig::deployJsonPath)
      .def_readwrite("min_score", &PoseEstimatorConfig::minScore);

  py::class_<BoundingBox2D>(m, "BoundingBox2D")
      .def(py::init<>())
      .def_readwrite("x", &BoundingBox2D::x)
      .def_readwrite("y", &BoundingBox2D::y)
      .def_readwrite("w", &BoundingBox2D::w)
      .def_readwrite("h", &BoundingBox2D::h)
      .def_readwrite("score", &BoundingBox2D::score)
      .def_readwrite("valid", &BoundingBox2D::valid);

  py::class_<Keypoint2D>(m, "Keypoint2D")
      .def(py::init<>())
      .def_readwrite("x", &Keypoint2D::x)
      .def_readwrite("y", &Keypoint2D::y)
      .def_readwrite("score", &Keypoint2D::score)
      .def_readwrite("valid", &Keypoint2D::valid);

  py::class_<Keypoint3D>(m, "Keypoint3D")
      .def(py::init<>())
      .def_readwrite("x", &Keypoint3D::x)
      .def_readwrite("y", &Keypoint3D::y)
      .def_readwrite("z", &Keypoint3D::z)
      .def_readwrite("score", &Keypoint3D::score)
      .def_readwrite("valid", &Keypoint3D::valid);

  py::class_<PersonDetectorOrt, std::shared_ptr<PersonDetectorOrt>>(
      m, "PersonDetectorOrt")
      .def(py::init<>())
      .def("initialize",
           [](PersonDetectorOrt& self, const std::string& modelPath,
              int inputSize, float confThresh, float nmsThresh) {
             PersonDetectorConfig cfg;
             cfg.modelPath = modelPath;
             cfg.inputSize = inputSize;
             cfg.confThreshold = confThresh;
             cfg.nmsThreshold = nmsThresh;
             return self.initialize(cfg);
           },
           py::arg("model_path"), py::arg("input_size") = 320,
           py::arg("conf_threshold") = 0.35f,
           py::arg("nms_threshold") = 0.45f)
      .def("is_initialized", &PersonDetectorOrt::isInitialized)
      .def("detect_largest_person_bgr",
           [](PersonDetectorOrt& self,
              py::array_t<uint8_t, py::array::c_style |
                                      py::array::forcecast> image) -> py::object {
             cv::Mat bgr = bgrArrayView(image);
             auto box = self.detectLargestPerson(bgr);
             if (!box.valid) return py::none();
             return py::cast(box);
           },
           py::arg("bgr"));

  py::class_<PoseInferenceResult>(m, "PoseInferenceResult")
      .def(py::init<>())
      .def_readonly("keypoints", &PoseInferenceResult::keypoints)
      .def_readonly("used_box", &PoseInferenceResult::usedBox)
      .def_readonly("mean_score", &PoseInferenceResult::meanScore)
      .def_readonly("valid_count", &PoseInferenceResult::validCount)
      .def_readonly("bbox_ms", &PoseInferenceResult::bboxMs)
      .def_readonly("pose_ms", &PoseInferenceResult::poseMs)
      .def_readonly("model_loaded", &PoseInferenceResult::modelLoaded);

  py::class_<PoseEstimatorRTMPoseOrt>(m, "PoseEstimatorRTMPoseOrt")
      .def(py::init<>())
      .def("initialize", &PoseEstimatorRTMPoseOrt::initialize, py::arg("config"))
      .def("is_initialized", &PoseEstimatorRTMPoseOrt::isInitialized)
      .def("infer_bgr",
           [](PoseEstimatorRTMPoseOrt& self,
              py::array_t<uint8_t, py::array::c_style |
                                      py::array::forcecast> image) {
             return self.infer(bgrArrayView(image));
           },
           py::arg("bgr"))
      .def("set_bounding_box_provider_fallback",
           [](PoseEstimatorRTMPoseOrt& self, BoundingBox2D box) {
             // Create a simple fallback provider from a fixed box
             struct FixedBoxProvider : public BoundingBoxProvider {
               BoundingBox2D box_;
               FixedBoxProvider(BoundingBox2D b) : box_(b) {}
               BoundingBox2D getPrimaryBox(const cv::Mat&) override {
                 auto out = box_; out.valid = true; return out;
               }
               void updateFromPose(const BoundingBox2D&, const Halpe26Skeleton2D&) override {}
               void reset() override {}
               std::string debugState() const override { return "fixed"; }
             };
             self.setBoundingBoxProvider(std::make_shared<FixedBoxProvider>(box));
           },
           py::arg("box"),
           "Set a fixed ROI box. Call before infer_bgr().");

  py::class_<Halpe26ToRehab22Mapper>(m, "Halpe26ToRehab22Mapper")
      .def(py::init<>())
      // ── P0: Halpe26 → Rehab22 mapping ──
      .def("map", &Halpe26ToRehab22Mapper::map, py::arg("halpe26"),
           "Map Halpe26 keypoints to Rehab22 joint set.");

  py::class_<DepthSampler>(m, "DepthSampler")
      .def(py::init<>())
      .def("sample_array",
           [](DepthSampler& self,
              py::array_t<uint16_t, py::array::c_style |
                                           py::array::forcecast> depth,
              const Rehab22Skeleton2D& joints2d,
              float depthUnitToMeter, int windowSize) -> std::array<float, 22> {
             return self.sample(depthArrayView(depth), joints2d,
                                depthUnitToMeter, windowSize);
           },
           py::arg("depth"), py::arg("joints_2d"),
           py::arg("depth_unit_to_meter") = 0.001f,
           py::arg("window_size") = 7,
           "Sample a uint16 depth array at Rehab22 joint positions.");

  py::class_<JointProjector3D>(m, "JointProjector3D")
      .def(py::init<>())
      .def("set_intrinsics",
           [](JointProjector3D& self, float fx, float fy, float cx, float cy) {
             CameraIntrinsics in;
             in.fx = fx; in.fy = fy; in.cx = cx; in.cy = cy;
             self.setIntrinsics(in);
           },
           py::arg("fx"), py::arg("fy"), py::arg("cx"), py::arg("cy"))
      .def("intrinsics_valid", &JointProjector3D::intrinsicsValid)
      .def("project",
           [](JointProjector3D& self,
              const Rehab22Skeleton2D& joints2d,
              const std::array<float, 22>& depthsMeters) -> Rehab22Skeleton3D {
             return self.project(joints2d, depthsMeters);
           },
           py::arg("joints_2d"), py::arg("depths_meters"),
           "Project Rehab22 2D joints + depth → 3D skeleton (meters, camera frame).");

  py::class_<EMASkeletonFilter>(m, "EMASkeletonFilter")
      .def(py::init<>())
      .def("reset", &EMASkeletonFilter::reset, py::arg("reason") = "")
      .def("filter",
           [](EMASkeletonFilter& self,
              const Rehab22Skeleton3D& raw, double dt) -> Rehab22Skeleton3D {
             return self.filter(raw, dt);
           },
           py::arg("raw"), py::arg("dt_seconds"),
           "EMA filter one frame of Rehab22 3D keypoints.");

  py::class_<SkeletonSmoother>(m, "SkeletonSmoother")
      .def(py::init<>())
      .def("reset", &SkeletonSmoother::reset)
      .def("smooth",
           [](SkeletonSmoother& self,
              const Rehab22Skeleton3D& input) -> Rehab22Skeleton3D {
             return self.smooth(input);
           },
           py::arg("input"),
           "Smooth one frame of Rehab22 3D keypoints.");
}

static void registerEmgBindings(py::module_& m) {
  py::enum_<EmgMuscleState>(m, "EmgMuscleState")
      .value("REST", EmgMuscleState::Rest)
      .value("SMOOTH_FLEX", EmgMuscleState::SmoothFlex)
      .value("TREMOR", EmgMuscleState::Tremor)
      .value("FATIGUE", EmgMuscleState::Fatigue)
      .export_values();

  py::class_<EmgRawSample>(m, "EmgRawSample")
      .def(py::init<>())
      .def_readwrite("host_ts_ns", &EmgRawSample::hostTsNs)
      .def_readwrite("seq", &EmgRawSample::seq)
      .def_readwrite("channels", &EmgRawSample::channels);

  py::class_<EmgFeatureFrame>(m, "EmgFeatureFrame")
      .def(py::init<>())
      .def_readwrite("host_ts_ns", &EmgFeatureFrame::hostTsNs)
      .def_readwrite("seq", &EmgFeatureFrame::seq)
      .def_readwrite("sample_rate_hz", &EmgFeatureFrame::sampleRateHz)
      .def_readwrite("channels", &EmgFeatureFrame::channels)
      .def("valid", &EmgFeatureFrame::valid);

  py::class_<EmgRpmsgClient>(m, "EmgRpmsgClient")
      .def(py::init<>())
      .def("configure", &EmgRpmsgClient::configure, py::arg("config"))
      .def("connect", &EmgRpmsgClient::connect)
      .def("close", &EmgRpmsgClient::close)
      .def("is_connected", &EmgRpmsgClient::isConnected)
      .def("set_on_feature",
           [](EmgRpmsgClient& self, py::object callback) {
             self.setOnFeature(
                 wrapCallback<EmgRpmsgClient::FeatureCallback>(callback));
           },
           py::arg("callback"))
      .def("set_on_status",
           [](EmgRpmsgClient& self, py::object callback) {
             self.setOnStatus(
                 wrapCallback<EmgRpmsgClient::StatusCallback>(callback));
           },
           py::arg("callback"));
}
#endif  // !STROKE_HARDWARE_ONLY

#endif  // !STROKE_ENGINE_STUB

// ============================================================================
// Module entry
// ============================================================================

PYBIND11_MODULE(_core, m) {
  m.doc() = R"pbdoc(
    Stroke Rehab C++ Engine
    -----------------------
    Low-level hardware drivers for the Python-main stroke rehabilitation
    training system. Legacy native algorithms are compatibility-only.
  )pbdoc";

  registerConfigBindings(m);
  registerLoggerBindings(m);

#ifndef STROKE_ENGINE_STUB
  registerCaptureBindings(m);
#ifndef STROKE_HARDWARE_ONLY
  registerSyncBindings(m);
  registerSyncedCaptureBinding(m);
  registerPoseBindings(m);
  registerEmgBindings(m);
#endif
#endif

  m.attr("__version__") = "0.1.0";

#ifdef STROKE_ENGINE_STUB
  m.attr("_stub_mode") = true;
#else
  m.attr("_stub_mode") = false;
#endif
}

/*
 * pybind11 bindings for the stroke-rehab C++ engine.
 *
 * Two build modes:
 *   STROKE_ENGINE_STUB=1 → Config + Logger + basic types only
 *   STROKE_ENGINE_STUB=0 → Full engine (capture/sync/align/pose/EMG)
 */

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "engine/common/Config.h"
#include "engine/util/Logger.h"

#ifndef STROKE_ENGINE_STUB
#include <opencv2/imgcodecs.hpp>
#include "engine/capture/DepthCaptureOpenNI.h"
#include "engine/capture/RgbCaptureV4L2.h"
#include "engine/common/FrameEnvelope.h"
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

// C++ capture threads call back into Python with JPEG-encoded RGB frames.
// This avoids binding cv::Mat / FrameEnvelope — no OpenCV Python needed on the C++ side.
// Python callback signature: callback(
//     jpeg_bytes: bytes, width: int, height: int,
//     ts_ns: int, frame_id: int, source: str
// )
using JpegFrameCallback = std::function<void(py::bytes, int, int, uint64_t, uint64_t, std::string)>;

static JpegFrameCallback wrapJpegCallback(py::object obj) {
  if (obj.is_none()) return nullptr;
  auto fn = obj.cast<py::function>();
  return [fn](py::bytes jpeg, int w, int h, uint64_t ts, uint64_t fid, std::string src) {
    py::gil_scoped_acquire gil;
    try { fn(jpeg, w, h, ts, fid, src); } catch (py::error_already_set&) {}
  };
}

static void registerCaptureBindings(py::module_& m) {
  py::class_<RgbCaptureV4L2>(m, "RgbCaptureV4L2")
      .def(py::init<>())
      .def("start",
           [](RgbCaptureV4L2& self, const DeviceConfig& config,
              py::object callback) {
             auto pyCb = wrapJpegCallback(callback);
             return self.start(config,
                 [pyCb](FrameEnvelope env) {
                   if (!pyCb || env.image.empty()) return;
                   // JPEG encode: quality 85 balances bandwidth and CPU
                   std::vector<uchar> buf;
                   std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 85};
                   cv::imencode(".jpg", env.image, buf, params);
                   py::bytes jpeg(reinterpret_cast<const char*>(buf.data()), buf.size());
                   pyCb(jpeg, env.width, env.height, env.hostTsNs, env.frameId, "rgb");
                 });
           },
           py::arg("config"), py::arg("frame_callback"))
      .def("stop", &RgbCaptureV4L2::stop)
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
             auto pyCb = wrapJpegCallback(callback);
             return self.start(config,
                 [pyCb](FrameEnvelope env) {
                   if (!pyCb || env.image.empty()) return;
                   // Depth is 16-bit mm — encode as 16-bit PNG to avoid loss
                   std::vector<uchar> buf;
                   std::vector<int> params = {cv::IMWRITE_PNG_COMPRESSION, 1};
                   cv::imencode(".png", env.image, buf, params);
                   py::bytes png(reinterpret_cast<const char*>(buf.data()), buf.size());
                   pyCb(png, env.width, env.height, env.hostTsNs, env.frameId, "depth");
                 });
           },
           py::arg("config"), py::arg("frame_callback"))
      .def("stop", &DepthCaptureOpenNI::stop)
      .def("is_running", &DepthCaptureOpenNI::isRunning)
      .def("hardware_d2c_active", &DepthCaptureOpenNI::hardwareD2CActive);
}

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
      .def("is_initialized", &PersonDetectorOrt::isInitialized);

  py::class_<PoseEstimatorRTMPoseOrt>(m, "PoseEstimatorRTMPoseOrt")
      .def(py::init<>())
      .def("initialize", &PoseEstimatorRTMPoseOrt::initialize, py::arg("config"))
      .def("is_initialized", &PoseEstimatorRTMPoseOrt::isInitialized);

  py::class_<Halpe26ToRehab22Mapper>(m, "Halpe26ToRehab22Mapper").def(py::init<>());
  py::class_<JointProjector3D>(m, "JointProjector3D").def(py::init<>());
  py::class_<EMASkeletonFilter>(m, "EMASkeletonFilter").def(py::init<>());
  py::class_<SkeletonSmoother>(m, "SkeletonSmoother").def(py::init<>());
  py::class_<DepthSampler>(m, "DepthSampler").def(py::init<>());
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

#endif  // !STROKE_ENGINE_STUB

// ============================================================================
// Module entry
// ============================================================================

PYBIND11_MODULE(rehab_engine, m) {
  m.doc() = R"pbdoc(
    Stroke Rehab C++ Engine
    -----------------------
    Low-level hardware drivers and AI inference engine for the
    stroke rehabilitation training system.
  )pbdoc";

  registerConfigBindings(m);
  registerLoggerBindings(m);

#ifndef STROKE_ENGINE_STUB
  registerCaptureBindings(m);
  registerSyncBindings(m);
  registerPoseBindings(m);
  registerEmgBindings(m);
#endif

  m.attr("__version__") = "0.1.0";

#ifdef STROKE_ENGINE_STUB
  m.attr("_stub_mode") = true;
#else
  m.attr("_stub_mode") = false;
#endif
}
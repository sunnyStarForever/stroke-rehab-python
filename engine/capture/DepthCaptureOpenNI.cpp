#include "engine/capture/DepthCaptureOpenNI.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <string>
#include <thread>

#include <opencv2/core.hpp>

#include "engine/common/Timestamp.h"
#include "engine/util/Logger.h"
#include "engine/util/ThreadAffinity.h"

#ifdef HAVE_OPENNI2
#include <OpenNI.h>
#endif

namespace rehab {

namespace {

double nsToMs(uint64_t ns) {
  return static_cast<double>(ns) / 1000000.0;
}

double nsToSec(uint64_t ns) {
  return static_cast<double>(ns) / 1000000000.0;
}

#ifdef HAVE_OPENNI2
float depthUnitToMeter(openni::PixelFormat format) {
  if (format == openni::PIXEL_FORMAT_DEPTH_100_UM) {
    return 0.0001f;
  }
  return 0.001f;
}

std::string pixelFormatName(openni::PixelFormat format) {
  switch (format) {
    case openni::PIXEL_FORMAT_DEPTH_1_MM:
      return "DEPTH_1_MM";
    case openni::PIXEL_FORMAT_DEPTH_100_UM:
      return "DEPTH_100_UM";
    default:
      return "DEPTH_UNKNOWN";
  }
}
#endif

double sampledZeroRatePercent(const cv::Mat& depth, int stride) {
  if (depth.empty() || depth.type() != CV_16UC1) {
    return 0.0;
  }
  const int step = std::max(1, stride);
  uint64_t zeros = 0;
  uint64_t total = 0;
  for (int y = 0; y < depth.rows; y += step) {
    const uint16_t* row = depth.ptr<uint16_t>(y);
    for (int x = 0; x < depth.cols; x += step) {
      ++total;
      if (row[x] == 0) {
        ++zeros;
      }
    }
  }
  return total > 0 ? (100.0 * static_cast<double>(zeros) /
                      static_cast<double>(total))
                   : 0.0;
}

void copyDepthFrameToMat(const void* src,
                         int width,
                         int height,
                         int strideBytes,
                         cv::Mat* out) {
  out->create(height, width, CV_16UC1);
  const std::size_t rowBytes = static_cast<std::size_t>(width) *
                               sizeof(uint16_t);
  if (strideBytes == static_cast<int>(rowBytes)) {
    std::memcpy(out->data, src, rowBytes * static_cast<std::size_t>(height));
    return;
  }
  const auto* srcBytes = static_cast<const uint8_t*>(src);
  for (int y = 0; y < height; ++y) {
    std::memcpy(out->ptr(y), srcBytes + static_cast<std::size_t>(y) *
                                      static_cast<std::size_t>(strideBytes),
                rowBytes);
  }
}

}  // namespace

DepthCaptureOpenNI::~DepthCaptureOpenNI() {
  stop();
}

bool DepthCaptureOpenNI::start(const DeviceConfig& config,
                               FrameCallback callback) {
  if (running_.load()) {
    return true;
  }

  config_ = config;
  callback_ = std::move(callback);
  tsNormalizer_.reset();
  running_.store(true);
  hardwareD2CActive_.store(false);
  realDepthActive_.store(false);
  worker_ = std::thread(&DepthCaptureOpenNI::run, this);
  return true;
}

void DepthCaptureOpenNI::stop() {
  running_.store(false);
  if (worker_.joinable()) {
    worker_.join();
  }
  realDepthActive_.store(false);
}

void DepthCaptureOpenNI::setQueueDropCounter(
    std::function<uint64_t()> counter) {
  queueDropCounter_ = std::move(counter);
}

void DepthCaptureOpenNI::run() {
  if (config_.enableCpuAffinity) {
    bindCurrentThreadToCpu(config_.depthCaptureCpu, "depth_capture");
  }

#ifdef HAVE_OPENNI2
  using namespace openni;

  if (OpenNI::initialize() != STATUS_OK) {
    Logger::warn(std::string("OpenNI2 init failed: ") +
                 OpenNI::getExtendedError() +
                 ". real depth is unavailable; synthetic fallback disabled.");
    running_.store(false);
    return;
  }

  Device device;
  const char* uri = config_.openniDeviceUri.empty()
                        ? ANY_DEVICE
                        : config_.openniDeviceUri.c_str();
  if (device.open(uri) != STATUS_OK) {
    Logger::warn(std::string("OpenNI2 device open failed: ") +
                 OpenNI::getExtendedError() +
                 ". real depth is unavailable; synthetic fallback disabled.");
    OpenNI::shutdown();
    running_.store(false);
    return;
  }

  VideoStream depthStream;
  if (depthStream.create(device, SENSOR_DEPTH) != STATUS_OK) {
    Logger::warn(std::string("OpenNI2 depth stream create failed: ") +
                 OpenNI::getExtendedError() +
                 ". real depth is unavailable; synthetic fallback disabled.");
    depthStream.destroy();
    device.close();
    OpenNI::shutdown();
    running_.store(false);
    return;
  }

  VideoMode depthMode = depthStream.getVideoMode();
  if (config_.depthWidth > 0 && config_.depthHeight > 0) {
    depthMode.setResolution(config_.depthWidth, config_.depthHeight);
  }
  if (config_.depthFps > 0) {
    depthMode.setFps(config_.depthFps);
  }
  if (config_.depthPixelFormat == "DEPTH_100_UM") {
    depthMode.setPixelFormat(PIXEL_FORMAT_DEPTH_100_UM);
  } else {
    depthMode.setPixelFormat(PIXEL_FORMAT_DEPTH_1_MM);
  }
  if (depthStream.setVideoMode(depthMode) != STATUS_OK) {
    Logger::warn(std::string("OpenNI2 depth video mode not accepted: ") +
                 OpenNI::getExtendedError());
  }

  VideoStream colorStream;
  bool colorStreamCreated = false;
  bool colorStreamStarted = false;
  if (config_.enableOpenNIColorStreamForDebug) {
    if (device.getSensorInfo(SENSOR_COLOR) != nullptr &&
        colorStream.create(device, SENSOR_COLOR) == STATUS_OK) {
      colorStreamCreated = true;
      if (colorStream.start() == STATUS_OK) {
        colorStreamStarted = true;
      } else {
        Logger::warn("[DEPTH INIT] OpenNI color stream debug start failed");
        colorStream.destroy();
        colorStreamCreated = false;
      }
    }
  } else {
    Logger::info(
        "[DEPTH INIT] OpenNI color stream disabled, RGB is handled by V4L2");
  }

  Logger::info(std::string("[DEPTH INIT] hardware D2C requested=") +
               (config_.enableHardwareD2C ? "true" : "false"));
  if (config_.enableHardwareD2C) {
    const bool registrationOk =
        device.setImageRegistrationMode(IMAGE_REGISTRATION_DEPTH_TO_COLOR) ==
        STATUS_OK;
    hardwareD2CActive_.store(registrationOk);
    if (!registrationOk) {
      Logger::warn("Hardware D2C not available, software aligner will be used.");
    }
  }
  Logger::info(std::string("[DEPTH INIT] hardware D2C active=") +
               (hardwareD2CActive_.load() ? "true" : "false"));

  if (config_.enableOpenNIDepthColorSync && colorStreamStarted) {
    const bool syncOk = device.setDepthColorSyncEnabled(true) == STATUS_OK;
    if (!syncOk) {
      Logger::warn("[DEPTH INIT] OpenNI depth-color sync requested but failed");
    }
  } else {
    Logger::info(
        "[DEPTH INIT] RGB uses V4L2/UVC, skip OpenNI depth-color sync");
  }

  if (depthStream.start() != STATUS_OK) {
    Logger::warn(std::string("OpenNI2 depth stream start failed: ") +
                 OpenNI::getExtendedError() +
                 ". real depth is unavailable; synthetic fallback disabled.");
    if (colorStreamStarted) {
      colorStream.stop();
    }
    if (colorStreamCreated) {
      colorStream.destroy();
    }
    depthStream.destroy();
    device.close();
    OpenNI::shutdown();
    running_.store(false);
    return;
  }

  realDepthActive_.store(true);

  const VideoMode actualMode = depthStream.getVideoMode();
  const std::string modeName =
      std::to_string(actualMode.getResolutionX()) + "x" +
      std::to_string(actualMode.getResolutionY()) + "@" +
      std::to_string(actualMode.getFps());
  const std::string unitName =
      actualMode.getPixelFormat() == PIXEL_FORMAT_DEPTH_100_UM ? "0.1mm"
                                                               : "1mm";

  uint64_t frameId = 0;
  uint64_t lastLogNs = monotonicRawNowNs();
  uint64_t lastHostTsNs = 0;
  uint64_t lastDeviceTsUs = 0;
  uint64_t framesSinceLog = 0;
  double sumWaitMs = 0.0;
  double sumReadMs = 0.0;
  double sumCopyMs = 0.0;
  double sumCbMs = 0.0;
  double maxCbMs = 0.0;
  double sumHostDeltaMs = 0.0;
  double sumOpenNiDeltaMs = 0.0;
  uint64_t deltaCount = 0;
  double lastZeroRate = 0.0;

  while (running_.load()) {
    int changedIndex = -1;
    VideoStream* streams[] = {&depthStream};

    const uint64_t waitStartNs = monotonicRawNowNs();
    if (OpenNI::waitForAnyStream(streams, 1, &changedIndex, 2000) !=
        STATUS_OK) {
      continue;
    }
    const uint64_t waitEndNs = monotonicRawNowNs();

    VideoFrameRef frame;
    const uint64_t readStartNs = waitEndNs;
    if (depthStream.readFrame(&frame) != STATUS_OK || !frame.isValid()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
      continue;
    }
    const uint64_t readEndNs = monotonicRawNowNs();

    const PixelFormat pixelFormat = frame.getVideoMode().getPixelFormat();
    const float unitToMeter = depthUnitToMeter(pixelFormat);
    const std::string framePixelFormatName = pixelFormatName(pixelFormat);

    cv::Mat depthCopy;
    const uint64_t copyStartNs = readEndNs;
    copyDepthFrameToMat(frame.getData(), frame.getWidth(), frame.getHeight(),
                        frame.getStrideInBytes(), &depthCopy);
    const uint64_t copyEndNs = monotonicRawNowNs();

    const uint64_t hostTsNs = copyEndNs;
    const uint64_t deviceTsUs =
        static_cast<uint64_t>(frame.getTimestamp());

    FrameEnvelope envelope;
    envelope.source = FrameSource::Depth;
    tsNormalizer_.stampDeviceMicroseconds(envelope, hostTsNs, deviceTsUs);
    envelope.frameId = frameId++;
    envelope.width = frame.getWidth();
    envelope.height = frame.getHeight();
    envelope.image = std::move(depthCopy);
    envelope.depthUnitToMeter = unitToMeter;
    envelope.pixelFormatName = framePixelFormatName;

    if (envelope.frameId == 0) {
      std::ostringstream trace;
      trace << "[DEPTH TS] device_us=" << deviceTsUs
            << " arrival_ns=" << hostTsNs
            << " sync_ns=" << envelope.syncTsNs
            << " quality=" << envelope.clockQuality
            << " unit=" << framePixelFormatName;
      Logger::info(trace.str());
    }

    if ((envelope.frameId % 30ULL) == 0ULL) {
      lastZeroRate = sampledZeroRatePercent(envelope.image, 16);
    }

    const uint64_t cbStartNs = monotonicRawNowNs();
    if (callback_) {
      callback_(std::move(envelope));
    }
    const uint64_t cbEndNs = monotonicRawNowNs();
    const double cbMs = nsToMs(cbEndNs - cbStartNs);
    sumWaitMs += nsToMs(waitEndNs - waitStartNs);
    sumReadMs += nsToMs(readEndNs - readStartNs);
    sumCopyMs += nsToMs(copyEndNs - copyStartNs);
    sumCbMs += cbMs;
    maxCbMs = std::max(maxCbMs, cbMs);
    ++framesSinceLog;

    if (lastHostTsNs > 0 && hostTsNs > lastHostTsNs) {
      sumHostDeltaMs += nsToMs(hostTsNs - lastHostTsNs);
      sumOpenNiDeltaMs +=
          static_cast<double>(deviceTsUs - lastDeviceTsUs) / 1000.0;
      ++deltaCount;
    }
    lastHostTsNs = hostTsNs;
    lastDeviceTsUs = deviceTsUs;

    const uint64_t nowNs = monotonicRawNowNs();
    const double intervalSec =
        std::max(0.1, config_.rawPerfLogIntervalSec);
    if (nowNs - lastLogNs >=
        static_cast<uint64_t>(intervalSec * 1000000000.0)) {
      const double elapsedSec = nsToSec(nowNs - lastLogNs);
      const double fps = elapsedSec > 0.0
                             ? static_cast<double>(framesSinceLog) / elapsedSec
                             : 0.0;
      const double denom = framesSinceLog > 0
                               ? static_cast<double>(framesSinceLog)
                               : 1.0;
      const double deltaDenom = deltaCount > 0
                                    ? static_cast<double>(deltaCount)
                                    : 1.0;
      const uint64_t queueDrop =
          queueDropCounter_ ? queueDropCounter_() : 0ULL;

      std::ostringstream oss;
      oss << std::fixed << std::setprecision(1)
          << "[DEPTH RAW] fps=" << fps << " mode=" << modeName
          << " unit=" << unitName << " wait=" << (sumWaitMs / denom)
          << "ms read=" << (sumReadMs / denom)
          << "ms copy=" << (sumCopyMs / denom)
          << "ms cb=" << (sumCbMs / denom)
          << "ms cb_max=" << maxCbMs
          << "ms queue_drop=" << queueDrop
          << " openni_ts_delta=" << (sumOpenNiDeltaMs / deltaDenom)
          << "ms host_delta=" << (sumHostDeltaMs / deltaDenom)
          << "ms zero_rate=" << lastZeroRate << "%";
      Logger::info(oss.str());

      lastLogNs = nowNs;
      framesSinceLog = 0;
      sumWaitMs = 0.0;
      sumReadMs = 0.0;
      sumCopyMs = 0.0;
      sumCbMs = 0.0;
      maxCbMs = 0.0;
      sumHostDeltaMs = 0.0;
      sumOpenNiDeltaMs = 0.0;
      deltaCount = 0;
    }
  }

  depthStream.stop();
  depthStream.destroy();
  if (colorStreamStarted) {
    colorStream.stop();
  }
  if (colorStreamCreated) {
    colorStream.destroy();
  }
  device.close();
  OpenNI::shutdown();
  realDepthActive_.store(false);
#else
  Logger::warn(
      "Built without OpenNI2; real depth is unavailable and synthetic "
      "fallback is disabled.");
  running_.store(false);
#endif
}

}  // namespace rehab

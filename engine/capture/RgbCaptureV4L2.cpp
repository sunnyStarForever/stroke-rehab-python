/*
 * 模块作用：
 * 本文件负责从 V4L2 摄像头采集 RGB 图像，并为每一帧附加统一的主机时间戳。
 * RGB 走 V4L2 是因为普通 UVC 摄像头在 Linux 下由 V4L2 暴露，
 * 可直接设置 MJPG/YUYV、分辨率和 fps，并可用 mmap 缓冲减少拷贝。
 */
#include "engine/capture/RgbCaptureV4L2.h"

#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <poll.h>
#include <string>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

#include <linux/videodev2.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "engine/common/Timestamp.h"
#include "engine/util/FpsCounter.h"
#include "engine/util/Logger.h"
#include "engine/util/ThreadAffinity.h"

namespace rehab {

namespace {

struct MappedBuffer {
  void* start{nullptr};
  size_t length{0};
};

// ioctl 可能被信号中断，循环重试可以避免采集线程因为 EINTR 误判设备失败。
int xioctl(int fd, unsigned long request, void* arg) {
  int result;
  do {
    result = ioctl(fd, request, arg);
  } while (result == -1 && errno == EINTR);
  return result;
}

std::string fourccToString(uint32_t fourcc) {
  char code[5] = {'\0', '\0', '\0', '\0', '\0'};
  code[0] = static_cast<char>(fourcc & 0xFF);
  code[1] = static_cast<char>((fourcc >> 8) & 0xFF);
  code[2] = static_cast<char>((fourcc >> 16) & 0xFF);
  code[3] = static_cast<char>((fourcc >> 24) & 0xFF);
  return std::string(code);
}

uint32_t pixelFormatFromString(const std::string& format) {
  // MJPG 压缩率高、USB 带宽压力小；YUYV 未压缩、CPU 转换简单但带宽更大。
  if (format == "YUYV" || format == "YUY2" || format == "yuyv") {
    return V4L2_PIX_FMT_YUYV;
  }
  return V4L2_PIX_FMT_MJPEG;
}

std::string normalizeFormatString(const std::string& format) {
  if (format == "YUYV" || format == "YUY2" || format == "yuyv") {
    return "YUYV";
  }
  return "MJPG";
}

std::string devicePath(const DeviceConfig& config) {
  if (!config.rgbDevicePath.empty()) {
    return config.rgbDevicePath;
  }
  return "/dev/video" + std::to_string(config.rgbDeviceIndex);
}

}  // namespace

RgbCaptureV4L2::~RgbCaptureV4L2() {
  stop();
}

void RgbCaptureV4L2::setOnStatus(StatusCallback callback) {
  std::lock_guard<std::mutex> lock(statusMutex_);
  statusCallback_ = std::move(callback);
}

void RgbCaptureV4L2::emitStatus(const std::string& status) {
  StatusCallback callback;
  {
    std::lock_guard<std::mutex> lock(statusMutex_);
    callback = statusCallback_;
  }
  if (callback) {
    callback(status);
  }
}

bool RgbCaptureV4L2::start(const DeviceConfig& config, FrameCallback callback) {
  if (running_.load()) {
    return true;
  }

  config_ = config;
  callback_ = std::move(callback);
  tsNormalizer_.reset();
  lastV4l2SyncTsNs_ = 0;
  running_.store(true);
  worker_ = std::thread(&RgbCaptureV4L2::run, this);
  return true;
}

void RgbCaptureV4L2::stop() {
  if (!running_.exchange(false)) {
    return;
  }
  if (worker_.joinable()) {
    worker_.join();
  }
}

void RgbCaptureV4L2::run() {
  /*
   * run()
   * 流程：
   * - 打开 V4L2 设备并设置格式、分辨率、fps；
   * - 申请 mmap 环形缓冲，避免用户态反复分配大块图像内存；
   * - poll 等待新帧，DQBUF 取出最新缓冲，解码/转换为 BGR；
   * - 用采集前后时间中点作为 host_ts_ns，交给同步层匹配 Depth。
   */
  if (config_.enableCpuAffinity) {
    bindCurrentThreadToCpu(config_.rgbCaptureCpu, "rgb_capture");
  }

  const std::string path = devicePath(config_);
  const uint32_t requestedFormat = pixelFormatFromString(config_.rgbPixelFormat);
  const uint32_t requestedWidth = static_cast<uint32_t>(config_.rgbWidth);
  const uint32_t requestedHeight = static_cast<uint32_t>(config_.rgbHeight);
  const uint32_t requestedFps = static_cast<uint32_t>(config_.rgbFps);

  int fd = open(path.c_str(), O_RDWR | O_NONBLOCK, 0);
  if (fd < 0) {
    emitStatus("[RGB OPEN] open failed device=" + path + " error=" + std::strerror(errno));
    Logger::warn("V4L2 camera open failed " + path + ": " + std::strerror(errno));
    runFallback();
    return;
  }

  v4l2_capability cap{};
  if (xioctl(fd, VIDIOC_QUERYCAP, &cap) < 0) {
    emitStatus("[RGB OPEN] querycap failed device=" + path + " error=" + std::strerror(errno));
    close(fd);
    runFallback();
    return;
  }

  if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
    emitStatus("[RGB OPEN] device not video capture " + path);
    close(fd);
    runFallback();
    return;
  }

  v4l2_format fmt{};
  fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  fmt.fmt.pix.width = requestedWidth;
  fmt.fmt.pix.height = requestedHeight;
  fmt.fmt.pix.pixelformat = requestedFormat;
  fmt.fmt.pix.field = V4L2_FIELD_ANY;

  if (xioctl(fd, VIDIOC_S_FMT, &fmt) < 0) {
    emitStatus("[RGB OPEN] set format failed device=" + path + " fmt=" + normalizeFormatString(config_.rgbPixelFormat) +
               " error=" + std::strerror(errno));
    close(fd);
    runFallback();
    return;
  }

  v4l2_streamparm parm{};
  parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  parm.parm.capture.timeperframe.numerator = 1;
  parm.parm.capture.timeperframe.denominator = requestedFps;
  // timeperframe=1/fps 是对驱动的帧率请求，实际 fps 仍以驱动返回值为准。
  xioctl(fd, VIDIOC_S_PARM, &parm);

  v4l2_requestbuffers req{};
  req.count = 4;
  req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  req.memory = V4L2_MEMORY_MMAP;
  if (xioctl(fd, VIDIOC_REQBUFS, &req) < 0 || req.count < 2) {
    emitStatus("[RGB OPEN] request buffers failed device=" + path + " error=" + std::strerror(errno));
    close(fd);
    runFallback();
    return;
  }

  std::vector<MappedBuffer> buffers(req.count);
  for (uint32_t i = 0; i < req.count; ++i) {
    v4l2_buffer buf{};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = i;
    if (xioctl(fd, VIDIOC_QUERYBUF, &buf) < 0) {
      emitStatus("[RGB OPEN] query buffer failed index=" + std::to_string(i));
      close(fd);
      runFallback();
      return;
    }

    buffers[i].length = buf.length;
    buffers[i].start = mmap(nullptr, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, buf.m.offset);
    if (buffers[i].start == MAP_FAILED) {
      emitStatus("[RGB OPEN] buffer mmap failed");
      close(fd);
      runFallback();
      return;
    }
  }

  for (uint32_t i = 0; i < req.count; ++i) {
    v4l2_buffer buf{};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = i;
    if (xioctl(fd, VIDIOC_QBUF, &buf) < 0) {
      emitStatus("[RGB OPEN] qbuf failed index=" + std::to_string(i));
      close(fd);
      runFallback();
      return;
    }
  }

  v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  if (xioctl(fd, VIDIOC_STREAMON, &type) < 0) {
    emitStatus("[RGB OPEN] stream on failed device=" + path);
    close(fd);
    runFallback();
    return;
  }

  const std::string actualFormat = fourccToString(fmt.fmt.pix.pixelformat);
  const std::string actualSize = std::to_string(fmt.fmt.pix.width) + "x" + std::to_string(fmt.fmt.pix.height);
  const int actualFps = (parm.parm.capture.timeperframe.denominator > 0 && parm.parm.capture.timeperframe.numerator > 0)
                            ? static_cast<int>(parm.parm.capture.timeperframe.denominator / parm.parm.capture.timeperframe.numerator)
                            : static_cast<int>(requestedFps);

  emitStatus("[RGB OPEN] device=" + path + " fmt=" + actualFormat + " size=" + actualSize + " fps=" + std::to_string(actualFps) + " open=ok");
  if (config_.mirrorRgbAtCapture) {
    Logger::info("RGB mirror at capture: enabled");
  } else {
    Logger::info("RGB mirror at capture: disabled");
  }

  uint64_t frameId = 0;
  uint64_t lastPerfLogNs = monotonicRawNowNs();
  double sumDqMs = 0.0;
  double sumDecodeMs = 0.0;
  double sumConvertMs = 0.0;
  double sumTotalMs = 0.0;
  uint64_t perfCount = 0;

  FpsCounter inputFpsCounter;
  FpsCounter outputFpsCounter;

  while (running_.load()) {
    const uint64_t dqStartNs = monotonicRawNowNs();

    pollfd pfd{};
    pfd.fd = fd;
    pfd.events = POLLIN | POLLPRI;
    const int pollResult = poll(&pfd, 1, 1000);
    if (pollResult < 0) {
      if (errno == EINTR) {
        continue;
      }
      emitStatus("[RGB OPEN] poll failed " + std::string(std::strerror(errno)));
      break;
    }
    if (pollResult == 0) {
      continue;
    }

    v4l2_buffer buf{};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    if (xioctl(fd, VIDIOC_DQBUF, &buf) < 0) {
      if (errno == EAGAIN) {
        continue;
      }
      emitStatus("[RGB OPEN] dqbuf failed " + std::string(std::strerror(errno)));
      break;
    }

    const uint64_t dqEndNs = monotonicRawNowNs();
    const uint64_t monotonicAtDqNs = monotonicNowNs();
    const double dqMs = static_cast<double>(dqEndNs - dqStartNs) / 1000000.0;

    cv::Mat decoded;
    double decodeMs = 0.0;
    double convertMs = 0.0;
    const uint64_t decodeStartNs = monotonicRawNowNs();

    if (fmt.fmt.pix.pixelformat == V4L2_PIX_FMT_MJPEG) {
      // MJPG 需要 JPEG 解码；适合 USB 带宽受限场景，但会消耗 CPU。
      std::vector<unsigned char> rawData(static_cast<unsigned char*>(buffers[buf.index].start),
                                         static_cast<unsigned char*>(buffers[buf.index].start) + buf.bytesused);
      decoded = cv::imdecode(rawData, cv::IMREAD_COLOR);
      const uint64_t decodeEndNs = monotonicRawNowNs();
      decodeMs = static_cast<double>(decodeEndNs - decodeStartNs) / 1000000.0;
    } else if (fmt.fmt.pix.pixelformat == V4L2_PIX_FMT_YUYV) {
      // YUYV 是未压缩 YUV422，需转换到 OpenCV 常用的 BGR 格式。
      cv::Mat raw(fmt.fmt.pix.height, fmt.fmt.pix.width, CV_8UC2, buffers[buf.index].start);
      const uint64_t convertStartNs = monotonicRawNowNs();
      cv::cvtColor(raw, decoded, cv::COLOR_YUV2BGR_YUY2);
      const uint64_t convertEndNs = monotonicRawNowNs();
      convertMs = static_cast<double>(convertEndNs - convertStartNs) / 1000000.0;
      decodeMs = 0.0;
    } else {
      cv::Mat raw(fmt.fmt.pix.height, fmt.fmt.pix.width, CV_8UC3, buffers[buf.index].start);
      decoded = raw.clone();
      decodeMs = 0.0;
      convertMs = 0.0;
    }

    const uint64_t totalEndNs = monotonicRawNowNs();
    const double totalMs = static_cast<double>(totalEndNs - dqStartNs) / 1000000.0;

    if (decoded.empty()) {
      emitStatus("[RGB OPEN] decode failed device=" + path + " fmt=" + actualFormat);
      v4l2_buffer qbuf{};
      qbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      qbuf.memory = V4L2_MEMORY_MMAP;
      qbuf.index = buf.index;
      xioctl(fd, VIDIOC_QBUF, &qbuf);
      continue;
    }

    if (config_.mirrorRgbAtCapture) {
      cv::flip(decoded, decoded, 1);
    }

    FrameEnvelope envelope;
    envelope.source = FrameSource::Rgb;
    const uint64_t arrivalTsNs = dqEndNs;
    const uint64_t v4l2TsNs = static_cast<uint64_t>(buf.timestamp.tv_sec) *
        1000000000ULL + static_cast<uint64_t>(buf.timestamp.tv_usec) * 1000ULL;
    const bool monotonicTimestamp =
        (buf.flags & V4L2_BUF_FLAG_TIMESTAMP_MASK) ==
        V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC;
    uint64_t mappedTsNs = 0;
    if (monotonicTimestamp && v4l2TsNs > 0) {
      const int64_t rawMinusMonotonic = static_cast<int64_t>(arrivalTsNs) -
          static_cast<int64_t>(monotonicAtDqNs);
      const int64_t mapped = static_cast<int64_t>(v4l2TsNs) + rawMinusMonotonic;
      if (mapped > 0) mappedTsNs = static_cast<uint64_t>(mapped);
    }
    if (mappedTsNs > lastV4l2SyncTsNs_) {
      tsNormalizer_.stampNativeMonotonic(envelope, arrivalTsNs, mappedTsNs,
                                         v4l2TsNs / 1000ULL);
      lastV4l2SyncTsNs_ = mappedTsNs;
    } else {
      tsNormalizer_.stampHostFallback(
          envelope, arrivalTsNs, v4l2TsNs / 1000ULL,
          monotonicTimestamp ? "v4l2_timestamp_non_monotonic" :
                               "v4l2_timestamp_not_monotonic");
    }
    if (frameId == 0) {
      std::ostringstream trace;
      trace << "[RGB TS] flags=0x" << std::hex << buf.flags << std::dec
            << " type=" << (monotonicTimestamp ? "monotonic" : "untrusted")
            << " device_us=" << (v4l2TsNs / 1000ULL)
            << " arrival_ns=" << arrivalTsNs
            << " sync_ns=" << envelope.syncTsNs
            << " quality=" << envelope.clockQuality;
      emitStatus(trace.str());
    }
    envelope.frameId = frameId++;
    envelope.width = decoded.cols;
    envelope.height = decoded.rows;
    envelope.image = decoded.clone();
    envelope.pixelFormatName = actualFormat;

    inputFpsCounter.tick(arrivalTsNs);
    outputFpsCounter.tick(arrivalTsNs);

    if (callback_) {
      // 上层采用 latest-only 处理策略；采集线程只交付当前帧，不在本层堆积历史帧。
      callback_(std::move(envelope));
    }

    sumDqMs += dqMs;
    sumDecodeMs += decodeMs;
    sumConvertMs += convertMs;
    sumTotalMs += totalMs;
    ++perfCount;

    const uint64_t nowNs = monotonicRawNowNs();
    if (nowNs - lastPerfLogNs >= 1000000000ULL) {
      const double fpsIn = inputFpsCounter.fps();
      const double fpsOut = outputFpsCounter.fps();
      const double avgDqMs = perfCount ? sumDqMs / static_cast<double>(perfCount) : 0.0;
      const double avgDecodeMs = perfCount ? sumDecodeMs / static_cast<double>(perfCount) : 0.0;
      const double avgConvertMs = perfCount ? sumConvertMs / static_cast<double>(perfCount) : 0.0;
      const double avgTotalMs = perfCount ? sumTotalMs / static_cast<double>(perfCount) : 0.0;

      std::ostringstream oss;
      oss << "[RGB PERF] device=" << path << " fps_in=" << std::fixed << std::setprecision(1) << fpsIn
          << " fps_out=" << fpsOut << " dqbuf=" << avgDqMs << "ms"
          << " decode=" << avgDecodeMs << "ms"
          << " convert=" << avgConvertMs << "ms"
          << " total=" << avgTotalMs << "ms"
          << " fmt=" << actualFormat << " size=" << actualSize;
      emitStatus(oss.str());

      lastPerfLogNs = nowNs;
      sumDqMs = 0.0;
      sumDecodeMs = 0.0;
      sumConvertMs = 0.0;
      sumTotalMs = 0.0;
      perfCount = 0;
    }

    v4l2_buffer qbuf{};
    qbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    qbuf.memory = V4L2_MEMORY_MMAP;
    qbuf.index = buf.index;
    if (xioctl(fd, VIDIOC_QBUF, &qbuf) < 0) {
      emitStatus("[RGB OPEN] qbuf failed " + std::string(std::strerror(errno)));
      break;
    }
  }

  type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  xioctl(fd, VIDIOC_STREAMOFF, &type);
  for (const auto& mapped : buffers) {
    if (mapped.start && mapped.start != MAP_FAILED) {
      munmap(mapped.start, mapped.length);
    }
  }
  close(fd);
}

void RgbCaptureV4L2::runFallback() {
  // 无真实摄像头时生成灰底测试帧，保证同步、UI 和录制流程仍可被调试。
  if (config_.enableCpuAffinity) {
    bindCurrentThreadToCpu(config_.rgbCaptureCpu, "rgb_capture");
  }

  uint64_t frameId = 0;
  const int width = config_.rgbWidth > 0 ? config_.rgbWidth : 640;
  const int height = config_.rgbHeight > 0 ? config_.rgbHeight : 480;

  while (running_.load()) {
    cv::Mat frame(height, width, CV_8UC3, cv::Scalar(30, 30, 30));
    const std::string text = "RGB fallback frame_id=" + std::to_string(frameId);
    cv::putText(frame, text, cv::Point(20, height / 2), cv::FONT_HERSHEY_SIMPLEX,
                0.7, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

    if (frameId == 0) {
      if (config_.mirrorRgbAtCapture) {
        Logger::info("RGB mirror at capture: enabled");
      } else {
        Logger::info("RGB mirror at capture: disabled");
      }
    }
    if (config_.mirrorRgbAtCapture) {
      cv::flip(frame, frame, 1);
    }

    FrameEnvelope envelope;
    envelope.source = FrameSource::Rgb;
    // fallback 帧同样使用主机单调时钟，便于和模拟 Depth 帧走同一同步路径。
    tsNormalizer_.stampHostFallback(envelope, monotonicRawNowNs(), 0,
                                    "synthetic_rgb_fallback");
    envelope.frameId = frameId++;
    envelope.width = frame.cols;
    envelope.height = frame.rows;
    envelope.image = std::move(frame);
    envelope.pixelFormatName = "BGR";

    if (callback_) {
      callback_(std::move(envelope));
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(33));
  }
}

}  // namespace rehab

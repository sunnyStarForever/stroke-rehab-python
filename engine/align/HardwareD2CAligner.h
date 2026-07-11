/*
 * 模块作用：
 * 本文件声明硬件 D2C 对齐器。硬件 D2C 成功时，深度设备已经把 Depth
 * 注册到 RGB 坐标系，pipeline 只需要验证尺寸和格式即可使用。
 */
#pragma once

#include "engine/common/FrameEnvelope.h"

namespace rehab {

/*
 * HardwareD2CAligner
 * 职责：
 * 1. 接收 OpenNI2 已完成 Depth-to-Color 的深度帧；
 * 2. 验证深度尺寸是否和 RGB 一致；
 * 3. 输出可直接按 RGB 关键点坐标采样的 alignedDepth。
 */
class HardwareD2CAligner {
 public:
  cv::Mat align(const FrameEnvelope& depth, const FrameEnvelope& rgb) const;
};

}  // namespace rehab

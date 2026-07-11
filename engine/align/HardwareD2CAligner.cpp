/*
 * 模块作用：
 * 本文件实现硬件 D2C 对齐的轻量验证。硬件对齐真正发生在 OpenNI2 设备侧，
 * 这里不重新投影，只确认深度图可作为 RGB 坐标系下的 alignedDepth。
 */
#include "engine/align/HardwareD2CAligner.h"

namespace rehab {

cv::Mat HardwareD2CAligner::align(const FrameEnvelope& depth,
                                  const FrameEnvelope& rgb) const {
  if (depth.image.empty() || rgb.image.empty()) {
    return {};
  }

  // 硬件 D2C 模式下，Depth 帧理论上已经落在 RGB 坐标系。
  // 如果尺寸不一致，说明设备侧注册不可用，返回空图让 pipeline 回退到软件对齐。
  if (depth.width != rgb.width || depth.height != rgb.height) {
    return {};
  }

  if (depth.image.type() == CV_16UC1) {
    // 保持 16 位深度原值，后续 DepthSampler 再根据 depthUnitToMeter 转成米。
    return depth.image.clone();
  }

  cv::Mat converted;
  depth.image.convertTo(converted, CV_16UC1);
  return converted;
}

}  // namespace rehab

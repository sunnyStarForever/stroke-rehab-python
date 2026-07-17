# Python 模型部署

Python 应用层现在负责检测与姿态模型的加载、Execution Provider 选择、预处理和后处理。原生 `_core` 不再决定模型配置；它只作为相机、深度投影或可选 native 推理适配器。

## 模型资产

默认从仓库中的原版模型目录读取：

```text
stroke-rehab/including/yolov8n/yolov8n.onnx
stroke-rehab/including/rtmpose-t/end2end.onnx
stroke-rehab/including/rtmpose-t/pipeline.json
stroke-rehab/including/rtmpose-t/detail.json
stroke-rehab/including/rtmpose-t/deploy.json
```

也可以通过 `PipelineConfig.pose` 的以下字段覆盖路径：

- `detector_model_path`
- `model_path`
- `pipeline_json_path`
- `detail_json_path`
- `deploy_json_path`

相对路径会依次相对 `python_version` 和仓库根目录解析。

## 后端配置

`configs/device.yaml` 中的 `pose` 段控制部署：

```yaml
pose:
  inference_backend: "python"       # python / native / auto
  onnx_execution_provider: "auto"   # 或 CPUExecutionProvider 等准确名称
  onnx_intra_op_threads: 1
  onnx_inter_op_threads: 1
```

- `python`：必须使用 Python ONNX Runtime，适合作为当前主框架的正式配置。
- `native`：使用 `_core` 中原有 C++ ONNX Runtime 实现。
- `auto`：优先 Python，Python 会话不可用时回退 native。

环境变量 `STROKE_INFERENCE_BACKEND` 和 `STROKE_ONNX_PROVIDER` 可覆盖 YAML。

## 安装和验证

Windows/Linux 开发环境：

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux: source .venv/bin/activate
pip install -r requirements.txt
python verify_models.py
python test_inference.py
```

`verify_models.py` 会实际加载两份 ONNX 文件、打印输入输出名称和 Execution Provider，并执行一帧 YOLO 与 RTMPose 推理。

## 目标 Linux 板

目标板应优先安装与 Python 版本、CPU 架构匹配的 ONNX Runtime wheel。如果目标板暂时没有可用的 Python ORT wheel，可将 `inference_backend` 临时设为 `auto` 或 `native`，继续复用已有 `_core` 推理适配器；Python 仍负责主程序、配置、采集生命周期、训练逻辑和输出。该降级必须在部署记录中明确标注，不能当作 Python 推理已经验证。

模型更新时必须重新运行 `verify_models.py`，因为脚本同时验证固定的 YOLO IO、RTMPose `input/simcc_x/simcc_y` 合同和 Halpe26 输出规模。

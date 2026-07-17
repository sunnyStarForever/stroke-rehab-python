# Python 训练语音提示

`rehab_engine.voice.VoiceAssistant` 使用独立工作线程串行执行 TTS。采集线程、评分 stdout 线程和 Qt 主线程只负责将短文本放入有界优先级队列，不等待音频播放完成。

## 训练事件

训练页会为以下事件生成中文提示：

- 课程开始、暂停、继续和结束；
- 新动作名称、目标次数与动作要领；
- 动作完成、平均评分和休息时长；
- 休息剩余 10、5、3、2、1 秒；
- 每个新完成周期的次数与综合评分。

高优先级状态提示可以替换队列中的低优先级反馈。相同 key 在冷却时间内不会重复排队，避免评分帧或状态回调造成连续播报。

## 配置

`configs/device.yaml`：

```yaml
voice:
  enabled: true
  backend: "auto"
  rate: 175
  volume: 0.9
  cooldown_seconds: 2.0
  queue_size: 12
```

当前 `auto` 使用 pyttsx3：Windows 通常走 SAPI，Linux 取决于系统可用的 speech driver。可用 `STROKE_VOICE_ENABLED=false` 临时禁用。

TTS 初始化或播放失败时，服务记录错误并停止后续语音，但训练、采集、评分、录制和安全停止继续运行。语音不是训练完成判定或安全控制条件。

## 验证

```bash
python test_voice.py
QT_QPA_PLATFORM=offscreen python test_ui.py
```

单元测试使用可控的假音频引擎验证非阻塞、优先级、去重和失败降级。目标板仍需验证实际扬声器、中文 voice、音量和现场噪声环境。

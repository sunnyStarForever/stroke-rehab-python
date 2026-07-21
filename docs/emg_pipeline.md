# Python 肌电采集与 CPU1 部署

Python 应用层统一管理 EMG 的关闭/真实采集状态、BLE 或串口采集、原始数据组块、RPMsg v2 通信、CPU1 特征返回、运行状态和训练录制。肌电未启用时不创建采集数据；启用后只允许真实硬件链路。原 C++ 代码中的协议约束被保留，但不再要求 C++ 管理主流程。

## 实时数据路径

```text
ESP32
  ├─ BLE GATT notify (204 bytes)
  └─ RFCOMM/serial: EMG,<seq>,<ch0>,<ch1>
                 │
                 v
        Python validation + sequence diagnostics
                 │  25 samples/chunk by default
                 v
             RPMsg v1 raw packet
                 │
                 v
          CPU1 feature extraction
                  │  RPMsg v1 feature packet
                 v
       Python UI/status/CSV/session summary
```

主要实现位于 `rehab_engine/emg.py`，录制逻辑位于 `rehab_engine/recorder.py`。

## BLE GATT

`bluez` 后端通过 Bleak 连接固件服务，并保持原握手顺序：

1. 订阅 status 与 data notify。
2. 向 command characteristic 写入 `START_EMG`。
3. 只有收到 `EMG_START_OK` 后才接受数据流。
4. 停止时发送 `STOP_EMG`，等待 `EMG_STOP_OK`，随后取消订阅并断开。

数据 notify 必须严格为 204 字节：4 字节 uint32 packet sequence，随后是 25 组双通道 int32。Python 会重建每个样本的主机时间戳，并记录丢包、重复包、乱序包及 uint32 回绕。

## 串口 / RFCOMM

`serial` 后端使用 pyserial 打开 `serial_device`，按行解析：

```text
EMG,<uint32 seq>,<int32 ch0>,<int32 ch1>
```

超过 512 字节的异常行会被清空；字段错误会累计 `parse_errors`；通道值限制到 int32 范围。状态中同时报告根据接收时间估算的采样率。BLE 与串口解析完成后进入同一个原始样本组块与 RPMsg 路径。

## RPMsg v1

Python 在 Linux 上通过 rpmsg-char 创建 endpoint，并严格匹配 CPU1 固件 `emg_protocol.h` 的 v1 小端协议：32 字节公共头、固定结构 CONFIG/RAW_CHUNK/FEATURE/HEARTBEAT/ERROR 包以及 int16 原始采样。endpoint 打开后，Python 先发送 CONFIG；只有收到带 `CFG1` 标志的 HEARTBEAT 确认后才发送原始数据。CPU1 ERROR 包会显示具体错误码、detail 和消息。启用肌电后默认要求 RPMsg 与采集端都启动成功；`strict_real_mode: true` 时启动失败会直接报告错误，不会生成模拟数据。

CPU1/remoteproc 必须先启动，并生成配置中的控制与数据设备，例如：

```yaml
rpmsg_ctrl_device: "/dev/rpmsg_ctrl0"
rpmsg_data_device: "/dev/rpmsg0"
rpmsg_endpoint_name: "emg_rpmsg"
rpmsg_config_timeout_ms: 1500
```

## 配置示例

BLE：

```yaml
enabled: true
capture_backend: "bluez"
ble_address: ""
strict_real_mode: true
```

串口 / 已绑定 RFCOMM：

```yaml
enabled: true
capture_backend: "serial"
serial_device: "/dev/rfcomm0"
serial_baud_rate: 460800
strict_real_mode: true
```

完整参数见 `configs/emg.yaml`。环境变量 `STROKE_EMG_ENABLED`、`STROKE_EMG_CAPTURE_BACKEND`、`STROKE_EMG_SERIAL_DEVICE`、`STROKE_EMG_BLE_ADDRESS` 和 `STROKE_EMG_STRICT_REAL` 可覆盖常用部署项。

## 验证

协议与生命周期回归：

```bash
python test_emg.py
```

目标板仍需用真实设备验收 BLE 握手、串口波特率、持续丢包统计、remoteproc/rpmsg endpoint、CPU1 特征数值及反复启停。桌面测试使用可控伪设备验证逻辑合同，不等同于蓝牙射频和板端链路已验证。

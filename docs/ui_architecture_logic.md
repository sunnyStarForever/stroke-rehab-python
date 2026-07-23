# Python UI 界面逻辑与架构说明

本文档梳理 `python_version/ui` 当前界面实现逻辑，目的是为下一步绘制软件架构图、界面功能图和训练流程图提供素材。整体结构可参考“前端架构模块 — 应用控制与页面调度 — 后端架构模块 — 本地数据与报告管理”的分层方式。

## 1. 总体定位

当前 Python 版界面是一个基于 PyQt5 与 QFluentWidgets 的桌面康复训练系统。UI 本身不直接生成训练数据，而是负责：

- 组织页面导航与用户交互。
- 展示真实 RGB、Depth、骨骼、评分、肌电状态。
- 调度采集 Pipeline、课程状态机、实时评分服务、录制与报告生成。
- 把配置、会话元数据、动作摘要和报告文件写入本地目录。
- 保证耗时任务在后台线程执行，避免界面卡死。

当前数据策略为“只接受真实数据”：相机、深度和骨骼链路没有模拟数据 fallback；真实采集核心或设备不可用时，采集启动失败并提示原因。

## 2. 建议作图时的总分层

可把系统画成四个主要区域：

```text
前端界面层
  ├─ MainWindow 主窗口 / 导航
  ├─ PatientHomePage 患者首页
  ├─ TrainingPage 训练页
  ├─ ReportsPage 报告页
  ├─ SettingsPage 设置页
  └─ 预览、评分、肌电、调试、日志/性能弹窗等控件

应用控制层
  ├─ 页面切换与信号连接
  ├─ 训练状态控制
  ├─ 课程动作调度
  ├─ 评分结果回传
  ├─ 报告触发与页面跳转
  └─ 配置保存与诊断刷新

后端业务与采集层
  ├─ SensorPipeline 真实 RGB-D 采集与骨骼处理
  ├─ CourseRunner 课程/动作/休息状态机
  ├─ ScoreBridge 实时评分子进程桥接
  ├─ OfflineReportRunner 单动作离线报告
  ├─ EmgManager 肌电链路管理
  └─ VoiceAssistant 语音提示

本地数据与报告层
  ├─ session_ui_meta.json
  ├─ course_summary.json
  ├─ skeleton_3d.csv
  ├─ actions/*/skeleton3d.csv
  ├─ actions/*/report/offline_action_report.html
  └─ session_report.html / 康复趋势汇总
```

## 3. 前端界面层

### 3.1 MainWindow 主窗口

对应文件：`python_version/ui/main_window.py`

`StrokeRehabWindow` 是整个界面的主容器，继承 `FluentWindow`，负责左侧导航、全局主题、日志接收和页面之间的信号连接。

主要功能：

- 初始化窗口标题、大小、主题色。
- 加载 `PipelineConfig`。
- 执行或接收系统诊断结果。
- 创建并挂载四个主页面：
  - 首页：`PatientHomePage`
  - 训练：`TrainingPage`
  - 报告：`ReportsPage`
  - 设置：`SettingsPage`
- 连接页面信号：
  - 首页选择课程后跳转训练页。
  - 训练结束后请求报告页加载报告。
  - 设置页修改课程、调试开关、主题后同步到首页和训练页。
  - 设置页请求日志/性能窗口时弹出独立窗口。
- 接收后端 logger 消息，写入内存日志列表，并根据告警节流策略显示 InfoBar。
- 每秒刷新一次状态栏，展示真实数据模式、训练状态、相机状态等摘要。

作图建议：MainWindow 可以放在“前端架构”最上层，作为页面容器和信号总线。

### 3.2 PatientHomePage 患者首页

对应文件：`python_version/ui/pages/patient_home_page.py`

首页承担“训练入口”和“患者概览”的角色。

主要功能：

- 左侧展示患者信息：姓名、性别、年龄、诊断、编号。
- 左侧展示最近训练记录，双击可跳转报告页。
- 右侧展示课程卡片：
  - 课程名称
  - 动作列表和目标次数
  - 难度
  - 预计训练时长
  - “进入训练”按钮
- 展示医生建议/训练计划说明。
- 从 `CourseRepository` 读取课程配置，从本地记录目录扫描最近训练历史。

核心信号：

- `course_selected(course_id)`：用户选择课程并进入训练。
- `report_requested(session_dir, csv_path)`：用户从历史记录打开报告。

作图建议：可画成“患者信息卡 + 课程选择 + 最近训练入口”。

### 3.3 TrainingPage 训练页

对应文件：`python_version/ui/pages/training_page.py`

训练页是当前系统最核心的界面，承担采集启动、训练开始、课程动作调度、实时预览、评分展示、肌电展示、录制、结束与报告触发。

页面结构：

- 顶部训练摘要区：
  - 当前课程
  - 当前动作
  - 训练状态
  - 本次训练时长
- 主体左侧：
  - `PreviewWidget` 实时动作捕捉预览
  - RGB / 深度 / 骨骼三种视图切换
  - 预览 HUD、训练进度和录制标记
- 主体右侧：
  - 训练进程卡片
  - 当前动作与训练要点
  - 目标次数、动作进度、休息倒计时
  - `ScorePanel` 实时训练质量
  - `EmgPanel` 肌电监测
  - `DebugPanel` 评分引擎调试入口
- 底部：
  - 即时反馈条/反馈日志
  - 控制按钮：开始采集、开始训练、结束本次训练、查看报告

当前训练状态：

```text
IDLE              待采集
STARTING_CAPTURE  正在启动采集
CAPTURING          采集中（未训练）
TRAINING           训练中
RESTING            休息中
STOPPING           正在停止
FINISHED           已完成
```

说明：代码中仍保留 `PAUSED` 枚举作为历史兼容，但界面暂停按钮已隐藏且禁用；当前策略是不因 Pipeline 非致命异常自动暂停。Pipeline 断开或停止时，界面只提示用户检查设备并手动结束/重新开始。

### 3.4 ReportsPage 报告页

对应文件：`python_version/ui/pages/reports_page.py`

报告页负责“单次训练报告 + 历史训练 + 康复趋势”。

主要功能：

- 左侧扫描本地训练记录，最多展示最近 50 次。
- 右侧显示报告 HTML。
- 如果 `session_report.html` 不存在，但存在 `skeleton_3d.csv`，会后台调用 `generate_session_report()` 生成报告。
- 支持保存当前报告 HTML。
- 支持打开报告所在文件夹。
- 支持“康复趋势”视图：
  - 近 6 次平均评分
  - 近 6 次平均完成率
  - 近 6 次训练总时长
  - 最近训练评分走势
  - 训练明细表
- 报告加载、历史扫描都在后台线程中执行，完成后通过 Qt signal 回到主线程刷新界面。

作图建议：报告页可分成“历史记录管理”和“报告展示/趋势分析”两个子模块。

### 3.5 SettingsPage 设置页

对应文件：`python_version/ui/pages/settings_page.py`

设置页负责采集设备、课程、患者、主题、肌电和诊断相关配置。

主要功能模块：

- 相机与深度采集：
  - RGB 设备路径
  - RGB 格式
  - RGB 分辨率
  - RGB FPS
  - 深度设备 URI
  - 深度分辨率
  - 深度 FPS
  - 硬件 D2C 对齐
- 默认训练课程和患者信息：
  - 训练对象
  - 课程
  - 主题：浅色/深色
  - 性别、年龄、诊断说明
  - 性能调试开关
- 肌电采集：
  - 启用/禁用 EMG
  - 后端：bluez / serial
  - 串口
  - BLE 设备扫描
  - RPMsg 高级参数
- 系统工具：
  - 应用设置
  - 设备测试/诊断
  - 打开日志窗口
  - 打开性能监控窗口

输出信号：

- `course_changed`
- `debug_changed`
- `theme_changed`
- `settings_applied`
- `log_requested`
- `performance_requested`

作图建议：设置页可画成“配置输入层”，箭头指向 `PipelineConfig`，再由 MainWindow 分发到 TrainingPage 和 PatientHomePage。

## 4. 关键控件层

### 4.1 PreviewWidget 统一预览控件

对应文件：`python_version/ui/widgets/preview_widget.py`

功能：

- 接收 `PreviewFrame`。
- 绘制 RGB 背景。
- 绘制真实深度可视化。
- 绘制 2D 骨骼连线与关节点。
- 支持 RGB / Depth / Skeleton 三种互斥视图。
- 显示真实数据模式、帧率和调试 HUD。
- 显示训练计数、目标次数、质量文本和录制红点。

输入来源：

- `TrainingPage._refresh_preview()` 每 33ms 从 `SensorPipeline.preview.latest_frame()` 拉取最新帧。

### 4.2 ScorePanel 评分面板

对应文件：`python_version/ui/widgets/score_panel.py`

功能：

- 显示动作计数。
- 显示综合评分。
- 显示五个子评分：
  - 幅度
  - 平滑性
  - 躯干稳定
  - 对称性
  - 节奏性

输入来源：

- `ScoreBridge` 回传 `ScoreResult`。
- `TrainingPage._on_score()` 更新 `ScorePanel`、课程状态机和预览进度。

### 4.3 EmgPanel 肌电监测面板

对应文件：`python_version/ui/widgets/emg_panel.py`

功能：

- 显示 EMG 状态：未启用、等待真实设备、已连接等。
- 显示两通道 RMS 趋势波形。
- 显示两通道 RMS 条形进度。
- 显示疲劳指数。

当前边界：

- 当前主界面已有轻量 RMS 趋势波形和基础特征显示。
- 如果后续需要更完整的实时肌电波形窗口，可新增独立 Dialog，展示原始波形、多通道时域特征和频域指标。

### 4.4 DebugPanel 评分调试面板

对应文件：`python_version/ui/widgets/debug_panel.py`

功能：

- 从 `ScoreBridge` 拉取 debug state。
- 绘制分割特征波形。
- 展示 detected peaks、accepted peaks、accepted centers。
- 展示当前计数、完成次数、FPS、分割参数等。

当前定位：

- 属于调试工具，不建议长期挤在主界面核心区域。
- 适合后续改成“点击按钮弹出调试窗口”的结构，主界面只保留入口和摘要状态。

### 4.5 LogDialog 与 PerformanceDialog

对应文件：`python_version/ui/dialogs/runtime_dialogs.py`

功能：

- `LogDialog`：展示运行日志，支持按 INFO / WARN / ERROR / PERF 过滤。
- `PerformanceDialog`：展示实时性能快照，如 RGB FPS、Depth FPS、同步 FPS、Worker FPS、Pose FPS、队列水位、写盘耗时等。

输入来源：

- MainWindow 接收 logger 回调。
- PerformanceDialog 由 MainWindow 定时或用户打开时读取 `TrainingPage.pipeline_stats()`。

## 5. 应用控制层

应用控制层不是单独文件，而是分散在 `MainWindow` 和 `TrainingPage` 中。它负责把页面事件转成后端任务，并把后端结果安全地回到 Qt 主线程刷新界面。

### 5.1 页面调度

主要路径：

```text
首页点击课程
  → PatientHomePage.course_selected(course_id)
  → MainWindow._open_course(course_id)
  → TrainingPage.set_course(course_id)
  → MainWindow.switchTo(training)
```

```text
训练结束生成报告
  → TrainingPage.report_requested(session_dir, csv_path)
  → MainWindow.navigate_to_reports(...)
  → ReportsPage.load_session(...)
  → MainWindow.switchTo(reports)
```

```text
设置页修改配置
  → SettingsPage 保存 PipelineConfig
  → course_changed / debug_changed / theme_changed / settings_applied
  → MainWindow 分发到 TrainingPage、PatientHomePage、主题系统
```

### 5.2 训练状态控制

训练页的主状态转换建议画成：

```text
IDLE
  └─ 点击“开始采集”
       → STARTING_CAPTURE
       → 后台启动 SensorPipeline
       → 成功：CAPTURING
       → 失败：IDLE + InfoBar 错误提示

CAPTURING
  └─ 点击“开始训练”
       → 创建 session 目录
       → start_recording()
       → CourseRunner.start_course()
       → TRAINING

TRAINING
  ├─ ScoreBridge 回传计数达到目标
  │    → CourseRunner 完成当前动作
  │    → RESTING 或 FINISHED
  └─ 点击“结束本次训练”
       → STOPPING
       → 停止课程、评分、动作录制、Pipeline
       → IDLE 或 FINISHED
       → 满足条件则跳转报告页

RESTING
  └─ 休息倒计时结束
       → 下一动作 TRAINING

FINISHED
  └─ 可查看报告，或重新开始采集
```

### 5.3 非阻塞策略

当前界面为了避免卡死，使用了以下策略：

- Pipeline 启动在线程 `pipeline-start` 中执行。
- Pipeline 停止通过 `SensorPipeline.stop(on_complete=...)` 异步完成。
- ScoreBridge 启动在线程 `score-start-*` 中执行。
- 报告加载/生成在线程 `report-load` 中执行。
- 历史记录扫描在线程 `history-scan` 中执行。
- 设置页 BLE 扫描和设备诊断在线程中执行。
- 后台线程不直接改 UI，而是通过 Qt signal 回到主线程。
- 使用 generation id 丢弃过期后台结果，避免旧线程结果覆盖新状态。

## 6. 后端业务与采集层

### 6.1 SensorPipeline

对应文件：`python_version/rehab_engine/sensor_pipeline.py`

职责：

- 启动真实 RGB 与 Depth 采集。
- 执行 RGB/Depth 同步。
- 执行 YOLO 检测与 RTMPose 姿态估计。
- 将 Halpe 关键点映射到 Rehab22。
- 采样深度并重建 3D 骨骼。
- 对骨骼进行 EMA 和平滑。
- 生成 `PreviewFrame` 给 UI。
- 写入 `skeleton_3d.csv`、视频或动作数据。
- 管理 EMG 生命周期。
- 提供性能统计。

数据流：

```text
真实 RGB 相机 + 真实 Depth 相机
  → NativeRgbDepthBackend / _core
  → RGB-D 时间同步
  → pair_queue
  → Worker
  → YOLO / RTMPose
  → Rehab22 骨骼
  → 3D 重建 / 滤波 / 平滑
  → PreviewComposer
  → TrainingPage / PreviewWidget
  → Recorder / ScoreBridge / Reports
```

重要约束：

- 采集核心不可用时 `start()` 返回失败。
- RGB 与 Depth 必须配置为 30 FPS。
- Pipeline 断开不会主动暂停界面；界面只提示异常状态。

### 6.2 CourseRunner

对应文件：`python_version/rehab_engine/course.py`

职责：

- 加载课程动作序列。
- 维护当前动作 index。
- 接收评分计数。
- 判断动作是否完成。
- 管理动作之间的休息倒计时。
- 触发课程完成。

当前计数逻辑：

- 课程进度严格跟随 `ScoreResult.count`。
- `ScoreResult.count` 来自评分引擎寻峰结果。
- 一个 peak 对应一次 repetition。
- 当前已移除延迟确认式的“计数偏移”逻辑。

### 6.3 ScoreBridge

对应文件：`python_version/rehab_engine/scoring.py`

职责：

- 启动实时评分子进程。
- 把当前动作 ID 与采样 FPS 传给评分服务。
- 接收骨骼帧输入。
- 回传 `ScoreResult`：
  - count
  - overall_score
  - amplitude_score
  - smoothness_score
  - trunk_score
  - symmetry_score
  - rhythm_score
  - status
- 支持 debug state 查询，用于调试波形和寻峰状态。

### 6.4 OfflineReportRunner

对应文件：`python_version/rehab_engine/scoring.py`

职责：

- 每个动作完成后，基于动作 CSV 生成离线动作报告。
- 输出到 `actions/*/report/offline_action_report.html`。
- 完成后通过回调通知 TrainingPage 更新 `course_summary.json`。

### 6.5 EmgManager

对应文件：`python_version/rehab_engine/emg.py`

职责：

- 管理 EMG 真实链路。
- 支持 Serial/RFCOMM、BLE 扫描、RPMsg 与 CPU1 协议。
- 解析 raw sample 和 feature frame。
- 缓存最近 EMG feature，融合到 PreviewFrame。
- 动作开始/结束时配合 `EmgRecorder` 写入动作级 EMG 数据。

当前 UI 展示：

- TrainingPage 右侧 `EmgPanel` 展示连接状态、RMS 趋势、通道强度和疲劳指数。

## 7. 本地数据与报告管理

### 7.1 会话目录结构

默认路径：

```text
recordings/sessions/<session_id>/
  ├─ skeleton_3d.csv
  ├─ session_ui_meta.json
  ├─ course_summary.json
  ├─ session_report.html
  └─ actions/
      └─ 01_Mx_<movement_id>/
          ├─ skeleton3d.csv
          ├─ emg...
          └─ report/
              └─ offline_action_report.html
```

### 7.2 关键文件作用

| 文件 | 生成位置 | 作用 |
| --- | --- | --- |
| `skeleton_3d.csv` | SensorPipeline / Skeleton3DRecorder | 整个会话的 3D 骨骼记录 |
| `session_ui_meta.json` | TrainingPage | 患者、课程、开始/结束时间、训练状态 |
| `course_summary.json` | TrainingPage | 每个动作的目标次数、实际次数、平均分、动作报告路径 |
| `actions/*/skeleton3d.csv` | ScoringCsvRecorder | 单动作评分/离线报告输入 |
| `offline_action_report.html` | OfflineReportRunner | 单动作离线报告 |
| `session_report.html` | generate_session_report | 会话级摘要报告 |

### 7.3 报告生成触发逻辑

训练结束时：

- 如果课程自然完成：生成报告。
- 如果用户主动结束：
  - 训练时长达到 `TRAINING_REPORT_MIN_SECONDS`，当前为 20 秒，则生成报告。
  - 或者已有动作次数/评分结果，也生成报告。
  - 否则只保存数据，不生成报告。

报告页打开时：

- 如果已有 `session_report.html`，直接加载。
- 如果没有但有 `skeleton_3d.csv`，后台生成后加载。
- 如果没有可用数据，显示默认说明页。

## 8. 界面与后端的数据/事件流

### 8.1 实时预览流

```text
真实 RGB/Depth 设备
  → SensorPipeline
  → PreviewComposer.latest_frame()
  → TrainingPage._refresh_preview()，约 33ms 一次
  → PreviewWidget.set_frame()
  → RGB / Depth / Skeleton 画面刷新
```

### 8.2 实时评分流

```text
当前动作开始
  → TrainingPage 创建 ScoreBridge
  → ScoreBridge.start(action_id, fps)
  → TrainingPage 将骨骼帧送入 ScoreBridge
  → ScoreBridge 回传 ScoreResult
  → ScorePanel 更新分数和计数
  → CourseRunner.on_score_updated()
  → 达到目标次数后完成动作
```

### 8.3 课程动作流

```text
CourseRepository 读取课程
  → TrainingPage 当前课程
  → CourseRunner.start_course()
  → on_action_changed
  → 创建动作目录 / 启动动作 CSV / 启动动作 EMG 记录 / 启动 ScoreBridge
  → on_action_completed
  → 停止动作记录 / 停止评分 / 生成动作报告 / 写 course_summary.json
  → 休息 or 下一个动作
  → 全部完成后生成会话报告
```

### 8.4 设置配置流

```text
SettingsPage 用户修改表单
  → save_pipeline_config()
  → settings_applied
  → MainWindow 刷新首页/诊断
  → course_changed/debug_changed/theme_changed
  → TrainingPage / PatientHomePage / 全局主题同步
```

### 8.5 日志与性能流

```text
rehab_engine.logger
  → MainWindow.engine_log_received
  → 内存日志列表
  → LogDialog 实时追加
  → UserNotificationGate 判断是否弹 InfoBar

TrainingPage.pipeline_stats()
  → MainWindow / PerformanceDialog
  → 展示 FPS、队列、耗时、丢帧等指标
```

## 9. 作图时可以使用的模块节点

### 前端架构模块

- MainWindow 主窗口
- Navigation 页面导航
- PatientHomePage 患者首页
- TrainingPage 训练页
- ReportsPage 报告页
- SettingsPage 设置页
- PreviewWidget 统一预览控件
- ScorePanel 评分面板
- EmgPanel 肌电监测面板
- DebugPanel 评分调试面板
- LogDialog 日志窗口
- PerformanceDialog 性能监控窗口

### 应用控制模块

- 页面切换与信号联动
- 训练状态机
- 采集启动/停止控制
- 课程加载与动作调度
- 评分服务生命周期管理
- 录制生命周期管理
- 报告触发与页面跳转
- 配置保存与主题切换

### 后端架构模块

- SensorPipeline 真实多模态采集链路
- NativeRgbDepthBackend / _core 真实硬件适配
- FrameSynchronizer RGB-D 时间同步
- YOLO PersonDetector
- RTMPose Estimator
- Rehab22 Mapper
- DepthSampler / JointProjector3D
- EMA Filter / SkeletonSmoother
- PreviewComposer
- CourseRunner
- ScoreBridge
- OfflineReportRunner
- EmgManager
- VoiceAssistant

### 数据管理模块

- PipelineConfig / config.user.json
- CourseRepository / courses.json
- Skeleton3DRecorder
- ScoringCsvRecorder
- EmgRecorder
- course_summary.json
- session_ui_meta.json
- session_report.html
- ReportsPage 历史扫描与趋势汇总

## 10. 推荐绘图版本

### 图 1：软件总体架构图

建议画四块：

```text
前端界面层 → 应用控制层 → 后端业务/采集层 → 本地数据与报告层
```

重点突出：

- 前端不直接处理底层硬件。
- TrainingPage 是训练流程中枢。
- SensorPipeline 是真实 RGB-D 和骨骼处理主链路。
- CourseRunner 和 ScoreBridge 共同决定动作进度与计数。
- ReportsPage 从本地记录中生成单次报告和趋势视图。

### 图 2：训练流程状态机图

建议画：

```text
待采集 → 正在启动采集 → 采集中 → 训练中 → 休息中/训练中循环 → 正在停止 → 已完成/待采集 → 报告页
```

补充异常路径：

- 采集启动失败：回到待采集。
- Pipeline 非主动断开：界面提示，但不自动暂停。
- 停止不完整：弹错误提示，建议重启应用。

### 图 3：实时数据流图

建议画：

```text
RGB/Depth 真实设备
  → 采集适配器
  → 同步队列
  → 姿态识别与三维重建
  → PreviewFrame
  → 预览显示 / 评分 / 录制 / 报告
```

### 图 4：报告生成流程图

建议画：

```text
训练开始
  → 创建 session
  → 持续写 skeleton_3d.csv
  → 每个动作写 actions/*/skeleton3d.csv
  → 动作完成生成 offline_action_report.html
  → 训练结束写 session_ui_meta.json + course_summary.json
  → ReportsPage 加载或生成 session_report.html
  → 康复趋势聚合历史记录
```

## 11. 当前界面设计上的可优化点

这些不是必须立即实现，但适合后续迭代：

1. 将 `DebugPanel` 完全弹窗化，主训练页只保留“调试状态/打开调试窗口”按钮。
2. 将 EMG 扩展为独立实时窗口，显示原始波形、RMS、MAV、IEMG、WL、ZC 等多特征趋势。
3. 报告页趋势视图可以增加折线图或雷达图，而不是仅用 HTML 进度条。
4. 设置页可以继续按“患者信息 / 训练方案 / 采集设备 / 肌电设备 / 系统工具”分组，使用户路径更清晰。
5. 主窗口状态栏可以进一步区分“设备状态、训练状态、性能状态、报告状态”。
6. 当前 `CourseRunner` 中仍有历史暂停方法，可后续清理或标注为内部兼容，避免代码语义与 UI 策略不一致。
7. 可以增加“设备准备页”或“训练前校准页”，把相机、深度、骨骼质量、EMG 状态集中确认后再进入训练。


# Stroke Rehab Python

基于 PyQt5、QFluentWidgets 和 C++ 扩展的卒中康复训练桌面应用。系统提供实时骨骼预览、课程状态管理、动作评分、肌电状态、训练记录和 HTML 会话报告。

## 当前能力

- 上肢、下肢康复课程选择与持久化
- 训练前设备和保存目录检查
- 训练开始、暂停、继续、停止和自动完成状态机
- 22 关节骨骼 CSV 强制录制，暂停期间不写入训练数据
- 实时骨骼帧送入 `ScoreBridge`，显示次数和五维动作质量
- STUB 模式合成骨骼与 EMG 演示数据
- 会话元数据、训练历史和始终可用的 HTML 摘要报告
- 相机、引擎、EMG 和保存环境诊断
- 普通训练视图和性能调试视图

> 会话摘要中的“有效关节比例”和“数据质量”描述采集完整度，不是临床诊断结果。

## 环境安装

推荐 Python 3.10 或更高版本：

```bash
python -m venv .venv
source .venv/bin/activate        # Linux
# .venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

Windows Miniforge 示例：

```powershell
conda create -n stroke-rehab python=3.10
conda activate stroke-rehab
pip install -r requirements.txt
```

## 启动

```bash
python main.py
```

应用启动时会先运行环境诊断。没有加载 C++ 扩展时会自动进入 STUB 模式，可用于界面、课程、录制和报告流程验证。

## 配置

- 内置课程：`configs/courses.json`
- 用户设置：首次点击“应用设置”后写入 `config.user.json`
- 用户设置文件已被 Git 忽略，不会意外上传设备路径或患者名称
- 可用 `STROKE_USER_CONFIG` 指定其他用户配置路径
- 可用 `STROKE_REHAB_ROOT` 指定完整工程根目录

设置页提供课程、训练对象、相机、深度、EMG、调试显示和高级 RPMsg 参数，并支持“测试设备”。

## 训练记录

默认保存目录：

```text
recordings/sessions/YYYYMMDD/YYYYMMDD_HHMMSS_mmm/
├── skeleton_3d.csv
├── session_ui_meta.json
└── session_report.html
```

`recordings/` 默认不进入 Git。

## 实时评分工具

`ScoreBridge` 会查找：

```text
tools/scoring_engine/score_server.py
```

完整工程中需保证 `tools/scoring_engine` 位于仓库父级工程，或通过 `STROKE_REHAB_ROOT` 指向包含该目录的工程根目录。评分器不可用时，训练和骨骼录制仍会继续，会话摘要报告仍可生成。

## C++ 扩展

Windows：

```powershell
build_win.bat
```

Linux / 开发板：

```bash
chmod +x build_linux.sh setup_board.sh
./setup_board.sh
./build_linux.sh
```

## 测试

```bash
python test_stage2.py
python test_ui.py
python test_workflows.py
```

无显示器的 Linux 环境可使用：

```bash
QT_QPA_PLATFORM=offscreen python test_ui.py
```

## 重要提示

- 正式训练前应确认患者站位、遮挡、相机距离和关节有效率。
- 停止或完成训练会统一关闭课程计时器、评分进程、录制器和传感器流水线。
- 本项目用于康复训练辅助和技术研究，不替代治疗师判断或医疗诊断。

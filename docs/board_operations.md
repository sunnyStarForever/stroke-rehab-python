# 飞腾派运行与恢复手册

本文适用于 `/home/user/stroke-rehab-runtime` 部署树。正式训练前必须完成 OpenSpec 变更 `deploy-and-validate-real-depth` 的全部安全关键验收；当前文档本身不代表临床可用批准。

## 启动与单实例

在 XFCE 桌面菜单中选择“卒中康复训练”，或在终端执行：

```bash
/home/user/stroke-rehab-runtime/python_version/run_board.sh
```

启动脚本固定项目工作目录，加载 `board_env.sh` 中的 OpenNI2 库和驱动路径，默认使用项目 `.venv/bin/python`，并确保 `python_version/recordings` 可写。可用 `STROKE_REHAB_PYTHON` 和 `STROKE_REHAB_RECORD_DIR` 显式覆盖解释器或录制根目录。

应用通过 Linux 文件锁限制为单实例。第二次启动应在打开摄像头前退出，并显示“程序已在运行，请先关闭现有窗口后再启动”。不得绕过该锁并行启动两个采集实例。

## 摄像头断开与恢复

1. 在界面中停止训练并正常关闭应用，确认没有 `main.py` 进程。
2. 重新插拔 Astra Pro，等待 `lsusb` 同时出现 `2bc5:0403` 与 `2bc5:0501`。
3. 确认 `/dev/video0`、`/dev/video1` 属于 `root:video` 且当前用户属于 `video` 组。
4. 重新启动应用；真实深度认证失败时不得继续训练，也不得把桩数据、合成数据或 RGB-only 数据标记为可信深度。

常用只读诊断：

```bash
pgrep -af 'python.*main.py'
lsusb | grep -E '2bc5:(0403|0501)'
fuser /dev/video0 /dev/video1 2>/dev/null
sha256sum rehab_engine/_core*.so
ldd rehab_engine/_core*.so | grep -i openni
```

硬件验证脚本应在应用关闭、摄像头无占用时运行：

```bash
cd /home/user/stroke-rehab-runtime/python_version
. ./board_env.sh
.venv/bin/python tools/validate_real_depth.py --help
.venv/bin/python tools/validate_rgb_depth_sync.py --help
.venv/bin/python tools/validate_real_pipeline.py --help
```

## SSH 安全要求

远程维护必须启用严格主机密钥校验。地址变化时应复用已核验的主机身份别名或由现场人员重新核验指纹；不得设置 `StrictHostKeyChecking=no`，不得把私钥内容写入日志、文档、命令输出或部署包。

## 回滚

部署前完整备份：

```text
/home/user/true-depth-predeploy-20260719-apply.tar.gz
SHA-256: 3b22eb3e5c73363761b33914eea1aeea48a8e83acc452afe953213e989036a03
```

回滚前先正常关闭应用并确认摄像头无占用。将备份解压到临时目录，核对清单后逐文件恢复；不要直接覆盖整个运行树。旧原生模块及评分修复前文件另有 `/home/user/*pre*20260719*` 备份。恢复后必须重新核对 `_core` 架构、OpenNI2 依赖、SHA-256、Python 导入和 `real_depth_active()`，并重新执行无设备故障关闭与真实硬件认证测试。

若回滚到缺少 `real_depth_active()` 的旧二进制，Python 层会按设计拒绝启动真实深度管线；不得通过删除认证检查来恢复运行。

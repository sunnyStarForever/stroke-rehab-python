"""
Stroke Rehab — Python application entry point.
Run: python main.py

Startup diagnostics are printed to console to verify camera/engine/hardware status.
If you don't see these diagnostics, the app may be crashing before reaching main().
"""

import sys
import os
import platform
from pathlib import Path

# Ensure python_version/ is on sys.path for imports
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def _acquire_single_instance_lock():
    """Prevent two Linux GUI instances from competing for the cameras."""
    if platform.system() != "Linux":
        return True, None

    import fcntl

    lock_path = Path("/tmp") / f"stroke-rehab-{os.getuid()}.lock"
    lock_file = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_file.close()
        return False, None

    lock_file.write(str(os.getpid()))
    lock_file.flush()
    return True, lock_file


def _startup_diagnostics():
    """Run system diagnostics before the UI launches.
    Prints detailed status to the console so users can verify what's working."""
    sep = "=" * 62
    print(f"\n{sep}", flush=True)
    print("  Stroke Rehab 康复训练系统 — 启动诊断", flush=True)
    print(f"  Python:  {sys.executable}", flush=True)
    print(f"  CWD:     {os.getcwd()}", flush=True)
    print(sep, flush=True)

    from rehab_engine.diagnostics import run_diagnostics, print_diagnostics
    diag = run_diagnostics()
    print_diagnostics(diag)

    # Return the diagnostics for possible UI use
    return diag


def main():
    lock_acquired, instance_lock = _acquire_single_instance_lock()
    if not lock_acquired:
        print("[startup] 程序已在运行，请先关闭现有窗口后再启动。", flush=True)
        return

    # ---- Platform-specific early fixes ----
    if platform.system() == "Linux":
        # On AArch64 boards, some Qt plugins may need explicit paths
        qt_plugin_path = os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH", "")
        if qt_plugin_path:
            print(f"[startup] QT_QPA_PLATFORM_PLUGIN_PATH={qt_plugin_path}", flush=True)

    # ---- Run diagnostics BEFORE creating QApplication ----
    # (QApplication steals stdout on some platforms, so we print before)
    diag = _startup_diagnostics()

    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    app = QApplication(sys.argv)

    from ui.main_window import StrokeRehabWindow

    window = StrokeRehabWindow(diagnostics=diag)
    window.show()

    exit_code = app.exec_()
    # Keep the lock file referenced until the Qt event loop has fully stopped.
    _ = instance_lock
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

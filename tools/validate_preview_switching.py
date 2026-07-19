"""Validate the three mutually exclusive training preview modes."""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("STROKE_VOICE_ENABLED", "false")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PyQt5.QtWidgets import QApplication

from rehab_engine._stub import PipelineConfig
from ui.pages.training_page import TrainingPage


def main() -> int:
    app = QApplication.instance() or QApplication([])
    config = PipelineConfig()
    config.voice.enabled = False
    page = TrainingPage(config)
    try:
        for mode in ("rgb", "depth", "skeleton"):
            page._preview_mode_buttons[mode].click()
            assert page._preview._view_mode == mode
            assert page._preview_mode_buttons[mode].isChecked()
            assert sum(
                button.isChecked()
                for button in page._preview_mode_buttons.values()
            ) == 1
        print("PREVIEW_SWITCH_OK", flush=True)
        return 0
    finally:
        page.shutdown()
        page.close()
        page.deleteLater()
        app.processEvents()
        del page
        gc.collect()
        app.processEvents()


if __name__ == "__main__":
    raise SystemExit(main())

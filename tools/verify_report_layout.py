"""Render an existing report at board window sizes and capture visual evidence."""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PyQt5.QtWidgets import QApplication

from ui.pages.reports_page import ReportsPage
from ui.report_images import adapt_report_images


def settle(app: QApplication, seconds: float = 0.25) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    html = args.report.read_text(encoding="utf-8", errors="ignore")
    app = QApplication.instance() or QApplication(sys.argv)
    page = ReportsPage()
    page._set_report_html(html, args.report.parent)
    failures = []
    for width, height in ((800, 600), (1280, 720), (2180, 1271)):
        page.resize(width, height)
        page.show()
        settle(app)
        page._apply_responsive_html()
        settle(app)
        viewport = page._browser.viewport().width()
        limit = max(1, viewport - 48)
        adapted = adapt_report_images(html, limit, args.report.parent)
        image_widths = [int(value) for value in re.findall(
            r'<img[^>]+width="(\d+)"', adapted)]
        if not image_widths or max(image_widths) > limit:
            failures.append((width, viewport, max(image_widths, default=-1)))
        scrollbar = page._browser.verticalScrollBar()
        for label, position in (("top", 0), ("middle", scrollbar.maximum() // 2),
                                ("bottom", scrollbar.maximum())):
            scrollbar.setValue(position)
            settle(app, 0.1)
            page.grab().save(str(args.output / f"report_{width}x{height}_{label}.png"))
        print(
            f"window={width}x{height} viewport={viewport} limit={limit} "
            f"images={len(image_widths)} max_image={max(image_widths, default=0)}",
            flush=True,
        )
    page.shutdown()
    page.close()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

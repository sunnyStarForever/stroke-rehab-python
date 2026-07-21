import base64
import re
import tempfile
import unittest
from pathlib import Path

from PyQt5.QtGui import QImage, QColor

from ui.report_images import adapt_report_images, fit_dimensions


def _png(path: Path, width=800, height=400) -> bytes:
    image = QImage(width, height, QImage.Format_RGB32)
    image.fill(QColor("#336699"))
    assert image.save(str(path), "PNG")
    return path.read_bytes()


def _size(html: str):
    match = re.search(r'<img[^>]+width="(\d+)" height="(\d+)"', html)
    return tuple(map(int, match.groups()))


class ReportImageSizingTests(unittest.TestCase):
    def test_fit_dimensions_never_enlarges_and_preserves_ratio(self):
        self.assertEqual(fit_dimensions(800, 400, 320), (320, 160))
        self.assertEqual(fit_dimensions(200, 100, 900), (200, 100))
        self.assertEqual(fit_dimensions(0, 100, 900), (0, 0))

    def test_embedded_image_adapts_to_narrow_medium_and_wide_viewports(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = base64.b64encode(_png(Path(tmp) / "chart.png")).decode()
            original = f'<html><img src="data:image/png;base64,{data}"></html>'
            self.assertEqual(_size(adapt_report_images(original, 240)), (240, 120))
            self.assertEqual(_size(adapt_report_images(original, 500)), (500, 250))
            self.assertEqual(_size(adapt_report_images(original, 1200)), (800, 400))
            self.assertNotIn('width="', original)

    def test_relative_file_image_uses_report_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _png(root / "trend.png", 640, 320)
            adapted = adapt_report_images(
                '<img src="trend.png" width="999" height="999">', 300, root)
            self.assertEqual(_size(adapted), (300, 150))
            self.assertEqual(adapted.count('width="'), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

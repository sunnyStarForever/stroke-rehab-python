"""Qt-rich-text compatible report image sizing utilities."""

import base64
import re
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import unquote, urlparse

from PyQt5.QtGui import QImage


_IMG_RE = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
_SRC_RE = re.compile(r"\bsrc\s*=\s*(['\"])(.*?)\1", re.IGNORECASE | re.DOTALL)
_SIZE_ATTR_RE = re.compile(
    r"\s+(?:width|height)\s*=\s*(?:['\"].*?['\"]|[^\s>]+)",
    re.IGNORECASE,
)
_STYLE_RE = re.compile(r"\s+style\s*=\s*(['\"])(.*?)\1", re.IGNORECASE | re.DOTALL)
_STYLE_SIZE_RE = re.compile(
    r"\s*(?:width|height|max-width|min-width|max-height|min-height)\s*:\s*[^;]+;?",
    re.IGNORECASE,
)


def fit_dimensions(width: int, height: int, available_width: int) -> Tuple[int, int]:
    """Fit an image without enlargement while preserving its aspect ratio."""
    if width <= 0 or height <= 0 or available_width <= 0:
        return 0, 0
    target_width = min(width, available_width)
    target_height = max(1, round(height * target_width / width))
    return target_width, target_height


def image_dimensions(src: str, base_dir: Optional[Path] = None) -> Tuple[int, int]:
    """Read intrinsic dimensions from embedded data or a local report image."""
    image = QImage()
    if src.startswith("data:image/"):
        marker = src.find(",")
        if marker >= 0:
            try:
                payload = base64.b64decode(src[marker + 1:], validate=False)
                image.loadFromData(payload)
            except (ValueError, TypeError):
                pass
    else:
        parsed = urlparse(src)
        if parsed.scheme == "file":
            raw_path = unquote(parsed.path)
            if parsed.netloc:
                raw_path = f"//{parsed.netloc}{raw_path}"
            # Windows file URIs are often represented as /E:/path by urlparse.
            if re.match(r"^/[A-Za-z]:", raw_path):
                raw_path = raw_path[1:]
            path = Path(raw_path)
        elif len(parsed.scheme) == 1 and len(src) > 2 and src[1] == ":":
            # Treat E:/chart.png or E:\chart.png as a Windows local path, not a URL scheme.
            path = Path(unquote(src))
        elif not parsed.scheme:
            path = Path(unquote(src))
            if not path.is_absolute() and base_dir is not None:
                path = base_dir / path
        else:
            return 0, 0
        image.load(str(path))
    return (image.width(), image.height()) if not image.isNull() else (0, 0)


def _clean_img_sizing(tag: str) -> str:
    """Remove size declarations that tend to override Qt's explicit attributes."""
    clean = _SIZE_ATTR_RE.sub("", tag[:-1]).rstrip()

    def replace_style(match: re.Match) -> str:
        quote = match.group(1)
        style = _STYLE_SIZE_RE.sub("", match.group(2)).strip()
        style = re.sub(r";{2,}", ";", style).strip("; ")
        return f' style={quote}{style}{quote}' if style else ""

    return _STYLE_RE.sub(replace_style, clean).rstrip()


def adapt_report_images(html: str, available_width: int,
                        base_dir: Optional[Path] = None) -> str:
    """Add explicit Qt-supported image sizes without changing the source HTML."""
    if not html or available_width <= 0:
        return html

    def replace(match: re.Match) -> str:
        tag = match.group(0)
        source_match = _SRC_RE.search(tag)
        if source_match is None:
            return tag
        width, height = image_dimensions(source_match.group(2), base_dir)
        fitted_width, fitted_height = fit_dimensions(width, height, available_width)
        if fitted_width <= 0:
            return tag
        clean = _clean_img_sizing(tag)
        return f'{clean} width="{fitted_width}" height="{fitted_height}">'

    return _IMG_RE.sub(replace, html)


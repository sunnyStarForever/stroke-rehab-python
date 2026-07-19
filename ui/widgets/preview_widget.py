"""
Preview widget — renders RGB frame with skeleton overlay.
Replaces app/widgets/UnifiedOverlayPreviewWidget.cpp (~300 lines).

Uses QPainter for real-time skeleton drawing.
No OpenCV dependency at UI level — PreviewFrame carries pre-composited data.

Startup overlay shows engine mode and camera status clearly.
Diagnostic HUD shows real-time performance metrics.
"""

from typing import Optional, List, Tuple

from PyQt5.QtCore import Qt, QPointF, QRect
from PyQt5.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont, QFontMetrics, QImage, QPixmap,
)
from PyQt5.QtWidgets import QWidget, QSizePolicy

from rehab_engine.preview import PreviewFrame

# Bone pairs for skeleton connectivity (21 bones)
BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
    (3, 6), (6, 7), (7, 8), (8, 9),
    (3, 10), (10, 11), (11, 12), (12, 13),
    (0, 14), (14, 15), (15, 16), (16, 17),
    (0, 18), (18, 19), (19, 20), (20, 21),
]

JOINT_NAMES = [
    "Waist", "Spine", "Chest", "Neck", "Head", "HeadTip",
    "L-Collar", "L-UpperArm", "L-Forearm", "L-Hand",
    "R-Collar", "R-UpperArm", "R-Forearm", "R-Hand",
    "L-UpperLeg", "L-LowerLeg", "L-Foot", "L-Toes",
    "R-UpperLeg", "R-LowerLeg", "R-Foot", "R-Toes",
]


class PreviewWidget(QWidget):
    """Real-time preview with skeleton overlay drawing."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAutoFillBackground(True)

        p = self.palette()
        p.setColor(self.backgroundRole(), QColor(16, 18, 20))
        self.setPalette(p)

        self._frame: Optional[PreviewFrame] = None
        self._view_mode = "skeleton"
        self._show_depth = True
        self._show_skeleton = True
        self._show_joint_index = False
        self._show_debug = True
        self._mirror = False
        self._recording = False
        self._progress_count = 0
        self._progress_target = 0
        self._quality_text = "等待评分"

        self._score_threshold = 0.35

        # ---- Diagnostic state ----
        self._engine_mode: str = "?"          # "STUB", "FULL"
        self._frame_count: int = 0
        self._last_frame_time: float = 0.0
        self._no_frame_warned: bool = False   # warn once if no frames arrive

    # ---- Setters ----

    def set_frame(self, frame: PreviewFrame):
        self._frame = frame
        self._frame_count += 1
        import time
        self._last_frame_time = time.monotonic()
        self._no_frame_warned = False
        self.update()

    def set_engine_mode(self, mode: str):
        """Set engine mode display: 'STUB' or 'FULL'."""
        self._engine_mode = mode
        self.update()

    def set_view_mode(self, mode: str):
        """Select an exclusive preview: RGB, depth, or RGB with skeleton."""
        if mode not in {"rgb", "depth", "skeleton"}:
            raise ValueError(f"Unsupported preview mode: {mode}")
        self._view_mode = mode
        self.update()

    def set_show_depth(self, v: bool):
        self._show_depth = v
        self.update()

    def set_show_skeleton(self, v: bool):
        self._show_skeleton = v
        self.update()

    def set_show_joint_index(self, v: bool):
        self._show_joint_index = v
        self.update()

    def set_show_debug(self, v: bool):
        self._show_debug = v
        self.update()

    def set_mirror(self, v: bool):
        self._mirror = v
        self.update()

    def set_recording(self, v: bool):
        self._recording = v
        self.update()

    def set_training_progress(self, count: int, target: int, quality: str = ""):
        self._progress_count = max(0, int(count))
        self._progress_target = max(0, int(target))
        self._quality_text = quality or "等待评分"
        self.update()

    # ---- RGB background drawing ----

    def _draw_rgb_background(self, painter: QPainter):
        import numpy as np
        rgb = self._frame.rgb_image
        if rgb is None:
            return
        try:
            h, w, _ = rgb.shape
            qimg = QImage(rgb.data.tobytes(), w, h, w * 3, QImage.Format_RGB888)
            target_rect = self.rect()
            scale = min(target_rect.width() / w, target_rect.height() / h)
            dw = int(w * scale)
            dh = int(h * scale)
            draw_rect = QRect(
                (target_rect.width() - dw) // 2,
                (target_rect.height() - dh) // 2,
                dw, dh,
            )
            painter.drawImage(draw_rect, qimg)
        except Exception:
            pass

    # ---- Depth overlay ----

    def _draw_depth_overlay(self, painter: QPainter, draw_rect, opacity: float = 0.45):
        import numpy as np
        import cv2
        depth = self._frame.depth_image
        if depth is None:
            return
        try:
            h, w = depth.shape
            # Normalise 16-bit depth to 0-255 for visualization
            valid = depth[depth > 0]
            if len(valid) == 0:
                return
            dmin, dmax = np.percentile(valid, [5, 95])
            drange = max(dmax - dmin, 1)
            vis = np.clip((depth.astype(np.float32) - dmin) / drange * 255, 0, 255).astype(np.uint8)
            # Apply color map (JET-like: near = warm, far = cold)
            colored = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
            colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
            h2, w2, _ = colored_rgb.shape
            qimg = QImage(colored_rgb.data.tobytes(), w2, h2, w2 * 3, QImage.Format_RGB888)
            target_rect = draw_rect
            painter.setOpacity(opacity)
            painter.drawImage(target_rect, qimg)
            painter.setOpacity(1.0)
        except Exception:
            pass

    # ---- Paint ----

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        # Background
        painter.fillRect(self.rect(), QColor(16, 18, 20))

        # ---- No frame at all: status overlay only ----
        if self._frame is None:
            self._draw_status_overlay(painter)
            return

        # Compute draw rect (640x480 nom., centered, keep aspect ratio)
        target_w, target_h = self.width(), self.height()
        src_w, src_h = 640, 480
        scale = min(target_w / src_w, target_h / src_h)
        draw_rect = QRect(
            (target_w - int(src_w * scale)) // 2,
            (target_h - int(src_h * scale)) // 2,
            int(src_w * scale), int(src_h * scale),
        )

        if self._view_mode == "depth":
            painter.fillRect(draw_rect, QColor(26, 30, 34))
            if (self._frame.depth_image is not None
                    and self._frame.depth_is_hardware):
                self._draw_depth_overlay(painter, draw_rect, opacity=1.0)
            else:
                painter.setPen(QColor(210, 215, 222))
                painter.setFont(QFont("Segoe UI", 15, QFont.Bold))
                painter.drawText(
                    draw_rect, Qt.AlignCenter, "未检测到真实深度数据")
        elif self._frame.rgb_image is not None:
            self._draw_rgb_background(painter)
        else:
            painter.fillRect(draw_rect, QColor(26, 30, 34))

        if (self._view_mode == "skeleton" and self._show_skeleton
                and self._frame.has_valid_2d):
            self._draw_skeleton(painter, draw_rect)

        # ── HUD layers ──
        if self._show_debug:
            self._draw_debug_panel(painter, draw_rect)
        self._draw_engine_badge(painter, draw_rect)
        self._draw_training_progress(painter, draw_rect)

        if self._recording:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(224, 48, 48))
            cx = draw_rect.right() - 30
            cy = draw_rect.top() + 20
            painter.drawEllipse(QPointF(cx, cy), 6, 6)
            painter.setPen(QColor(224, 48, 48))
            painter.setFont(QFont("Segoe UI", 11, QFont.Bold))
            painter.drawText(cx - 70, cy + 5, "● REC")

        # ── Startup: show status overlay until first frame with data ──
        if not self._frame.has_valid_2d and self._frame.rgb_image is None:
            self._draw_status_overlay(painter)

    def _draw_training_progress(self, painter: QPainter, img_rect: QRect):
        if self._progress_target <= 0:
            return
        width, height = 250, 66
        panel = QRect(
            img_rect.center().x() - width // 2,
            img_rect.bottom() - height - 18,
            width, height,
        )
        painter.setPen(QPen(QColor(255, 255, 255, 35), 1))
        painter.setBrush(QColor(8, 15, 28, 210))
        painter.drawRoundedRect(panel, 14, 14)

        count_font = QFont("Segoe UI", 22, QFont.Bold)
        painter.setFont(count_font)
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(
            QRect(panel.left() + 14, panel.top() + 5, panel.width() - 28, 34),
            Qt.AlignCenter,
            f"{self._progress_count} / {self._progress_target} 次",
        )
        quality_font = QFont("Segoe UI", 10)
        painter.setFont(quality_font)
        painter.setPen(QColor(125, 211, 252))
        painter.drawText(
            QRect(panel.left() + 14, panel.top() + 38, panel.width() - 28, 20),
            Qt.AlignCenter, self._quality_text,
        )

    # ================================================================
    # Status overlay (shown when no frame data)
    # ================================================================

    def _draw_status_overlay(self, painter: QPainter):
        """Draw a diagnostic status screen when there's no frame data."""
        rect = self.rect()
        center_x = rect.center().x()
        center_y = rect.center().y()

        # ---- Title ----
        title_font = QFont("Segoe UI", 20, QFont.Bold)
        painter.setFont(title_font)
        painter.setPen(QColor(200, 210, 220))
        painter.drawText(
            QRect(0, center_y - 80, rect.width(), 40),
            Qt.AlignCenter, "等待训练画面…")

        # ---- Engine mode badge ----
        import time
        badge_font = QFont("Segoe UI", 13)
        painter.setFont(badge_font)

        if self._engine_mode == "STUB":
            badge_text = "🔶 引擎模式: STUB（模拟数据）"
            badge_color = QColor(240, 180, 50)   # amber
        elif self._engine_mode == "FULL":
            badge_text = "🔷 引擎模式: FULL（真实引擎）"
            badge_color = QColor(72, 180, 240)    # blue
        else:
            badge_text = "❓ 引擎模式: 未知"
            badge_color = QColor(200, 200, 200)

        painter.setPen(badge_color)
        painter.drawText(
            QRect(0, center_y - 35, rect.width(), 28),
            Qt.AlignCenter, badge_text)

        # ---- Frame count / timeout warning ----
        info_font = QFont("Segoe UI", 11)
        painter.setFont(info_font)

        if self._frame_count == 0:
            import time
            elapsed = time.monotonic() - self._last_frame_time if self._last_frame_time else 0
            if elapsed > 10 and not self._no_frame_warned:
                self._no_frame_warned = True

            info_text = "按\"开始训练\"启动 Pipeline"
            painter.setPen(QColor(150, 160, 170))
            painter.drawText(
                QRect(0, center_y + 5, rect.width(), 24),
                Qt.AlignCenter, info_text)
        else:
            # We've received frames before
            import time
            elapsed = time.monotonic() - self._last_frame_time
            if elapsed > 5:
                painter.setPen(QColor(246, 90, 90))
                painter.drawText(
                    QRect(0, center_y + 5, rect.width(), 24),
                    Qt.AlignCenter,
                    f"⚠ 画面中断 ({elapsed:.0f}秒无数据)")
            else:
                painter.setPen(QColor(150, 160, 170))
                painter.drawText(
                    QRect(0, center_y + 5, rect.width(), 24),
                    Qt.AlignCenter,
                    f"已接收 {self._frame_count} 帧")

        # ---- Footer: camera status hint ----
        hint_font = QFont("Segoe UI", 10)
        painter.setFont(hint_font)
        painter.setPen(QColor(120, 130, 140))

        hints = []
        import os
        for idx in range(3):
            if os.path.exists(f"/dev/video{idx}"):
                hints.append(f"✓ /dev/video{idx}")
        if not hints:
            hints.append("✗ 未检测到摄像头")

        hint_text = "摄像头: " + "  ".join(hints)
        painter.drawText(
            QRect(0, center_y + 40, rect.width(), 22),
            Qt.AlignCenter, hint_text)

        if self._engine_mode == "STUB":
            painter.setPen(QColor(180, 150, 50))
            painter.drawText(
                QRect(0, center_y + 70, rect.width(), 22),
                Qt.AlignCenter,
                "编译 C++ 引擎后可显示真实画面 (运行 setup_board.sh)")

    # ================================================================
    # Engine mode badge (on active preview)
    # ================================================================

    def _draw_engine_badge(self, painter: QPainter, img_rect: QRect):
        """Small engine mode indicator in bottom-left corner."""
        if not self._engine_mode:
            return

        font = QFont("Consolas", 9)
        painter.setFont(font)
        fm = QFontMetrics(font)

        if self._engine_mode == "STUB":
            text = " STUB "
            bg = QColor(180, 140, 30, 200)
            fg = QColor(255, 255, 255)
        elif self._engine_mode == "FULL":
            text = " FULL "
            bg = QColor(30, 140, 200, 200)
            fg = QColor(255, 255, 255)
        else:
            text = " ? "
            bg = QColor(80, 80, 80, 200)
            fg = QColor(200, 200, 200)

        tw = fm.horizontalAdvance(text) + 8
        th = fm.height() + 4
        x = img_rect.left() + 10
        y = img_rect.bottom() - th - 10

        badge_rect = QRect(x, y, tw, th)
        painter.setPen(Qt.NoPen)
        painter.setBrush(bg)
        painter.drawRoundedRect(badge_rect, 4, 4)

        painter.setPen(fg)
        painter.drawText(badge_rect, Qt.AlignCenter, text)

    # ================================================================
    # Skeleton drawing
    # ================================================================

    def _draw_skeleton(self, painter: QPainter, img_rect: QRect):
        joints = self._frame.joints_2d
        joints_3d = self._frame.joints_3d

        # Helper to map joint from source (640x480) to draw rect
        def map_joint(idx: int) -> QPointF:
            if idx >= len(joints):
                return QPointF(0, 0)
            j = joints[idx]
            x = j.x
            if self._mirror:
                x = 640 - x
            px = img_rect.left() + x * img_rect.width() / 640.0
            py = img_rect.top() + j.y * img_rect.height() / 480.0
            return QPointF(px, py)

        def is_valid(idx: int) -> bool:
            if idx >= len(joints):
                return False
            j = joints[idx]
            return j.valid and j.score >= self._score_threshold

        def has_3d(idx: int) -> bool:
            if idx >= len(joints_3d):
                return False
            return joints_3d[idx].valid

        # Draw bones
        bone_pen = QPen(QColor(255, 215, 64), 2.5, Qt.SolidLine, Qt.RoundCap)
        painter.setPen(bone_pen)
        for i0, i1 in BONES:
            if is_valid(i0) and is_valid(i1):
                painter.drawLine(map_joint(i0), map_joint(i1))

        # Draw joints
        for i in range(min(len(joints), 22)):
            if not is_valid(i):
                continue
            pt = map_joint(i)
            is_3d = has_3d(i)

            # Circle
            color = QColor(72, 221, 120) if is_3d else QColor(246, 90, 90)
            painter.setBrush(color)
            painter.setPen(QPen(QColor(8, 10, 12), 1.0))
            painter.drawEllipse(pt, 5, 5)

            # Joint index
            if self._show_joint_index:
                painter.setPen(QColor(255, 255, 255))
                painter.setBrush(Qt.NoBrush)
                font = QFont("Segoe UI", 8)
                painter.setFont(font)
                painter.drawText(pt + QPointF(7, -5), str(i))

    # ================================================================
    # Debug HUD
    # ================================================================

    def _draw_debug_panel(self, painter: QPainter, img_rect: QRect):
        f = self._frame
        lines = [
            f"RGB {f.rgb_fps:.1f} fps  Depth {f.depth_fps:.1f} fps",
            f"Pair {f.pair_fps:.1f} fps  delta {f.delta_ms:.1f}ms",
            f"Pose {f.pose_fps:.1f} fps  YOLO {f.yolo_ms:.1f}ms  Pose {f.pose_ms:.1f}ms",
            f"Queue {f.queue_length}  Dropped {f.dropped_pairs}  BBox {f.bbox_mode}",
            f"3D valid: {sum(1 for j in f.joints_3d if j.valid) if f.joints_3d else 0}/22",
            f"Frame #{self._frame_count}",
        ]

        font = QFont("Consolas", 9)
        painter.setFont(font)
        fm = QFontMetrics(font)
        line_h = fm.height() + 2
        padding = 8

        text_width = max(fm.horizontalAdvance(line) for line in lines) + padding * 2
        text_height = line_h * len(lines) + padding * 2

        panel_rect = QRect(
            img_rect.left() + 10, img_rect.top() + 10,
            text_width, text_height,
        )
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 160))
        painter.drawRoundedRect(panel_rect, 6, 6)

        painter.setPen(QColor(240, 245, 250))
        y = panel_rect.top() + padding + fm.ascent()
        for line in lines:
            painter.drawText(panel_rect.left() + padding, y, line)
            y += line_h

        # Right-side status
        right_lines = [
            "人体: " + ("已检测" if f.has_valid_2d else "未检测"),
            "3D: " + ("有效" if f.has_valid_3d else "等待"),
            self._engine_mode + " 模式" if self._engine_mode else "",
        ]
        right_lines = [l for l in right_lines if l]
        if self._recording:
            right_lines.append("REC ●")

        rw = max(fm.horizontalAdvance(l) for l in right_lines) + padding * 2
        rh = line_h * len(right_lines) + padding * 2
        rp = QRect(img_rect.right() - rw - 10, img_rect.top() + 10, rw, rh)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 160))
        painter.drawRoundedRect(rp, 6, 6)
        painter.setPen(QColor(240, 245, 250))
        y = rp.top() + padding + fm.ascent()
        for line in right_lines:
            painter.drawText(rp.left() + padding, y, line)
            y += line_h

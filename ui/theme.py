"""Shared visual tokens and small stylesheet helpers for the desktop UI."""

from typing import Dict


COLORS: Dict[str, str] = {
    "primary": "#2563EB",
    "primary_dark": "#1D4ED8",
    "primary_soft": "#EAF2FF",
    "cyan": "#0891B2",
    "success": "#059669",
    "success_soft": "#E8F7F1",
    "warning": "#D97706",
    "warning_soft": "#FFF5E6",
    "danger": "#DC2626",
    "danger_soft": "#FEECEC",
    "ink": "#172033",
    "text": "#344054",
    "muted": "#667085",
    "border": "#E4EAF2",
    "surface": "#FFFFFF",
    "canvas": "#F4F7FB",
}


PAGE_STYLE = f"""
QWidget#trainingPage, QWidget#reportsPage, QScrollArea#settingsPage {{
    background: {COLORS['canvas']};
}}
QLabel#pageEyebrow {{
    color: {COLORS['primary']};
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
}}
QLabel#pageTitle {{
    color: {COLORS['ink']};
    font-size: 24px;
    font-weight: 700;
}}
QLabel#pageSubtitle, QLabel#sectionHint {{
    color: {COLORS['muted']};
    font-size: 12px;
}}
QLabel#sectionTitle {{
    color: {COLORS['ink']};
    font-size: 15px;
    font-weight: 700;
}}
QLabel#metricLabel {{
    color: {COLORS['muted']};
    font-size: 11px;
}}
QLabel#metricValue {{
    color: {COLORS['ink']};
    font-size: 17px;
    font-weight: 700;
}}
QTextEdit#feedbackLog {{
    background: #F8FAFD;
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    color: {COLORS['text']};
    padding: 6px;
    selection-background-color: {COLORS['primary_soft']};
}}
QProgressBar {{
    min-height: 8px;
    max-height: 8px;
    border: none;
    border-radius: 4px;
    background: #E9EEF5;
    color: transparent;
}}
QProgressBar::chunk {{
    border-radius: 4px;
    background: {COLORS['primary']};
}}
QCheckBox {{
    color: {COLORS['text']};
    spacing: 8px;
}}
"""


def state_badge_style(state_name: str) -> str:
    palettes = {
        "TRAINING": (COLORS["success"], COLORS["success_soft"]),
        "RESTING": (COLORS["warning"], COLORS["warning_soft"]),
        "PAUSED": (COLORS["warning"], COLORS["warning_soft"]),
        "FINISHED": (COLORS["primary"], COLORS["primary_soft"]),
        "IDLE": (COLORS["muted"], "#EEF2F6"),
    }
    foreground, background = palettes.get(state_name, palettes["IDLE"])
    return (
        f"color:{foreground}; background:{background}; border-radius:10px; "
        "padding:4px 12px; font-size:12px; font-weight:700;"
    )


def pill_style(kind: str = "primary") -> str:
    palettes = {
        "primary": (COLORS["primary"], COLORS["primary_soft"]),
        "success": (COLORS["success"], COLORS["success_soft"]),
        "warning": (COLORS["warning"], COLORS["warning_soft"]),
        "neutral": (COLORS["muted"], "#EEF2F6"),
    }
    foreground, background = palettes[kind]
    return (
        f"color:{foreground}; background:{background}; border-radius:9px; "
        "padding:3px 10px; font-size:11px; font-weight:600;"
    )

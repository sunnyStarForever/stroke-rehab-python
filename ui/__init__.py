"""UI package initialization and Linux distribution compatibility."""

# Ubuntu 20.04 packages SIP 4 as a top-level ``sip`` module, while recent
# qframelesswindow imports it as ``PyQt5.sip``.  Alias the same binary module
# without modifying the system Python installation.
try:
    from PyQt5 import sip as _pyqt_sip  # type: ignore  # noqa: F401
except ImportError:
    try:
        import sys
        import sip as _legacy_sip  # type: ignore
        import PyQt5

        PyQt5.sip = _legacy_sip
        sys.modules.setdefault("PyQt5.sip", _legacy_sip)
    except ImportError:
        pass

# QFluentWidgets 1.11 imports QCalendar, which is unavailable in the PyQt5
# bindings shipped by Ubuntu 20.04 on the Phytium Pi.  Its calendar view only
# needs monthName(), so provide that API through the older QLocale equivalent.
try:
    from PyQt5.QtCore import QCalendar as _QCalendar  # type: ignore  # noqa: F401
except ImportError:
    try:
        from PyQt5 import QtCore

        class _CompatQCalendar:
            def monthName(self, locale, month, year=None):
                del year
                return locale.monthName(month, QtCore.QLocale.LongFormat)

        QtCore.QCalendar = _CompatQCalendar
    except (ImportError, AttributeError):
        pass

# Qt 5.13 added QFont.setFamilies(); QFluentWidgets uses it for fallback font
# lists.  Qt 5.12 can retain the primary family with setFamily().
try:
    from PyQt5.QtGui import QFont

    if not hasattr(QFont, "setFamilies"):
        def _set_families(self, families):
            if families:
                self.setFamily(families[0])

        QFont.setFamilies = _set_families
except ImportError:
    pass

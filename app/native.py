"""Platform dispatch facade — the single ``sys.platform`` site in the app.

Shared code imports ``app.native`` and reaches every platform capability
through its sub-namespaces::

    native.io.paste_text(...)        input injection / clipboard / monitors
    native.util.show_info(...)       autostart / single instance / dialogs
    native.hotkey.HotkeyHook(...)    global push-to-talk hook
    native.mousehook.MouseHook(...)  mouse side-button hook
    native.indicator.Indicator()     "正在听…" status bar
    native.uia.probe_focused()       UI-Automation style value writes
    native.data_root()               per-user data directory

Windows re-exports the existing ``win*`` modules; macOS re-exports the
``mac*`` modules. No other module may branch on the platform.
"""

import os
import sys
from pathlib import Path

APP_DIR_NAME = "xxl-whisper"

if sys.platform == "win32":
    from app import hotkey, indicator, mousehook, uia
    from app import winio as io
    from app import winutil as util
else:
    from app import mac_indicator as indicator
    from app import machotkey as hotkey
    from app import macio as io
    from app import macmousehook as mousehook
    from app import macuia as uia
    from app import macutil as util

__all__ = ["data_root", "hotkey", "indicator", "io", "mousehook", "uia", "util"]


def data_root() -> Path:
    """Return the per-user data root (config, logs, models)."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        root = Path(base) if base else Path.home() / "AppData" / "Local"
        return root / APP_DIR_NAME
    return Path.home() / "Library" / "Application Support" / APP_DIR_NAME

"""Platform dispatch facade — the single ``sys.platform`` site in the app.

Shared code imports ``app.native`` and reaches every platform capability
through its sub-namespaces::

    native.io.paste_text(...)        input injection / clipboard / monitors
    native.util.show_info(...)       autostart / single instance / dialogs
    native.hotkey.HotkeyHook(...)    global push-to-talk hook
    native.mousehook.MouseHook(...)  mouse side-button hook
    native.indicator.Indicator()     "正在听…" status bar
    native.overlay.Overlay()         "上屏字幕" subtitle bar (platform-agnostic)
    native.uia.probe_focused()       UI-Automation style value writes
    native.data_root()               per-user data directory

Windows re-exports the existing ``win*`` modules; macOS re-exports the
``mac*`` modules. No other module may branch on the platform.
"""

import os
import sys
from pathlib import Path
from types import ModuleType
from typing import NoReturn, override

APP_DIR_NAME = "xxl-whisper"


class _Unavailable:
    """Stand-in for one backend attribute on an unsupported platform.

    Deliberately *not* raising on attribute read: pytest's ``monkeypatch.setattr``
    reads the old value before overwriting it, and a backend that explodes on
    read would make every emit/hotkey test unrunnable off Windows/macOS. The
    error surfaces when the attribute is actually used.
    """

    def __init__(self, backend: str, name: str) -> None:
        self._backend = backend
        self._name = name

    def __call__(self, *_args: object, **_kwargs: object) -> NoReturn:
        self._explain()

    def __getattr__(self, name: str) -> NoReturn:
        self._explain()

    def _explain(self) -> NoReturn:
        msg = (
            f"app.{self._backend}.{self._name} is unavailable on {sys.platform!r}: "
            "xxl-whisper ships Windows and macOS backends only"
        )
        raise RuntimeError(msg)

    @override
    def __repr__(self) -> str:
        return f"<unavailable app.{self._backend}.{self._name}>"


def _unsupported_backend(backend: str) -> ModuleType:
    """Return a stub backend module whose every attribute is unavailable."""
    module = ModuleType(f"app.{backend}")

    def _unavailable(name: str) -> _Unavailable:
        return _Unavailable(backend, name)

    module.__getattr__ = _unavailable
    return module


if sys.platform == "win32":
    from app import hotkey, indicator, mousehook, overlay, uia
    from app import winio as io
    from app import winutil as util
elif sys.platform == "darwin":
    from app import mac_indicator as indicator
    from app import machotkey as hotkey
    from app import macio as io
    from app import macmousehook as mousehook
    from app import macuia as uia
    from app import macutil as util
    from app import overlay
else:
    # Unsupported platform (Linux CI, headless dev boxes): importing must still
    # succeed so the platform-independent core (config, filtering, batching,
    # emit routing) stays importable and unit-testable. Reaching any real
    # backend attribute fails loudly with the platform name instead of the
    # AttributeError (or, worse, the macOS import error) that a bare ``else``
    # treating every non-Windows platform as macOS used to produce.
    from app import overlay

    indicator = _unsupported_backend("indicator")
    hotkey = _unsupported_backend("hotkey")
    io = _unsupported_backend("io")
    mousehook = _unsupported_backend("mousehook")
    uia = _unsupported_backend("uia")
    util = _unsupported_backend("util")

#: Windows keeps the capture stream open for the app's lifetime (invisible
#: cost, saves the 50-150 ms device-open from the push-to-talk budget). macOS
#: opens it per hold: an open stream keeps the system microphone indicator lit
#: (measured open cost ~68 ms).
RECORDER_KEEP_OPEN: bool = sys.platform == "win32"

#: The code the capture hook reports for Escape: 0x1B (VK_ESCAPE) on Windows,
#: 0x35 (kVK_Escape) on macOS. Sharing the Windows constant made Esc unable to
#: cancel a hotkey capture on macOS — and turned Esc itself into a hotkey.
ESCAPE_VK: int = 0x1B if sys.platform == "win32" else 0x35

__all__ = [
    "ESCAPE_VK",
    "RECORDER_KEEP_OPEN",
    "data_root",
    "hotkey",
    "indicator",
    "io",
    "mousehook",
    "overlay",
    "uia",
    "util",
]


def data_root() -> Path:
    """Return the per-user data root (config, logs, models)."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        root = Path(base) if base else Path.home() / "AppData" / "Local"
        return root / APP_DIR_NAME
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_DIR_NAME
    # Unsupported platform (CI): still give the pure logic a writable root.
    base = os.environ.get("XDG_DATA_HOME")
    root = Path(base) if base else Path.home() / ".local" / "share"
    return root / APP_DIR_NAME

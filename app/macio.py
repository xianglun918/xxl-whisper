"""macOS input injection / clipboard facade.

M3 fills the minimal surface the dictation loop needs (clipboard swap +
Cmd+V injection, injection liveness, key names, frontmost app); the remaining
functions (``type_text``, screen work area, fullscreen detection) land in M4.
Mirrors ``app/winio.py`` so shared code reaches it via ``app.native.io``.

The Cmd+V sequence and pasteboard handling are the ones validated by
``scripts/probe_mac_paste.py`` (M1): ``CombinedSessionState`` source posted to
``kCGSessionEventTap``. pyobjc is imported with ``importlib`` (M1 constraint).
"""

import importlib
import logging
import time
from collections.abc import Mapping
from types import MappingProxyType

AppKit = importlib.import_module("AppKit")
Quartz = importlib.import_module("Quartz")

log = logging.getLogger(__name__)

_PB_TYPE = AppKit.NSPasteboardTypeString
_KEY_V = 0x09
_KEY_RCMD = 0x36

#: mac keycodes -> display names for tray/diagnostics labels.
_KEY_NAMES: Mapping[int, str] = MappingProxyType(
    {
        0x36: "右 Command",
        0x37: "左 Command",
        0x3C: "右 Shift",
        0x38: "左 Shift",
        0x3D: "右 Option",
        0x3A: "左 Option",
        0x3E: "右 Control",
        0x3B: "左 Control",
        0x7A: "F1",
        0x78: "F2",
        0x63: "F3",
        0x76: "F4",
        0x60: "F5",
        0x61: "F6",
        0x62: "F7",
        0x64: "F8",
        0x65: "F9",
        0x6D: "F10",
        0x67: "F11",
        0x6F: "F12",
        0x24: "Return",
        0x30: "Tab",
        0x31: "Space",
        0x33: "Delete",
        0x35: "Escape",
    }
)


class PasteError(Exception):
    """Raised when the macOS paste channel fails."""

    def __init__(self, api: str, code: int) -> None:
        super().__init__(f"{api} failed ({code})")
        self.api = api
        self.code = code


def _snapshot_pasteboard() -> list[tuple[str, bytes]]:
    """Capture every type on the general pasteboard for later restore."""
    pasteboard = AppKit.NSPasteboard.generalPasteboard()
    snapshot: list[tuple[str, bytes]] = []
    for ptype in pasteboard.types() or []:
        data = pasteboard.dataForType_(ptype)
        if data is not None:
            snapshot.append((str(ptype), bytes(data)))
    return snapshot


def _restore_pasteboard(snapshot: list[tuple[str, bytes]]) -> None:
    """Put a snapshot back onto the general pasteboard."""
    pasteboard = AppKit.NSPasteboard.generalPasteboard()
    pasteboard.clearContents()
    for ptype, data in snapshot:
        pasteboard.setData_forType_(AppKit.NSData.dataWithBytes_length_(data, len(data)), ptype)


def _post_cmd_v() -> None:
    """Post Cmd+V to the session tap (M1's winning configuration)."""
    source = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateCombinedSessionState)
    command = Quartz.kCGEventFlagMaskCommand

    event = Quartz.CGEventCreateKeyboardEvent(source, _KEY_RCMD, True)
    Quartz.CGEventSetType(event, Quartz.kCGEventFlagsChanged)
    Quartz.CGEventSetFlags(event, command)
    Quartz.CGEventPost(Quartz.kCGSessionEventTap, event)
    time.sleep(0.02)

    for down in (True, False):
        event = Quartz.CGEventCreateKeyboardEvent(source, _KEY_V, down)
        Quartz.CGEventSetFlags(event, command)
        Quartz.CGEventPost(Quartz.kCGSessionEventTap, event)
        time.sleep(0.02)

    event = Quartz.CGEventCreateKeyboardEvent(source, _KEY_RCMD, False)
    Quartz.CGEventSetType(event, Quartz.kCGEventFlagsChanged)
    Quartz.CGEventSetFlags(event, 0)
    Quartz.CGEventPost(Quartz.kCGSessionEventTap, event)


def tap_key(vk: int) -> None:
    """No-op: the tap is listen-only, so a quick tap already passed through."""
    log.debug("tap_key(%d) is a no-op on macOS", vk)


def paste_text(text: str, restore_clipboard: bool, delay_ms: int) -> None:
    """Clipboard swap + Cmd+V injection into the frontmost app.

    Raises :class:`PasteError` when synthetic keystrokes are not permitted
    (Accessibility not granted), so the emit ladder falls back gracefully.
    """
    if not Quartz.CGPreflightPostEventAccess():
        raise PasteError(api="CGPreflightPostEventAccess", code=0)
    snapshot = _snapshot_pasteboard() if restore_clipboard else None
    set_clipboard_text(text)
    _post_cmd_v()
    if snapshot is not None:
        time.sleep(max(delay_ms, 50) / 1000)
        _restore_pasteboard(snapshot)


def type_text(text: str) -> None:
    """Unicode typing fallback."""
    msg = f"macio.type_text({len(text)} chars) lands in M4 (mac native surface)"
    raise NotImplementedError(msg)


def foreground_window_title() -> str:
    """Localized name of the frontmost application."""
    front = AppKit.NSWorkspace.sharedWorkspace().frontmostApplication()
    return str(front.localizedName()) if front is not None else ""


def set_clipboard_text(text: str) -> None:
    """Replace the general pasteboard contents with ``text``."""
    pasteboard = AppKit.NSPasteboard.generalPasteboard()
    pasteboard.clearContents()
    pasteboard.setString_forType_(text, _PB_TYPE)


def keyboard_injection_alive() -> bool:
    """Whether synthetic keystrokes can be posted (Accessibility granted)."""
    return bool(Quartz.CGPreflightPostEventAccess())


def post_wm_paste_to_focus() -> bool:
    """No WM_PASTE on macOS; the channel is never available."""
    return False


def active_monitor_work_area() -> tuple[int, int, int, int]:
    """Work area (x, y, width, height) of the active screen."""
    msg = "macio.active_monitor_work_area lands in M4 (mac native surface)"
    raise NotImplementedError(msg)


def exclusive_fullscreen_owner_active() -> bool:
    """Whether an exclusive-fullscreen app owns the active screen."""
    msg = "macio.exclusive_fullscreen_owner_active lands in M4 (mac native surface)"
    raise NotImplementedError(msg)


def key_name(vk: int) -> str:
    """Human-readable name for a mac keycode."""
    return _KEY_NAMES.get(vk, f"键码 {vk}")


def tap_mouse_x(vk: int) -> None:
    """No-op: macOS drops mouse side-button hotkeys."""
    log.debug("tap_mouse_x(%d) is a no-op on macOS", vk)


def focused_control_class() -> str:
    """No Win32 control classes on macOS; the WM_PASTE channel never applies."""
    return ""

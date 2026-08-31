"""Win32 input injection: SendInput keystrokes and clipboard swap-paste.

Everything here is a thin, typed wrapper over ctypes calls used by the ASR
worker thread: tapping the original hotkey (click passthrough), and pasting
recognized text via clipboard + Ctrl+V with best-effort clipboard restore.
"""

import ctypes
import time
from ctypes import wintypes

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

_INPUT_KEYBOARD: int = 1
_KEYEVENTF_KEYUP: int = 0x0002
_KEYEVENTF_UNICODE: int = 0x0004
_VK_CONTROL: int = 0x11
_VK_V: int = 0x56
_VK_RETURN: int = 0x0D
_VK_F13: int = 0x7C
_WM_PASTE: int = 0x0302
_CF_UNICODETEXT: int = 13
_GMEM_MOVEABLE: int = 0x0002
_CLIPBOARD_OPEN_RETRIES: int = 20
_CLIPBOARD_OPEN_RETRY_DELAY_S: float = 0.05
_UNICODE_EVENTS_PER_CHAR: int = 2  # key down + key up


class PasteError(Exception):
    """Raised when the clipboard or SendInput path fails irrecoverably."""

    def __init__(self, api: str, code: int) -> None:
        super().__init__(f"{api} failed with Win32 error {code}")
        self.api = api
        self.code = code


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = (
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    )


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = (
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    )


class _HARDWAREINPUT(ctypes.Structure):
    _fields_ = (
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    )


class _INPUTUNION(ctypes.Union):
    _fields_ = (
        ("mi", _MOUSEINPUT),
        ("ki", _KEYBDINPUT),
        ("hi", _HARDWAREINPUT),
    )


class _INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = (("type", wintypes.DWORD), ("u", _INPUTUNION))


# 64-bit pointer hygiene: every API that crosses a pointer/HANDLE must declare
# restype/argtypes, or ctypes truncates to 32-bit ints (-> access violations
# on machines whose heap lives above 4 GB).
user32.OpenClipboard.argtypes = (wintypes.HWND,)
user32.OpenClipboard.restype = wintypes.BOOL
user32.CloseClipboard.restype = wintypes.BOOL
user32.EmptyClipboard.restype = wintypes.BOOL
user32.GetClipboardData.argtypes = (wintypes.UINT,)
user32.GetClipboardData.restype = wintypes.HANDLE
user32.SetClipboardData.argtypes = (wintypes.UINT, wintypes.HANDLE)
user32.SetClipboardData.restype = wintypes.HANDLE
kernel32.GlobalAlloc.argtypes = (wintypes.UINT, ctypes.c_size_t)
kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
kernel32.GlobalLock.argtypes = (wintypes.HGLOBAL,)
kernel32.GlobalLock.restype = ctypes.c_void_p
kernel32.GlobalUnlock.argtypes = (wintypes.HGLOBAL,)
kernel32.GlobalFree.argtypes = (wintypes.HGLOBAL,)
user32.GetForegroundWindow.restype = ctypes.c_void_p
user32.GetWindowTextW.argtypes = (ctypes.c_void_p, wintypes.LPWSTR, ctypes.c_int)
user32.GetAsyncKeyState.restype = ctypes.c_short
user32.GetAsyncKeyState.argtypes = (ctypes.c_int,)
user32.MapVirtualKeyW.restype = wintypes.UINT
user32.MapVirtualKeyW.argtypes = (wintypes.UINT, wintypes.UINT)
user32.GetKeyNameTextW.restype = ctypes.c_int
user32.GetKeyNameTextW.argtypes = (wintypes.LPARAM, wintypes.LPWSTR, ctypes.c_int)
shell32 = ctypes.WinDLL("shell32")
shell32.SHQueryUserNotificationState.argtypes = (ctypes.POINTER(ctypes.c_int),)
shell32.SHQueryUserNotificationState.restype = ctypes.HRESULT

_QUNS_RUNNING_D3D_FULL_SCREEN: int = 3
_QUNS_PRESENTATION_MODE: int = 4
user32.PostMessageW.restype = wintypes.BOOL
user32.PostMessageW.argtypes = (ctypes.c_void_p, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)
user32.GetWindowThreadProcessId.restype = wintypes.DWORD
user32.GetWindowThreadProcessId.argtypes = (ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD))
user32.GetGUIThreadInfo.restype = wintypes.BOOL
user32.GetGUIThreadInfo.argtypes = (wintypes.DWORD, ctypes.c_void_p)


def tap_key(vk: int) -> None:
    """Send one non-injected-looking key tap (down + up) via SendInput."""
    _send_key(vk, up=False)
    _send_key(vk, up=True)


def paste_text(text: str, restore_clipboard: bool, delay_ms: int) -> None:
    """Put ``text`` on the clipboard, Ctrl+V it, then restore the old content.

    ``delay_ms`` lets the target application read the clipboard before we
    swap the previous content back in.
    """
    previous = _clipboard_text() if restore_clipboard else None
    _set_clipboard_text(text)
    _press_ctrl_v()
    time.sleep(delay_ms / 1000)
    if previous is not None:
        _set_clipboard_text(previous)


def type_text(text: str) -> None:
    """Clipboard-free fallback: type text via KEYEVENTF_UNICODE packets.

    Slower than paste and ignores the target's key bindings, but immune to
    clipboard locks (security software, clipboard managers, RDP glue).
    """
    for char in text:
        if char == "\n":
            tap_key(_VK_RETURN)
            continue
        _send_unicode(char)


def _send_unicode(char: str) -> None:
    """Send one character as a down+up VK_PACKET pair."""
    entry = _INPUT(type=_INPUT_KEYBOARD)
    entry.wVk = 0
    entry.wScan = ord(char)
    entry.dwFlags = _KEYEVENTF_UNICODE
    sent = user32.SendInput(1, ctypes.byref(entry), ctypes.sizeof(_INPUT))
    entry.dwFlags = _KEYEVENTF_UNICODE | _KEYEVENTF_KEYUP
    sent += user32.SendInput(1, ctypes.byref(entry), ctypes.sizeof(_INPUT))
    if sent != _UNICODE_EVENTS_PER_CHAR:
        raise PasteError(api="SendInput(unicode)", code=ctypes.get_last_error())


def foreground_window_title() -> str:
    """Title of the window that will receive injected keys right now."""
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return "<no foreground window>"
    buffer = ctypes.create_unicode_buffer(256)
    user32.GetWindowTextW(hwnd, buffer, 256)
    return buffer.value or "<untitled>"


def set_clipboard_text(text: str) -> None:
    """Public clipboard setter (the emit chain always stages text here)."""
    _set_clipboard_text(text)


class _GUITHREADINFO(ctypes.Structure):
    _fields_ = (
        ("cbSize", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("hwndActive", ctypes.c_void_p),
        ("hwndFocus", ctypes.c_void_p),
        ("hwndCapture", ctypes.c_void_p),
        ("hwndMenuOwner", ctypes.c_void_p),
        ("hwndMoveSize", ctypes.c_void_p),
        ("hwndCaret", ctypes.c_void_p),
        ("rcCaret", wintypes.RECT),
    )


def keyboard_injection_alive() -> bool:
    """Probe whether synthetic keyboard events survive this machine.

    Some resident software (uTools/Doubao/ArmouryCrate/G HUB class apps with
    misbehaving WH_KEYBOARD_LL hooks) swallows injected events before the
    input state updates. Taps harmless F13 and checks the async key state.
    """
    tap_key(_VK_F13)
    time.sleep(0.01)
    return bool(user32.GetAsyncKeyState(_VK_F13) & 0x8000)


def focused_control_hwnd() -> int:
    """HWND of the keyboard-focused control in the foreground window."""
    foreground = user32.GetForegroundWindow()
    if not foreground:
        return 0
    thread_id = user32.GetWindowThreadProcessId(foreground, None)
    info = _GUITHREADINFO(cbSize=ctypes.sizeof(_GUITHREADINFO))
    if not user32.GetGUIThreadInfo(thread_id, ctypes.byref(info)):
        return 0
    return info.hwndFocus or 0


def post_wm_paste_to_focus() -> bool:
    """Post WM_PASTE to the focused control; bypasses the input stream.

    Works for classic EDIT/RichEdit surfaces even when injected keyboard
    events are being filtered system-wide. Returns False when no focused
    control could be resolved.
    """
    focus = focused_control_hwnd()
    if not focus:
        return False
    return bool(user32.PostMessageW(focus, _WM_PASTE, 0, 0))


class _MONITORINFO(ctypes.Structure):
    _fields_ = (
        ("cbSize", wintypes.DWORD),
        ("rcMonitor", wintypes.RECT),
        ("rcWork", wintypes.RECT),
        ("dwFlags", wintypes.DWORD),
    )


def active_monitor_work_area() -> tuple[int, int, int, int]:
    """Work area (left, top, right, bottom) of the active screen.

    "Active" = the monitor holding the foreground window, i.e. the screen
    the user is actually looking at (multi-monitor aware).
    """
    monitor = user32.MonitorFromWindow(user32.GetForegroundWindow(), 1)  # NEAREST
    info = _MONITORINFO(cbSize=ctypes.sizeof(_MONITORINFO))
    if not user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
        return (0, 0, user32.GetSystemMetrics(16), user32.GetSystemMetrics(17))
    work = info.rcWork
    return (work.left, work.top, work.right, work.bottom)


def exclusive_fullscreen_owner_active() -> bool:
    """Report whether an exclusive-fullscreen app owns the screen right now.

    Detects fullscreen 3D apps (games) and presentation mode via the OS
    notification state (the same signal overlays like Discord consult):
    showing a topmost bar in these modes would fight the game for the
    screen, so the indicator must stay hidden instead.
    """
    state = ctypes.c_int(0)
    shell32.SHQueryUserNotificationState(ctypes.byref(state))
    return state.value in (_QUNS_RUNNING_D3D_FULL_SCREEN, _QUNS_PRESENTATION_MODE)


_MAPVK_VK_TO_VSC_EX: int = 4  # 3 is VSC_TO_VK_EX; do not confuse them
_EXTENDED_SCAN_PREFIX: int = 0xE000
#: VKs that are physically extended (E0-prefixed) even when the EX mapping
#: omits the prefix — needed so arrows don't display as numpad keys.
_EXTENDED_VKS: frozenset[int] = frozenset(
    {0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x2C, 0x2D, 0x2E, 0x5B,
     0x5C, 0x5D, 0x6F, 0x90, 0xA3, 0xA5}
)


def key_name(vk: int) -> str:
    """Human-readable key name for a virtual-key code (for menus/logs)."""
    scan = user32.MapVirtualKeyW(vk, _MAPVK_VK_TO_VSC_EX)
    lparam = (scan & 0xFF) << 16
    if (scan & 0xFF00) == _EXTENDED_SCAN_PREFIX or vk in _EXTENDED_VKS:
        lparam |= 1 << 24
    buffer = ctypes.create_unicode_buffer(64)
    written = user32.GetKeyNameTextW(lparam, buffer, 64)
    if written > 0:
        return buffer.value
    return f"VK 0x{vk:02X}"


def _send_key(vk: int, *, up: bool) -> None:
    entry = _INPUT(type=_INPUT_KEYBOARD)
    entry.wVk = vk
    entry.dwFlags = _KEYEVENTF_KEYUP if up else 0
    sent = user32.SendInput(1, ctypes.byref(entry), ctypes.sizeof(_INPUT))
    if sent != 1:
        raise PasteError(api="SendInput", code=ctypes.get_last_error())


def _press_ctrl_v() -> None:
    _send_key(_VK_CONTROL, up=False)
    _send_key(_VK_V, up=False)
    _send_key(_VK_V, up=True)
    _send_key(_VK_CONTROL, up=True)


def _open_clipboard() -> None:
    for _ in range(_CLIPBOARD_OPEN_RETRIES):
        if user32.OpenClipboard(None):
            return
        time.sleep(_CLIPBOARD_OPEN_RETRY_DELAY_S)
    raise PasteError(api="OpenClipboard", code=ctypes.get_last_error())


def _clipboard_text() -> str | None:
    _open_clipboard()
    try:
        handle = user32.GetClipboardData(_CF_UNICODETEXT)
        if not handle:
            return None
        ptr = kernel32.GlobalLock(handle)
        if not ptr:
            raise PasteError(api="GlobalLock", code=ctypes.get_last_error())
        try:
            return ctypes.wstring_at(ptr)
        finally:
            kernel32.GlobalUnlock(handle)
    finally:
        user32.CloseClipboard()


def _set_clipboard_text(text: str) -> None:
    payload = ctypes.create_unicode_buffer(text)
    byte_size = ctypes.sizeof(payload)  # includes the NUL terminator
    _open_clipboard()
    try:
        if not user32.EmptyClipboard():
            raise PasteError(api="EmptyClipboard", code=ctypes.get_last_error())
        handle = kernel32.GlobalAlloc(_GMEM_MOVEABLE, byte_size)
        if not handle:
            raise PasteError(api="GlobalAlloc", code=ctypes.get_last_error())
        ptr = kernel32.GlobalLock(handle)
        if not ptr:
            raise PasteError(api="GlobalLock", code=ctypes.get_last_error())
        try:
            ctypes.memmove(ptr, payload, byte_size)
        finally:
            kernel32.GlobalUnlock(handle)
        if not user32.SetClipboardData(_CF_UNICODETEXT, handle):
            kernel32.GlobalFree(handle)
            raise PasteError(api="SetClipboardData", code=ctypes.get_last_error())
    finally:
        user32.CloseClipboard()

"""Mouse side-button hotkey via WH_MOUSE_LL on a dedicated message-pump thread.

Watches XButton1/XButton2 (VK_XBUTTON1=0x05, VK_XBUTTON2=0x06). The target
button is suppressed system-wide while armed; injected events are passed
through so the app can re-synthesize the native back/forward click.
"""

import ctypes
import logging
import threading
from collections.abc import Callable
from ctypes import wintypes
from typing import override

log = logging.getLogger(__name__)

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

_WH_MOUSE_LL: int = 14
_WM_XBUTTONDOWN: int = 0x020B
_WM_XBUTTONUP: int = 0x020C
_WM_QUIT: int = 0x0012
_LLMHF_INJECTED: int = 0x01
_LLMHF_LOWER_IL_INJECTED: int = 0x02

VK_XBUTTON1: int = 0x05
VK_XBUTTON2: int = 0x06


class MouseHookError(Exception):
    """Raised when the low-level mouse hook cannot be installed."""

    def __init__(self, code: int) -> None:
        super().__init__(f"SetWindowsHookExW failed with Win32 error {code}")
        self.code = code


class _MSLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = (
        ("pt", wintypes.POINT),
        ("mouseData", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    )


class _MSG(ctypes.Structure):
    _fields_ = (
        ("hwnd", wintypes.HWND),
        ("message", wintypes.UINT),
        ("wParam", wintypes.WPARAM),
        ("lParam", wintypes.LPARAM),
        ("time", wintypes.DWORD),
        ("pt", wintypes.POINT),
    )


_LRESULT = ctypes.c_ssize_t
_HOOKPROC = ctypes.WINFUNCTYPE(_LRESULT, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)
_LLHOOKPTR = ctypes.POINTER(_MSLLHOOKSTRUCT)

user32.SetWindowsHookExW.restype = wintypes.HHOOK
user32.SetWindowsHookExW.argtypes = (ctypes.c_int, _HOOKPROC, wintypes.HINSTANCE, wintypes.DWORD)
user32.CallNextHookEx.restype = _LRESULT
user32.CallNextHookEx.argtypes = (wintypes.HHOOK, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)
user32.UnhookWindowsHookEx.restype = wintypes.BOOL
user32.UnhookWindowsHookEx.argtypes = (wintypes.HHOOK,)
user32.PostThreadMessageW.restype = wintypes.BOOL
user32.PostThreadMessageW.argtypes = (
    wintypes.DWORD,
    wintypes.UINT,
    wintypes.WPARAM,
    wintypes.LPARAM,
)
user32.GetMessageW.argtypes = (ctypes.POINTER(_MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT)
kernel32.GetCurrentThreadId.restype = wintypes.DWORD


class MouseHook(threading.Thread):
    """Daemon thread owning the hook; emits button transitions for X1/X2.

    ``on_transition(pressed)`` runs on the hook thread — it must only enqueue.
    vk=0 disables interception entirely (all events pass through).
    """

    def __init__(self, vk: int, on_transition: Callable[[bool], None]) -> None:
        super().__init__(daemon=True, name="mouse-hook")
        self._vk = vk
        self._on_transition = on_transition
        self._is_down = False
        self._thread_id = 0
        self._proc = _HOOKPROC(self._hook_proc)
        self._install_error: MouseHookError | None = None
        self._started = threading.Event()

    @override
    def run(self) -> None:
        self._thread_id = kernel32.GetCurrentThreadId()
        hook = user32.SetWindowsHookExW(_WH_MOUSE_LL, self._proc, None, 0)
        if not hook:
            self._install_error = MouseHookError(ctypes.get_last_error())
            self._started.set()
            return
        self._started.set()
        log.info("mouse hook installed: vk=0x%02X thread=%d", self._vk, self._thread_id)
        try:
            msg = _MSG()
            while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
                pass
        finally:
            user32.UnhookWindowsHookEx(hook)

    def start_and_wait(self) -> None:
        """Start the thread and raise if hook installation failed."""
        self.start()
        self._started.wait(timeout=5)
        if self._install_error is not None:
            raise self._install_error

    def stop(self) -> None:
        if self._thread_id:
            user32.PostThreadMessageW(self._thread_id, _WM_QUIT, 0, 0)
        self.join(timeout=2)

    def retarget(self, vk: int) -> None:
        """Watch a different button (0=disabled); resets dangling down-state."""
        self._vk = vk
        self._is_down = False

    def _hook_proc(self, ncode: int, wparam: int, lparam: int) -> int:
        if ncode >= 0 and self._vk:
            mb = ctypes.cast(lparam, _LLHOOKPTR).contents
            injected = mb.flags & (_LLMHF_INJECTED | _LLMHF_LOWER_IL_INJECTED)
            if not injected and wparam in (_WM_XBUTTONDOWN, _WM_XBUTTONUP):
                button = mb.mouseData >> 16  # high word: 1=X1, 2=X2
                if button == (1 if self._vk == VK_XBUTTON1 else 2):
                    self._emit_transition(pressed=wparam == _WM_XBUTTONDOWN)
                    return 1  # suppress native back/forward
        return user32.CallNextHookEx(None, ncode, wparam, lparam)

    def _emit_transition(self, pressed: bool) -> None:
        if pressed:
            if self._is_down:
                return
            self._is_down = True
        else:
            if not self._is_down:
                return
            self._is_down = False
        self._on_transition(pressed)

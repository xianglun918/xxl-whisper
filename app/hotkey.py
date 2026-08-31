"""Global hotkey via a WH_KEYBOARD_LL hook on a dedicated message-pump thread.

The target key (default CapsLock) is suppressed system-wide so it stops doing
its native job while held. Injected events (LLKHF_INJECTED) are always passed
through, which is what lets :mod:`app.winio` re-synthesize a real CapsLock
tap for the click case without us swallowing it.
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

_WH_KEYBOARD_LL: int = 13
_WM_KEYDOWN: int = 0x0100
_WM_KEYUP: int = 0x0101
_WM_SYSKEYDOWN: int = 0x0104
_WM_SYSKEYUP: int = 0x0105
_WM_QUIT: int = 0x0012
_LLKHF_INJECTED: int = 0x10
_LLKHF_LOWER_IL_INJECTED: int = 0x02
_VK_ESCAPE: int = 0x1B
_MODIFIER_VKS: frozenset[int] = frozenset(
    {0x10, 0x11, 0x12, 0x5B, 0x5C, 0xA0, 0xA1, 0xA2, 0xA3, 0xA4, 0xA5}
)  # Shift/Ctrl/Alt/Win (left+right variants)


class HotkeyError(Exception):
    """Raised when the low-level keyboard hook cannot be installed."""

    def __init__(self, code: int) -> None:
        super().__init__(f"SetWindowsHookExW failed with Win32 error {code}")
        self.code = code


class _KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = (
        ("vkCode", wintypes.DWORD),
        ("scanCode", wintypes.DWORD),
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
_LLHOOKPTR = ctypes.POINTER(_KBDLLHOOKSTRUCT)

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


class HotkeyHook(threading.Thread):
    """Daemon thread owning the hook; emits key transitions for one VK code.

    ``on_transition(pressed)`` runs on the hook thread — it must only enqueue.
    Auto-repeat is filtered here, so the callback sees clean down/up pairs.
    """

    def __init__(self, vk: int, on_transition: Callable[[bool], None]) -> None:
        super().__init__(daemon=True, name="hotkey-hook")
        self._vk = vk
        self._on_transition = on_transition
        self._is_down = False
        self._thread_id = 0
        self._proc = _HOOKPROC(self._hook_proc)
        self._install_error: HotkeyError | None = None
        self._started = threading.Event()
        self._capture_cb: Callable[[int], None] | None = None

    @override
    def run(self) -> None:
        self._thread_id = kernel32.GetCurrentThreadId()
        hook = user32.SetWindowsHookExW(_WH_KEYBOARD_LL, self._proc, None, 0)
        if not hook:
            self._install_error = HotkeyError(ctypes.get_last_error())
            self._started.set()
            return
        self._started.set()
        log.info("hook installed: vk=0x%02X thread=%d", self._vk, self._thread_id)
        try:
            msg = _MSG()
            while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
                pass  # pure pump; we never translate or dispatch
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
        """Watch a different key from now on, without reinstalling the hook.

        Resets the down-state so a key held across the switch cannot leave a
        dangling press; the old key instantly regains its native behavior.
        """
        self._vk = vk
        self._is_down = False

    def arm_capture(self, on_key: Callable[[int], None]) -> None:
        """One-shot: deliver the next physical key press to ``on_key``.

        The captured key (and Esc as the cancel signal) is suppressed so it
        never reaches applications. Pure modifiers are ignored — they cannot
        serve as a hold-to-talk key.
        """
        self._capture_cb = on_key

    def _hook_proc(self, ncode: int, wparam: int, lparam: int) -> int:
        if ncode >= 0:  # HC_ACTION
            kb = ctypes.cast(lparam, _LLHOOKPTR).contents
            injected = kb.flags & (_LLKHF_INJECTED | _LLKHF_LOWER_IL_INJECTED)
            if not injected and self._capture_filter(kb.vkCode, wparam):
                return 1  # swallow the captured key
            if kb.vkCode == self._vk and self._vk and not injected:
                pressed = wparam in (_WM_KEYDOWN, _WM_SYSKEYDOWN)
                self._emit_transition(pressed)
                return 1  # suppress the native key function
        return user32.CallNextHookEx(None, ncode, wparam, lparam)

    def _capture_filter(self, vk: int, wparam: int) -> bool:
        """Consume an armed one-shot capture; True when the event is swallowed.

        Pure modifiers never satisfy a capture (they cannot serve as a
        hold-to-talk key) and are delivered normally.
        """
        if self._capture_cb is None or wparam not in (_WM_KEYDOWN, _WM_SYSKEYDOWN):
            return False
        if vk in _MODIFIER_VKS:
            return False
        callback = self._capture_cb
        self._capture_cb = None
        callback(vk)
        return True

    def _emit_transition(self, pressed: bool) -> None:
        if pressed:
            if self._is_down:
                return  # auto-repeat
            self._is_down = True
        else:
            if not self._is_down:
                return  # stray release
            self._is_down = False
        self._on_transition(pressed)

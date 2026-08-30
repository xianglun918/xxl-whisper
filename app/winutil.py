"""Small Win32 utilities: autostart registry, single-instance mutex, DPI, dialogs."""

import ctypes
import sys
import winreg
from ctypes import wintypes

_RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
_APP_NAME = "xxl-whisper"
_MUTEX_NAME = "Local\\xxl-whisper-single-instance"
_ERROR_ALREADY_EXISTS = 183

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
user32 = ctypes.WinDLL("user32", use_last_error=True)

kernel32.CreateMutexW.restype = wintypes.HANDLE
kernel32.CreateMutexW.argtypes = (ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR)
user32.MessageBoxW.argtypes = (wintypes.HWND, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.UINT)

_mutex_handle: wintypes.HANDLE | None = None  # module-global: keeps the mutex alive


def autostart_enabled() -> bool:
    """Whether the HKCU Run entry for this app exists."""
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _RUN_KEY) as key:
            winreg.QueryValueEx(key, _APP_NAME)
            return True
    except FileNotFoundError:
        return False


def set_autostart(enabled: bool) -> None:
    """Create or remove the HKCU Run entry (no admin rights needed)."""
    if enabled:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _RUN_KEY, 0, winreg.KEY_SET_VALUE) as key:
            winreg.SetValueEx(key, _APP_NAME, 0, winreg.REG_SZ, f'"{sys.executable}"')
    else:
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _RUN_KEY, 0, winreg.KEY_SET_VALUE) as key:
                winreg.DeleteValue(key, _APP_NAME)
        except FileNotFoundError:
            pass


def acquire_single_instance() -> bool:
    """Return False when another instance already holds the mutex."""
    global _mutex_handle  # noqa: PLW0603 — handle must outlive this call
    _mutex_handle = kernel32.CreateMutexW(None, False, _MUTEX_NAME)
    return ctypes.get_last_error() != _ERROR_ALREADY_EXISTS


def show_error(message: str) -> None:
    """Fatal-error dialog for the process boundary."""
    user32.MessageBoxW(None, message, "xxl-whisper", 0x00000010)  # MB_ICONERROR


def set_dpi_awareness() -> None:
    """Best-effort per-monitor DPI awareness so the Tk bar isn't blurry."""
    try:
        shcore = ctypes.WinDLL("shcore")
        shcore.SetProcessDpiAwareness(2)  # PER_MONITOR_AWARE
    except (OSError, AttributeError):
        user32.SetProcessDPIAware()

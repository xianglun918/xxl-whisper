"""Small Win32 utilities: autostart registry, single-instance mutex, DPI, dialogs."""

import ctypes
import sys
import winreg
from ctypes import wintypes

_RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
_APP_NAME = "xxl-whisper"
_MUTEX_NAME = "Local\\xxl-whisper-single-instance"
_ERROR_ALREADY_EXISTS = 183
_IDYES = 6
_MB_SETFOREGROUND = 0x00010000
_MB_TOPMOST = 0x00040000
_SYNCHRONIZE = 0x00100000

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
user32 = ctypes.WinDLL("user32", use_last_error=True)

kernel32.CreateMutexW.restype = wintypes.HANDLE
kernel32.CreateMutexW.argtypes = (ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR)
kernel32.OpenMutexW.restype = wintypes.HANDLE
kernel32.OpenMutexW.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR)
kernel32.CloseHandle.restype = wintypes.BOOL
kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
user32.MessageBoxW.argtypes = (
    wintypes.HWND,
    wintypes.LPCWSTR,
    wintypes.LPCWSTR,
    wintypes.UINT,
)
user32.MessageBoxW.restype = ctypes.c_int

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


def instance_running() -> bool:
    """Whether another process holds the single-instance mutex.

    Probes with ``OpenMutexW`` (never creating one) so the update helper can
    watch for the running app to exit without owning the mutex itself.
    """
    handle = kernel32.OpenMutexW(_SYNCHRONIZE, False, _MUTEX_NAME)
    if not handle:
        return False
    kernel32.CloseHandle(handle)
    return True


def show_error(message: str) -> None:
    """Fatal-error dialog for the process boundary."""
    user32.MessageBoxW(None, message, "xxl-whisper", 0x00000010)  # MB_ICONERROR


def ask_yes_no(message: str, title: str = "xxl-whisper") -> bool:
    """Non-fatal question dialog; True when the user picks Yes."""
    flags = 0x00000024 | _MB_SETFOREGROUND | _MB_TOPMOST  # ICONQUESTION|YESNO
    result = user32.MessageBoxW(None, message, title, flags)
    return result == _IDYES


def show_info(message: str, title: str = "xxl-whisper") -> None:
    """Non-fatal informational dialog."""
    flags = 0x00000040 | _MB_SETFOREGROUND | _MB_TOPMOST  # MB_ICONINFORMATION
    user32.MessageBoxW(None, message, title, flags)


def save_file_dialog() -> str | None:
    """Open a save dialog and return the chosen path, or None on cancel."""
    import tkinter as tk  # noqa: PLC0415 — lazy so dialogs are only paid for on use
    from tkinter import filedialog  # noqa: PLC0415

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        path = filedialog.asksaveasfilename(
            parent=root,
            title="选择输出文件",
            defaultextension=".txt",
            initialfile="xxl-whisper-转写.txt",
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")],
        )
    finally:
        root.destroy()
    return path or None


def open_file_dialog() -> str | None:
    """Open a file dialog and return the chosen file, or None on cancel."""
    import tkinter as tk  # noqa: PLC0415 — lazy so dialogs are only paid for on use
    from tkinter import filedialog  # noqa: PLC0415

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        path = filedialog.askopenfilename(
            parent=root,
            title="选择要转写的音频/视频文件",
            filetypes=[
                ("音频/视频", "*.mp3 *.wav *.m4a *.aac *.flac *.ogg *.mp4 *.mov *.mkv *.webm"),
                ("所有文件", "*.*"),
            ],
        )
    finally:
        root.destroy()
    return path or None


def set_dpi_awareness() -> None:
    """Best-effort per-monitor DPI awareness so the Tk bar isn't blurry."""
    try:
        shcore = ctypes.WinDLL("shcore")
        shcore.SetProcessDpiAwareness(2)  # PER_MONITOR_AWARE
    except (OSError, AttributeError):
        user32.SetProcessDPIAware()


def permission_report() -> list[str]:
    """No TCC permission model on Windows; contributes no diagnostics lines."""
    return []


def prompt_permissions() -> None:
    """No-op on Windows: there are no per-user TCC grants to request."""

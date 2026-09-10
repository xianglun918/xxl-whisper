"""macOS process-level utilities.

M3 fills ``acquire_single_instance`` (the app cannot start without it); the
rest (autostart, dialogs, notifications, permission checks) land in M5.
Mirrors ``app/winutil.py`` so shared code reaches it via ``app.native.util``.
``set_dpi_awareness`` is a real no-op: Retina scaling is automatic on macOS.
"""

import fcntl
import importlib
import logging
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

#: Mirrors app.native.data_root()'s darwin branch; importing native here would
#: be circular (native re-exports this module).
_DATA_ROOT: Path = Path.home() / "Library" / "Application Support" / "xxl-whisper"

_LOCK_HANDLES: list[object] = []

_ACCESS_URL = "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
_LISTEN_URL = "x-apple.systempreferences:com.apple.preference.security?Privacy_ListenEvent"


def _as_literal(text: str) -> str:
    """Render ``text`` as an AppleScript string literal (newlines as return)."""
    parts = [part.replace("\\", "\\\\").replace('"', '\\"') for part in text.split("\n")]
    return " & return & ".join(f'"{part}"' for part in parts)


def _osascript(script: str) -> str:
    """Run one AppleScript snippet; return trimmed stdout (empty on failure)."""
    result = subprocess.run(  # noqa: S603 — fixed /usr/bin/osascript, no shell
        ["/usr/bin/osascript", "-e", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    return result.stdout.strip()


def autostart_enabled() -> bool:
    """Whether the app is registered to launch at login.

    Always False in a source run: SMAppService only applies to a bundled .app
    (the registration path lands with packaging, M6).
    """
    return False


def set_autostart(enabled: bool) -> None:
    """Register/unregister the app as a login item (needs the bundled app, M6)."""
    log.warning("autostart toggle (%s) is unavailable in source runs (M6)", enabled)


def acquire_single_instance() -> bool:
    """Take a per-user single-instance lock; False when already running."""
    _DATA_ROOT.mkdir(parents=True, exist_ok=True)
    handle = (_DATA_ROOT / "xxl-whisper.lock").open("w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return False
    _LOCK_HANDLES.append(handle)
    return True


def _dialog_script(message: str, *, title: str, buttons: str = "", icon: str = "") -> str:
    """Build a ``display dialog`` AppleScript for the given parts."""
    parts = [f"display dialog {_as_literal(message)}"]
    if buttons:
        parts.append(f"buttons {buttons}")
    if icon:
        parts.append(f"with icon {icon}")
    parts.append(f"with title {_as_literal(title)}")
    return " ".join(parts)


def show_error(message: str) -> None:
    """Show a modal error dialog."""
    _osascript(_dialog_script(message, title="xxl-whisper", icon="stop"))


def ask_yes_no(message: str, title: str = "xxl-whisper") -> bool:
    """Ask a modal yes/no question; True when the user confirms."""
    script = _dialog_script(message, title=title, buttons='{"取消", "确定"}')
    return "确定" in _osascript(script)


def show_info(message: str, title: str = "xxl-whisper") -> None:
    """Show a modal information dialog."""
    _osascript(_dialog_script(message, title=title, icon="note"))


def _permission_state() -> list[tuple[str, bool, str]]:
    """Return (label, granted, settings URL) for the checkable TCC grants."""
    app_services = importlib.import_module("ApplicationServices")
    quartz = importlib.import_module("Quartz")
    return [
        ("辅助功能（注入 Cmd+V）", bool(app_services.AXIsProcessTrusted()), _ACCESS_URL),
        ("输入监控（监听热键）", bool(quartz.CGPreflightListenEventAccess()), _LISTEN_URL),
    ]


def permission_report() -> list[str]:
    """Human-readable TCC permission state for the diagnostics dialog."""
    lines: list[str] = []
    for label, granted, url in _permission_state():
        lines.append(f"{label}：{'已授权' if granted else '未授权'}")
        if not granted:
            lines.append(f"  → 系统设置：{url}")
    lines.append("麦克风（录音）：首次录音时系统弹窗授权")
    return lines


def prompt_permissions() -> None:
    """Onboarding: register the app in TCC and offer to open System Settings.

    The prompting API variants are required: a freshly bundled app only appears
    in the Accessibility / Input Monitoring lists after it has asked.
    """
    app_services = importlib.import_module("ApplicationServices")
    quartz = importlib.import_module("Quartz")
    missing: list[tuple[str, str]] = []
    if not app_services.AXIsProcessTrusted():
        app_services.AXIsProcessTrustedWithOptions(
            {app_services.kAXTrustedCheckOptionPrompt: True}
        )
        missing.append(("辅助功能（注入 Cmd+V）", _ACCESS_URL))
    if not quartz.CGPreflightListenEventAccess():
        quartz.CGRequestListenEventAccess()
        missing.append(("输入监控（监听热键）", _LISTEN_URL))
    if not missing:
        return
    label, url = missing[0]
    message = f"缺少「{label}」授权，语音功能将不可用。"
    script = _dialog_script(message, title="xxl-whisper 需要授权", buttons='{"稍后", "打开设置"}')
    if "打开设置" in _osascript(script):
        subprocess.run(["/usr/bin/open", url], check=False)  # noqa: S603 — fixed /usr/bin/open


def set_dpi_awareness() -> None:
    """No-op on macOS: Retina scaling is handled by the system."""

"""macOS process-level utilities.

M3 fills ``acquire_single_instance`` (the app cannot start without it); the
rest (autostart, dialogs, notifications, permission checks) land in M5.
Mirrors ``app/winutil.py`` so shared code reaches it via ``app.native.util``.
``set_dpi_awareness`` is a real no-op: Retina scaling is automatic on macOS.
"""

import fcntl
import logging
from pathlib import Path

log = logging.getLogger(__name__)

#: Mirrors app.native.data_root()'s darwin branch; importing native here would
#: be circular (native re-exports this module).
_DATA_ROOT: Path = Path.home() / "Library" / "Application Support" / "xxl-whisper"

_LOCK_HANDLES: list[object] = []


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


def show_error(message: str) -> None:
    """Show a modal error dialog."""
    msg = f"macutil.show_error({len(message)} chars) lands in M5 (NSAlert)"
    raise NotImplementedError(msg)


def ask_yes_no(message: str, title: str = "xxl-whisper") -> bool:
    """Ask a modal yes/no question."""
    msg = f"macutil.ask_yes_no({len(message)} chars, {title!r}) lands in M5 (NSAlert)"
    raise NotImplementedError(msg)


def show_info(message: str, title: str = "xxl-whisper") -> None:
    """Show a modal information dialog."""
    msg = f"macutil.show_info({len(message)} chars, {title!r}) lands in M5 (NSAlert)"
    raise NotImplementedError(msg)


def set_dpi_awareness() -> None:
    """No-op on macOS: Retina scaling is handled by the system."""

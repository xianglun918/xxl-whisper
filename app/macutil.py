"""macOS process-level utilities — skeleton, bodies land in M5.

Mirrors the public surface of ``app/winutil.py`` so shared code reaches it via
``app.native.util``. ``set_dpi_awareness`` is a real no-op: Retina scaling is
automatic on macOS. pyobjc is imported lazily with ``importlib`` inside the
bodies (see the M1 findings in ``docs/macos-port-plan.md``).
"""


def autostart_enabled() -> bool:
    """Whether the app is registered to launch at login."""
    msg = "macutil.autostart_enabled lands in M5 (SMAppService)"
    raise NotImplementedError(msg)


def set_autostart(enabled: bool) -> None:
    """Register/unregister the app as a login item."""
    msg = f"macutil.set_autostart(enabled={enabled}) lands in M5 (SMAppService)"
    raise NotImplementedError(msg)


def acquire_single_instance() -> bool:
    """Take a per-user single-instance lock; False when already running."""
    msg = "macutil.acquire_single_instance lands in M5 (lock file)"
    raise NotImplementedError(msg)


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

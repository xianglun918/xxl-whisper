"""macOS input injection / clipboard facade — skeleton, bodies land in M4.

Mirrors the public surface of ``app/winio.py`` so shared code reaches it via
``app.native.io``. Every body is a placeholder until M4; pyobjc is imported
lazily with ``importlib`` inside the bodies (see the M1 findings in
``docs/macos-port-plan.md``).

Win-only concepts (``post_wm_paste_to_focus``, ``focused_control_class``) have
no macOS equivalent; M4 reshapes the mac emit ladder and may drop them here.
"""


class PasteError(Exception):
    """Raised when the macOS paste channel fails."""

    def __init__(self, api: str, code: int) -> None:
        super().__init__(f"{api} failed ({code})")
        self.api = api
        self.code = code


def tap_key(vk: int) -> None:
    """Re-synthesize a native key tap (mac keycode)."""
    msg = f"macio.tap_key(vk={vk}) lands in M4 (mac native surface)"
    raise NotImplementedError(msg)


def paste_text(text: str, restore_clipboard: bool, delay_ms: int) -> None:
    """Clipboard swap + Cmd+V injection into the frontmost app."""
    msg = (
        f"macio.paste_text({len(text)} chars, restore={restore_clipboard}, "
        f"delay={delay_ms}) lands in M4 (mac native surface)"
    )
    raise NotImplementedError(msg)


def type_text(text: str) -> None:
    """Unicode typing fallback."""
    msg = f"macio.type_text({len(text)} chars) lands in M4 (mac native surface)"
    raise NotImplementedError(msg)


def foreground_window_title() -> str:
    """Title of the frontmost window."""
    msg = "macio.foreground_window_title lands in M4 (mac native surface)"
    raise NotImplementedError(msg)


def set_clipboard_text(text: str) -> None:
    """Replace the general pasteboard contents."""
    msg = f"macio.set_clipboard_text({len(text)} chars) lands in M4 (mac native surface)"
    raise NotImplementedError(msg)


def keyboard_injection_alive() -> bool:
    """Whether synthetic keystrokes can be posted (Accessibility granted)."""
    msg = "macio.keyboard_injection_alive lands in M4 (mac native surface)"
    raise NotImplementedError(msg)


def post_wm_paste_to_focus() -> bool:
    """Windows-only WM_PASTE channel; no macOS equivalent."""
    msg = "macio.post_wm_paste_to_focus lands in M4 (mac native surface)"
    raise NotImplementedError(msg)


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
    msg = f"macio.key_name(vk={vk}) lands in M4 (mac native surface)"
    raise NotImplementedError(msg)


def tap_mouse_x(vk: int) -> None:
    """Re-synthesize a mouse side-button tap."""
    msg = f"macio.tap_mouse_x(vk={vk}) lands in M4 (mac native surface)"
    raise NotImplementedError(msg)


def focused_control_class() -> str:
    """Windows-only control class name; no macOS equivalent."""
    msg = "macio.focused_control_class lands in M4 (mac native surface)"
    raise NotImplementedError(msg)

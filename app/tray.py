"""System-tray icon and menu (pystray).

Rendering only: the *layout* lives in :mod:`app.menu_spec` as pure data, so it
can be unit-tested without a display. This module turns each
:class:`~app.menu_spec.Entry` into a pystray item, gathers the machine-dependent
lists (microphones, system-audio endpoints, hotkey presets) and owns the icon.

The menu is rebuilt on every open (pystray re-evaluates callables), which is what
lets the parent rows carry live state such as
``翻译与字幕：日→中 · 上屏+打字``.
"""

from collections.abc import Callable

import pystray
from PIL import Image, ImageDraw

import app.loopback as loopback_mod
import app.recorder as recorder_mod
from app.config import HOTKEY_VK
from app.menu_spec import (
    Entry,
    MenuInputs,
    TrayCallbacks,
    TrayState,
    build_menu,
)


def list_hotkey_names() -> tuple[str, ...]:
    """Preset hotkey names, in config order."""
    return tuple(HOTKEY_VK)


def list_microphone_names() -> tuple[str, ...]:
    """Capture devices PortAudio can see, in index order."""
    return tuple(device.name for device in recorder_mod.list_input_devices())


def list_system_audio_devices() -> tuple[tuple[str, str], ...]:
    """System-audio endpoints as ``(label, device_id)``; empty off Windows."""
    return tuple(
        (device.name, device.device_id) for device in loopback_mod.list_loopback_devices()
    )


class Tray:
    """Owns the pystray icon; ``run()`` blocks the calling thread."""

    def __init__(
        self, callbacks: TrayCallbacks, state_provider: Callable[[], TrayState]
    ) -> None:
        self._callbacks = callbacks
        self._state_provider = state_provider
        self._icon = pystray.Icon(
            name="xxl-whisper",
            icon=_draw_icon(),
            title="xxl-whisper 语音听写",
            menu=self._build_menu(),
        )

    def run(self) -> None:
        """Run the tray event loop (blocks)."""
        self._icon.run()

    def stop(self) -> None:
        """Stop the tray event loop."""
        self._icon.stop()

    def notify(self, message: str, title: str = "xxl-whisper") -> None:
        """Show a native notification balloon."""
        self._icon.notify(message, title)

    def refresh_menu(self) -> None:
        """Force the native menu to rebuild (a switch changed)."""
        self._icon.update_menu()

    # -- rendering ---------------------------------------------------------

    def _build_menu(self) -> pystray.Menu:
        """Render the current spec; pystray calls this again on every open."""
        inputs = MenuInputs(
            microphones=list_microphone_names(),
            system_audio=list_system_audio_devices(),
            hotkey_names=list_hotkey_names(),
        )
        spec = build_menu(self._state_provider(), self._callbacks, inputs)
        return pystray.Menu(*(self._render(entry) for entry in spec))

    def _render(self, entry: Entry) -> pystray.MenuItem:
        """Turn one spec row into a pystray item (recursively for submenus)."""
        if entry.separator:
            return pystray.Menu.SEPARATOR
        label: str | Callable[[object], str]
        if entry.text is not None:
            getter = entry.text
            label = lambda _item: getter()  # noqa: E731 - pystray wants a callable
        else:
            label = entry.label
        if entry.children:
            return pystray.MenuItem(label, pystray.Menu(*(self._render(c) for c in entry.children)))
        checked = None if entry.checked is None else (lambda _item, value=entry.checked: value)
        return pystray.MenuItem(
            label,
            entry.action,
            checked=checked,
            enabled=entry.enabled,
            radio=entry.radio,
            default=False,
        )


def _draw_icon() -> Image.Image:
    image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((4, 4, 60, 60), radius=14, fill="#0f172a")
    draw.rounded_rectangle((24, 12, 40, 38), radius=8, fill="#7dd3fc")  # mic head
    draw.arc((16, 24, 48, 46), start=15, end=165, fill="#7dd3fc", width=3)  # cradle
    draw.line((32, 46, 32, 52), fill="#7dd3fc", width=3)  # stem
    draw.line((24, 52, 40, 52), fill="#7dd3fc", width=3)  # base
    return image

"""System-tray icon and menu (pystray). Menu content is rebuilt on open."""

from collections.abc import Callable
from dataclasses import dataclass

import pystray
from PIL import Image, ImageDraw

import app.recorder as recorder_mod
from app.config import HOTKEY_VK

_HOTKEY_LABELS = {
    "caps_lock": "CapsLock",
    "f2": "F2",
    "f4": "F4",
    "f6": "F6",
    "f8": "F8",
    "scroll_lock": "Scroll Lock",
}


@dataclass(frozen=True, slots=True)
class TrayCallbacks:
    on_exit: Callable[[], None]
    on_toggle_pause: Callable[[], None]
    on_select_mic: Callable[[str], None]
    on_toggle_autostart: Callable[[], None]
    on_check_update: Callable[[], None]
    on_select_hotkey: Callable[[str], None]


@dataclass(frozen=True, slots=True)
class TrayState:
    ready: bool
    paused: bool
    autostart: bool
    current_mic: str
    current_hotkey: str


class Tray:
    """Owns the pystray icon; ``run()`` blocks the calling thread."""

    def __init__(self, callbacks: TrayCallbacks, state_provider: Callable[[], TrayState]) -> None:
        self._callbacks = callbacks
        self._state_provider = state_provider
        self._icon = pystray.Icon(
            name="xxl-whisper",
            icon=_draw_icon(),
            title="xxl-whisper 语音听写",
            menu=self._build_menu(),
        )

    def run(self) -> None:
        self._icon.run()

    def stop(self) -> None:
        self._icon.stop()

    def notify(self, message: str, title: str = "xxl-whisper") -> None:
        self._icon.notify(message, title)

    def _build_menu(self) -> pystray.Menu:
        return pystray.Menu(
            pystray.MenuItem(
                self._status_text,
                None,
                enabled=False,
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                "暂停语音热键",
                self._callbacks.on_toggle_pause,
                checked=lambda _item: self._state_provider().paused,
            ),
            pystray.MenuItem(
                "麦克风",
                pystray.Menu(self._mic_items),
            ),
            pystray.MenuItem(
                "热键",
                pystray.Menu(self._hotkey_items),
            ),
            pystray.MenuItem(
                "开机自启",
                self._callbacks.on_toggle_autostart,
                checked=lambda _item: self._state_provider().autostart,
            ),
            pystray.MenuItem("检查更新", self._callbacks.on_check_update),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("退出", self._callbacks.on_exit),
        )

    def _mic_items(self) -> list[pystray.MenuItem]:
        """Return fresh items; pystray re-evaluates this on each menu open."""
        state = self._state_provider()
        items = [
            pystray.MenuItem(
                "（系统默认）" + (" ✓" if state.current_mic == "" else ""),
                lambda: self._callbacks.on_select_mic(""),
            )
        ]
        for device in recorder_mod.list_input_devices():
            mark = " ✓" if device.name == state.current_mic else ""
            items.append(
                pystray.MenuItem(
                    f"{device.name}{mark}",
                    lambda name=device.name: self._callbacks.on_select_mic(name),
                )
            )
        return items

    def _hotkey_items(self) -> list[pystray.MenuItem]:
        """Hotkey choices; pystray re-evaluates this on each menu open."""
        current = self._state_provider().current_hotkey
        return [
            pystray.MenuItem(
                _HOTKEY_LABELS.get(name, name) + (" ✓" if name == current else ""),
                lambda name=name: self._callbacks.on_select_hotkey(name),
            )
            for name in HOTKEY_VK
        ]

    def _status_text(self, _item: object) -> str:
        state = self._state_provider()
        if not state.ready:
            return "状态：启动中…"
        label = _HOTKEY_LABELS.get(state.current_hotkey, state.current_hotkey)
        if state.paused:
            return f"状态：已暂停（{label} 只保留原功能）"
        return f"状态：就绪 — 按住 {label} 说话，松开出字"


def _draw_icon() -> Image.Image:
    image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((4, 4, 60, 60), radius=14, fill="#0f172a")
    draw.rounded_rectangle((24, 12, 40, 38), radius=8, fill="#7dd3fc")  # mic head
    draw.arc((16, 24, 48, 46), start=15, end=165, fill="#7dd3fc", width=3)  # cradle
    draw.line((32, 46, 32, 52), fill="#7dd3fc", width=3)  # stem
    draw.line((24, 52, 40, 52), fill="#7dd3fc", width=3)  # base
    return image

"""System-tray icon and menu (pystray). Menu content is rebuilt on open."""

from collections.abc import Callable
from dataclasses import dataclass

import pystray
from PIL import Image, ImageDraw

import app.loopback as loopback_mod
import app.recorder as recorder_mod
from app import native
from app.config import HOTKEY_VK
from app.translator import TRANSLATE_ROUTES

_HOTKEY_LABELS = {
    "caps_lock": "CapsLock",
    "f2": "F2",
    "f4": "F4",
    "f6": "F6",
    "f8": "F8",
    "scroll_lock": "Scroll Lock",
    "mouse_x1": "鼠标侧键 X1",
    "mouse_x2": "鼠标侧键 X2",
}

_MODEL_LABELS = {
    "sensevoice": "SenseVoice-Small（默认 · 230MB · 快）",
    "funasr_nano": "FunASR-Nano（更准 · 约 1GB · 首次需下载）",
}

#: Compact route label for the status line (the full label lives in the menu).
_ROUTE_ARROWS: dict[str, str] = {
    "zh-en": "中→英",
    "en-zh": "英→中",
    "ja-zh": "日→中",
}


def _hotkey_label(hotkey: str | int) -> str:
    if isinstance(hotkey, int):
        return native.io.key_name(hotkey)
    return _HOTKEY_LABELS.get(hotkey, hotkey)


@dataclass(frozen=True, slots=True)
class TrayCallbacks:
    on_exit: Callable[[], None]
    on_toggle_pause: Callable[[], None]
    on_select_mic: Callable[[str], None]
    on_toggle_autostart: Callable[[], None]
    on_check_update: Callable[[], None]
    on_select_hotkey: Callable[[str | int], None]
    on_capture_hotkey: Callable[[], None]
    on_select_model: Callable[[str], None]
    on_show_diagnostics: Callable[[], None]
    on_toggle_disfluency: Callable[[], None]
    on_toggle_continuous: Callable[[], None]
    on_toggle_translate: Callable[[], None]
    on_select_translate_route: Callable[[str], None]
    on_select_system_audio: Callable[[str], None]
    on_toggle_overlay: Callable[[], None]
    on_select_overlay_output: Callable[[str], None]


@dataclass(frozen=True, slots=True)
class TrayState:
    ready: bool
    paused: bool
    autostart: bool
    current_mic: str
    current_hotkey: str | int
    current_model: str
    disfluency: str
    continuous: bool
    translate: bool
    translate_route: str
    audio_source: str
    current_system_audio: str
    overlay: bool
    overlay_output: str


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

    def refresh_menu(self) -> None:
        """Force the native menu to rebuild (dynamic states changed)."""
        self._icon.update_menu()

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
                "模型",
                pystray.Menu(self._model_items),
            ),
            pystray.MenuItem(
                "语义顺滑",
                self._callbacks.on_toggle_disfluency,
                checked=lambda _item: self._state_provider().disfluency == "smooth",
            ),
            pystray.MenuItem(
                "持续听写",
                self._callbacks.on_toggle_continuous,
                checked=lambda _item: self._state_provider().continuous,
            ),
            pystray.MenuItem(
                "离线翻译",
                pystray.Menu(self._translate_items),
            ),
            pystray.MenuItem(
                "上屏字幕",
                pystray.Menu(self._overlay_items),
            ),
            pystray.MenuItem(
                "开机自启",
                self._callbacks.on_toggle_autostart,
                checked=lambda _item: self._state_provider().autostart,
            ),
            pystray.MenuItem("输入诊断", self._callbacks.on_show_diagnostics),
            pystray.MenuItem("检查更新", self._callbacks.on_check_update),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("退出", self._callbacks.on_exit),
        )

    def _mic_items(self) -> list[pystray.MenuItem]:
        """Return fresh items; pystray re-evaluates this on each menu open."""
        items = [
            pystray.MenuItem(
                "（系统默认）",
                self._select_mic(""),
                checked=lambda _item: self._state_provider().current_mic == "",
            )
        ]
        items.extend(
            pystray.MenuItem(
                device.name,
                self._select_mic(device.name),
                checked=self._mic_checked(device.name),
            )
            for device in recorder_mod.list_input_devices()
        )
        # System audio (WASAPI loopback) shows up as extra entries: picking one
        # switches the source, picking a real mic switches it back.
        loopbacks = loopback_mod.list_loopback_devices()
        if loopbacks:
            items.append(pystray.Menu.SEPARATOR)
            items.extend(
                pystray.MenuItem(
                    device.name,
                    self._select_system_audio(device.device_id),
                    checked=self._system_audio_checked(device.device_id),
                )
                for device in loopbacks
            )
        return items

    def _select_system_audio(self, device_id: str) -> Callable[[], None]:
        """Zero-arg action factory (same pystray arity trap as _select_mic)."""

        def action() -> None:
            self._callbacks.on_select_system_audio(device_id)

        return action

    def _system_audio_checked(self, device_id: str) -> Callable[[object], bool]:
        def checked(_item: object) -> bool:
            state = self._state_provider()
            return state.audio_source == "system" and state.current_system_audio == device_id

        return checked

    def _mic_checked(self, name: str) -> Callable[[object], bool]:
        def checked(_item: object) -> bool:
            return self._state_provider().current_mic == name

        return checked

    def _select_mic(self, name: str) -> Callable[[], None]:
        """Zero-arg action factory: pystray hands 1-arg actions an Icon, not a name."""

        def action() -> None:
            self._callbacks.on_select_mic(name)

        return action

    def _select_hotkey(self, key: str | int) -> Callable[[], None]:
        """Zero-arg action factory (same pystray arity trap as _select_mic)."""

        def action() -> None:
            self._callbacks.on_select_hotkey(key)

        return action

    def _hotkey_checked(self, key: str | int) -> Callable[[object], bool]:
        def checked(_item: object) -> bool:
            return self._state_provider().current_hotkey == key

        return checked

    def _select_model(self, kind: str) -> Callable[[], None]:
        """Zero-arg action factory (same pystray arity trap as _select_mic)."""

        def action() -> None:
            self._callbacks.on_select_model(kind)

        return action

    def _model_checked(self, kind: str) -> Callable[[object], bool]:
        def checked(_item: object) -> bool:
            return self._state_provider().current_model == kind

        return checked

    def _model_items(self) -> list[pystray.MenuItem]:
        """Model choices; pystray re-evaluates this on each menu open."""
        return [
            pystray.MenuItem(
                label,
                self._select_model(kind),
                checked=self._model_checked(kind),
                radio=True,
            )
            for kind, label in _MODEL_LABELS.items()
        ]

    def _translate_items(self) -> list[pystray.MenuItem]:
        """Return the translation submenu: on/off plus the direction (which models to use)."""
        state = self._state_provider()
        return [
            pystray.MenuItem(
                "开启离线翻译",
                self._callbacks.on_toggle_translate,
                checked=lambda _item: self._state_provider().translate,
            ),
            pystray.Menu.SEPARATOR,
            *[
                pystray.MenuItem(
                    label,
                    self._select_translate_route(route),
                    checked=self._translate_route_checked(route),
                    radio=True,
                )
                for route, label in TRANSLATE_ROUTES.items()
            ],
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                _translate_model_hint(state),
                None,
                enabled=False,
            ),
        ]

    def _select_translate_route(self, route: str) -> Callable[[], None]:
        """Zero-arg action factory (same pystray arity trap as _select_mic)."""

        def action() -> None:
            self._callbacks.on_select_translate_route(route)

        return action

    def _translate_route_checked(self, route: str) -> Callable[[object], bool]:
        def checked(_item: object) -> bool:
            return self._state_provider().translate_route == route

        return checked

    def _overlay_items(self) -> list[pystray.MenuItem]:
        """Subtitle-bar submenu: on/off plus where the text is delivered."""
        return [
            pystray.MenuItem(
                "开启上屏字幕",
                self._callbacks.on_toggle_overlay,
                checked=lambda _item: self._state_provider().overlay,
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                "只上屏（不打字）",
                self._select_overlay_output("screen"),
                checked=self._overlay_output_checked("screen"),
                radio=True,
            ),
            pystray.MenuItem(
                "上屏 + 打字到光标",
                self._select_overlay_output("both"),
                checked=self._overlay_output_checked("both"),
                radio=True,
            ),
            pystray.MenuItem(
                "只打字（上屏仅预览）",
                self._select_overlay_output("type"),
                checked=self._overlay_output_checked("type"),
                radio=True,
            ),
        ]

    def _select_overlay_output(self, mode: str) -> Callable[[], None]:
        """Zero-arg action factory (same pystray arity trap as _select_mic)."""

        def action() -> None:
            self._callbacks.on_select_overlay_output(mode)

        return action

    def _overlay_output_checked(self, mode: str) -> Callable[[object], bool]:
        def checked(_item: object) -> bool:
            return self._state_provider().overlay_output == mode

        return checked

    def _hotkey_items(self) -> list[pystray.MenuItem]:
        """Hotkey presets plus a custom-key capture entry; re-evaluated per open."""
        return [
            pystray.MenuItem(
                _hotkey_label(name),
                self._select_hotkey(name),
                checked=self._hotkey_checked(name),
                radio=True,
            )
            for name in HOTKEY_VK
        ] + [
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("自定义按键…", self._callbacks.on_capture_hotkey),
        ]

    def _status_text(self, _item: object) -> str:
        state = self._state_provider()
        if not state.ready:
            return "状态：启动中…"
        arrow = _ROUTE_ARROWS.get(state.translate_route, "中→英")
        if state.continuous:
            suffix = f" · 翻译开（{arrow}）" if state.translate else ""
            return f"状态：持续听写中（热键已停用，说话即上屏）{suffix}"
        label = _hotkey_label(state.current_hotkey)
        if state.paused:
            return f"状态：已暂停（{label} 只保留原功能）"
        suffix = f" · 翻译开（{arrow}）" if state.translate else ""
        return f"状态：就绪 — 按住 {label} 说话，松开出字{suffix}"


def _translate_model_hint(state: TrayState) -> str:
    """Explain which local model the current direction uses (disabled label)."""
    if state.translate_route == "ja-zh":
        return "模型：opus-mt-ja-en → opus-mt-en-zh（首次约 240MB + 245MB）"
    if state.translate_route == "en-zh":
        return "模型：opus-mt-en-zh"
    return "模型：opus-mt-zh-en"


def _draw_icon() -> Image.Image:
    image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((4, 4, 60, 60), radius=14, fill="#0f172a")
    draw.rounded_rectangle((24, 12, 40, 38), radius=8, fill="#7dd3fc")  # mic head
    draw.arc((16, 24, 48, 46), start=15, end=165, fill="#7dd3fc", width=3)  # cradle
    draw.line((32, 46, 32, 52), fill="#7dd3fc", width=3)  # stem
    draw.line((24, 52, 40, 52), fill="#7dd3fc", width=3)  # base
    return image

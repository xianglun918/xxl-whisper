"""Tray menu structure as pure data (no pystray, no platform imports).

Kept separate from :mod:`app.tray` for two reasons:

* the layout is a **design decision** worth testing — at most six top-level
  entries, related things grouped, and labels that carry the current state so
  the user does not have to open a submenu to find out what is on;
* :mod:`app.tray` pulls in pystray/PIL (display-bound), so nothing defined there
  can be unit-tested on a headless machine.

The menu is grouped by intent:

    状态行
    ─────────────
    暂停语音热键
    输入源      ▸ 麦克风 / 系统声音
    翻译与字幕  ▸ 翻译开关、方向、字幕开关、输出方式
    更多设置    ▸ 热键 / 模型 / 顺滑 / 持续听写 / 自启 / 诊断 / 更新
    ─────────────
    退出

Device and hotkey lists are injected (:class:`MenuInputs`) rather than imported,
so this module depends on nothing but the standard library.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

#: Menu labels for the translation routes, in menu order.
ROUTE_LABELS: dict[str, str] = {
    "zh-en": "中 → 英（说中文出英文）",
    "en-zh": "英 → 中（说英文出中文）",
    "ja-zh": "日 → 中（说日语出中文）",
}

#: Menu labels for the subtitle/typing split.
OUTPUT_LABELS: dict[str, str] = {
    "screen": "只上屏（不打字）",
    "both": "上屏 + 打字到光标",
    "type": "只打字（无字幕）",
}

#: Short route names echoed in the parent label.
_ROUTE_ARROWS: dict[str, str] = {"zh-en": "中→英", "en-zh": "英→中", "ja-zh": "日→中"}

#: Short output names echoed in the parent label.
_OUTPUT_ARROWS: dict[str, str] = {"screen": "只上屏", "both": "上屏+打字", "type": "只打字"}

#: Preset hotkey names -> labels.
HOTKEY_LABELS: dict[str, str] = {
    "caps_lock": "CapsLock",
    "f2": "F2",
    "f4": "F4",
    "f6": "F6",
    "f8": "F8",
    "scroll_lock": "Scroll Lock",
    "mouse_x1": "鼠标侧键 X1",
    "mouse_x2": "鼠标侧键 X2",
}

MODEL_LABELS: dict[str, str] = {
    "sensevoice": "SenseVoice-Small（默认 · 230MB · 快）",
    "funasr_nano": "FunASR-Nano（更准 · 约 1GB · 首次需下载）",
}


@dataclass(frozen=True, slots=True)
class TrayCallbacks:
    """Every action the menu can trigger."""

    on_exit: Callable[[], None]
    on_toggle_pause: Callable[[], None]
    on_select_mic: Callable[[str], None]
    on_select_system_audio: Callable[[str], None]
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
    on_toggle_overlay: Callable[[], None]
    on_select_overlay_output: Callable[[str], None]


@dataclass(frozen=True, slots=True)
class TrayState:
    """Everything the menu needs to render itself."""

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
    translator_ready: bool
    overlay: bool
    overlay_output: str
    audio_source: str
    current_system_audio: str


@dataclass(frozen=True, slots=True)
class MenuInputs:
    """Machine-dependent lists, gathered by the tray and injected here."""

    microphones: tuple[str, ...] = ()
    system_audio: tuple[tuple[str, str], ...] = ()  # (label, device id)
    hotkey_names: tuple[str, ...] = tuple(HOTKEY_LABELS)


@dataclass(frozen=True, slots=True)
class Entry:
    """One menu row: an action, a checkbox, a disabled label, or a submenu."""

    label: str = ""
    action: Callable[[], None] | None = None
    checked: bool | None = None
    enabled: bool = True
    radio: bool = False
    children: tuple[Entry, ...] = ()
    separator: bool = False
    #: Rendered instead of ``label`` when the menu is opened (live state).
    text: Callable[[], str] | None = None


def separator() -> Entry:
    """Return a menu separator row."""
    return Entry(separator=True)


def _pick(callback: Callable[[str], None], value: str) -> Callable[[], None]:
    """Bind one value into a zero-argument action (pystray's arity quirk)."""

    def action() -> None:
        callback(value)

    return action


def _pick_key(callback: Callable[[str | int], None], value: str | int) -> Callable[[], None]:
    def action() -> None:
        callback(value)

    return action


def build_menu(
    state: TrayState, callbacks: TrayCallbacks, inputs: MenuInputs | None = None
) -> tuple[Entry, ...]:
    """Return the whole tray menu for *state* (rebuilt on every open)."""
    lists = inputs or MenuInputs()
    return (
        Entry(text=lambda: state_text(state), enabled=False),
        separator(),
        Entry(label="暂停语音热键", action=callbacks.on_toggle_pause, checked=state.paused),
        Entry(label=input_label(state), children=_input_entries(state, callbacks, lists)),
        Entry(
            label=translate_label(state),
            children=_translate_entries(state, callbacks),
        ),
        Entry(label="更多设置", children=_settings_entries(state, callbacks, lists)),
        separator(),
        Entry(label="退出", action=callbacks.on_exit),
    )


# -- submenus ------------------------------------------------------------------


def _input_entries(
    state: TrayState, callbacks: TrayCallbacks, inputs: MenuInputs
) -> tuple[Entry, ...]:
    """Microphones plus system-audio endpoints (Windows loopback)."""
    entries: list[Entry] = [
        Entry(
            label="（系统默认）",
            action=_pick(callbacks.on_select_mic, ""),
            checked=state.audio_source == "mic" and state.current_mic == "",
        )
    ]
    entries.extend(
        Entry(
            label=name,
            action=_pick(callbacks.on_select_mic, name),
            checked=state.audio_source == "mic" and state.current_mic == name,
        )
        for name in inputs.microphones
    )
    if inputs.system_audio:
        entries.append(separator())
        entries.extend(
            Entry(
                label=label,
                action=_pick(callbacks.on_select_system_audio, device_id),
                checked=state.audio_source == "system"
                and state.current_system_audio == device_id,
            )
            for label, device_id in inputs.system_audio
        )
    return tuple(entries)


def _translate_entries(state: TrayState, callbacks: TrayCallbacks) -> tuple[Entry, ...]:
    """Return the translation/subtitle rows: two switches plus their radios."""
    entries: list[Entry] = [
        Entry(
            label="开启离线翻译",
            action=callbacks.on_toggle_translate,
            checked=state.translate,
        ),
        separator(),
    ]
    entries.extend(
        Entry(
            label=label,
            action=_pick(callbacks.on_select_translate_route, route),
            checked=state.translate_route == route,
            radio=True,
            enabled=state.translate,
        )
        for route, label in ROUTE_LABELS.items()
    )
    entries.append(separator())
    entries.append(
        Entry(
            label="开启上屏字幕",
            action=callbacks.on_toggle_overlay,
            checked=state.overlay,
        )
    )
    entries.extend(
        Entry(
            label=label,
            action=_pick(callbacks.on_select_overlay_output, mode),
            checked=state.overlay_output == mode,
            radio=True,
            enabled=state.overlay,
        )
        for mode, label in OUTPUT_LABELS.items()
    )
    entries.append(separator())
    entries.append(Entry(text=lambda: translate_hint(state), enabled=False))
    return tuple(entries)


def _settings_entries(
    state: TrayState, callbacks: TrayCallbacks, inputs: MenuInputs
) -> tuple[Entry, ...]:
    """Everything configured once: keys, model, behaviour, updates."""
    hotkey_entries: list[Entry] = [
        Entry(
            label=HOTKEY_LABELS.get(name, str(name)),
            action=_pick_key(callbacks.on_select_hotkey, name),
            checked=state.current_hotkey == name,
            radio=True,
        )
        for name in inputs.hotkey_names
    ]
    hotkey_entries.append(separator())
    hotkey_entries.append(Entry(label="自定义按键…", action=callbacks.on_capture_hotkey))

    model_entries = [
        Entry(
            label=label,
            action=_pick(callbacks.on_select_model, kind),
            checked=state.current_model == kind,
            radio=True,
        )
        for kind, label in MODEL_LABELS.items()
    ]

    return (
        Entry(label="热键", children=tuple(hotkey_entries)),
        Entry(label="识别模型", children=tuple(model_entries)),
        Entry(
            label="语义顺滑（去语气词）",
            action=callbacks.on_toggle_disfluency,
            checked=state.disfluency == "smooth",
        ),
        Entry(label="持续听写", action=callbacks.on_toggle_continuous, checked=state.continuous),
        Entry(label="开机自启", action=callbacks.on_toggle_autostart, checked=state.autostart),
        separator(),
        Entry(label="输入诊断", action=callbacks.on_show_diagnostics),
        Entry(label="检查更新", action=callbacks.on_check_update),
    )


# -- labels --------------------------------------------------------------------


def state_text(state: TrayState) -> str:
    """Status line: what the app is doing and which source it listens to."""
    if not state.ready:
        return "状态：启动中…"
    source = "系统声音" if state.audio_source == "system" else "麦克风"
    if state.continuous:
        return f"状态：持续听写中（{source}）{translate_suffix(state)}"
    if state.paused:
        return "状态：已暂停（热键只保留原功能）"
    return f"状态：就绪（{source}）{translate_suffix(state)}"


def translate_suffix(state: TrayState) -> str:
    """Say what the subtitles actually show — translated, or source only."""
    if not state.translate:
        return " · 字幕显示原文（翻译未开）" if state.overlay else ""
    if not state.translator_ready:
        return " · 翻译开但模型未就绪"
    arrow = _ROUTE_ARROWS.get(state.translate_route, "中→英")
    return f" · 翻译开（{arrow}）"


def input_label(state: TrayState) -> str:
    """Parent row for the input submenu, carrying the current source."""
    if state.audio_source == "system":
        return f"输入源：系统声音 · {state.current_system_audio}"
    if state.current_mic:
        return f"输入源：{state.current_mic}"
    return "输入源：系统默认麦克风"


def translate_label(state: TrayState) -> str:
    """Parent row for translation+subtitles, carrying both states."""
    parts: list[str] = []
    if state.translate:
        parts.append(_ROUTE_ARROWS.get(state.translate_route, "翻译"))
    if state.overlay:
        parts.append(_OUTPUT_ARROWS.get(state.overlay_output, "字幕"))
    if not parts:
        return "翻译与字幕：关"
    return "翻译与字幕：" + " · ".join(parts)


def translate_hint(state: TrayState) -> str:
    """Explain what the current combination actually does (disabled row)."""
    if state.overlay and not state.translate:
        return "提示：字幕当前显示原文，未翻译"
    if state.translate and not state.translator_ready:
        return "提示：翻译模型未就绪（首次需下载约 240MB）"
    if state.overlay:
        return "提示：字幕即时刷新，打字按整句输出"
    return "提示：文字打到光标处"

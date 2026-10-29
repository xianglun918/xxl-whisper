"""Tray menu structure as pure data (no pystray, no platform imports).

Kept separate from :mod:`app.tray` for two reasons:

* the layout is a **design decision** worth testing — a short top-level list,
  related things grouped, and labels that carry the current state;
* :mod:`app.tray` pulls in pystray/PIL (display-bound), so nothing defined there
  can be unit-tested on a headless machine.

Everything dynamic is a **callable over a state provider**, never a snapshot:
pystray builds the native menu once and then asks the callbacks for the current
label/tick every time the menu is opened. Capturing ``state.paused`` at build
time (an earlier version did) freezes the tick marks — the switch works but the
menu keeps showing the old state.

Menu layout::

    状态行
    ─────────────
    暂停语音热键
    持续听写
    输入源      ▸ 麦克风 / 系统声音
    翻译与字幕  ▸ 翻译开关、方向、字幕开关、输出方式
    更多设置    ▸ 热键 / 模型 / 顺滑 / 自启 / 诊断 / 更新
    ─────────────
    退出
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from app.options import (
    DISFLUENCY_OPTIONS,
    HOTKEY_OPTIONS,
    MODEL_OPTIONS,
    OUTPUT_OPTIONS,
    OUTPUT_SHORT,
    ROUTE_OPTIONS,
    ROUTE_SHORT,
    Option,
    label_of,
)


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
    on_select_disfluency: Callable[[str], None]
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


type StateProvider = Callable[[], TrayState]


@dataclass(frozen=True, slots=True)
class MenuInputs:
    """Machine-dependent lists, gathered by the tray and injected here."""

    microphones: tuple[str, ...] = ()
    system_audio: tuple[tuple[str, str], ...] = ()  # (label, device id)


@dataclass(frozen=True, slots=True)
class Entry:
    """One menu row: an action, a checkbox, a disabled label, or a submenu."""

    label: str = ""
    action: Callable[[], None] | None = None
    #: Called when the menu opens; a plain bool here would freeze the tick mark.
    checked: Callable[[], bool] | None = None
    enabled: bool = True
    radio: bool = False
    children: tuple[Entry, ...] = ()
    separator: bool = False
    #: Rendered instead of ``label`` when the menu opens (live state).
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


def _tick(provider: StateProvider, getter: Callable[[TrayState], bool]) -> Callable[[], bool]:
    """Wrap a state getter so the tick is evaluated on every menu open."""
    return lambda: getter(provider())


def build_menu(
    state_provider: StateProvider,
    callbacks: TrayCallbacks,
    inputs: MenuInputs | None = None,
) -> tuple[Entry, ...]:
    """Return the tray menu; dynamic rows read *state_provider* when opened."""
    lists = inputs or MenuInputs()
    return (
        Entry(text=lambda: state_text(state_provider()), enabled=False),
        separator(),
        Entry(
            label="暂停语音热键",
            action=callbacks.on_toggle_pause,
            checked=_tick(state_provider, lambda state: state.paused),
        ),
        # 持续听写是和暂停同级的日常开关：换场景就要开/关，埋在二级里太深。
        Entry(
            label="持续听写",
            action=callbacks.on_toggle_continuous,
            checked=_tick(state_provider, lambda state: state.continuous),
        ),
        Entry(
            text=lambda: input_label(state_provider()),
            children=_input_entries(state_provider, callbacks, lists),
        ),
        Entry(
            text=lambda: translate_label(state_provider()),
            children=_translate_entries(state_provider, callbacks),
        ),
        Entry(
            label="更多设置",
            children=_settings_entries(state_provider, callbacks),
        ),
        separator(),
        Entry(label="退出", action=callbacks.on_exit),
    )


# -- submenus ------------------------------------------------------------------


def _option_entries(
    provider: StateProvider,
    options: tuple[Option, ...],
    callback: Callable[[str], None],
    current: Callable[[TrayState], object],
) -> list[Entry]:
    """Render one radio group; every group goes through this single path.

    Using one builder for all option groups is the point: the tick always reads
    ``current(state)`` from the live provider, so a group cannot silently end up
    comparing against the wrong config field (which is how the hotkey and
    filler-word rows once rendered with no tick at all).
    """
    return [
        Entry(
            label=option.label,
            action=_pick(callback, option.key),
            checked=_tick(provider, lambda state, key=option.key: current(state) == key),
            radio=True,
        )
        for option in options
    ]


def _input_entries(
    provider: StateProvider, callbacks: TrayCallbacks, inputs: MenuInputs
) -> tuple[Entry, ...]:
    """Microphones plus system-audio endpoints (Windows loopback)."""
    entries: list[Entry] = [
        Entry(
            label="（系统默认）",
            action=_pick(callbacks.on_select_mic, ""),
            checked=_tick(
                provider,
                lambda state: state.audio_source == "mic" and state.current_mic == "",
            ),
        )
    ]
    entries.extend(
        Entry(
            label=name,
            action=_pick(callbacks.on_select_mic, name),
            checked=_tick(
                provider,
                lambda state, name=name: state.audio_source == "mic"
                and state.current_mic == name,
            ),
        )
        for name in inputs.microphones
    )
    if inputs.system_audio:
        entries.append(separator())
        entries.extend(
            Entry(
                label=label,
                action=_pick(callbacks.on_select_system_audio, device_id),
                checked=_tick(
                    provider,
                    lambda state, device_id=device_id: state.audio_source == "system"
                    and state.current_system_audio == device_id,
                ),
            )
            for label, device_id in inputs.system_audio
        )
    return tuple(entries)


def _translate_entries(provider: StateProvider, callbacks: TrayCallbacks) -> tuple[Entry, ...]:
    """Return the translation/subtitle rows: two switches plus their radios."""
    entries: list[Entry] = [
        Entry(
            label="开启离线翻译",
            action=callbacks.on_toggle_translate,
            checked=_tick(provider, lambda state: state.translate),
        ),
        separator(),
    ]
    entries.extend(
        _option_entries(
            provider, ROUTE_OPTIONS, callbacks.on_select_translate_route,
            lambda state: state.translate_route,
        )
    )
    entries.append(separator())
    entries.append(
        Entry(
            label="开启上屏字幕",
            action=callbacks.on_toggle_overlay,
            checked=_tick(provider, lambda state: state.overlay),
        )
    )
    entries.extend(
        _option_entries(
            provider, OUTPUT_OPTIONS, callbacks.on_select_overlay_output,
            lambda state: state.overlay_output,
        )
    )
    entries.append(separator())
    entries.append(Entry(text=lambda: translate_hint(provider()), enabled=False))
    return tuple(entries)


def _settings_entries(
    provider: StateProvider, callbacks: TrayCallbacks
) -> tuple[Entry, ...]:
    """Everything configured once. Every choice is a radio over the config value."""
    hotkeys = _option_entries(
        provider, HOTKEY_OPTIONS, callbacks.on_select_hotkey,
        lambda state: state.current_hotkey,
    )
    hotkeys.append(separator())
    hotkeys.append(Entry(label="自定义按键…", action=callbacks.on_capture_hotkey))
    return (
        Entry(
            text=lambda: f"热键：{label_of(HOTKEY_OPTIONS, provider().current_hotkey)}",
            children=tuple(hotkeys),
        ),
        Entry(
            text=lambda: f"识别模型：{label_of(MODEL_OPTIONS, provider().current_model)}",
            children=tuple(
                _option_entries(
                    provider, MODEL_OPTIONS, callbacks.on_select_model,
                    lambda state: state.current_model,
                )
            ),
        ),
        Entry(
            text=lambda: f"语义顺滑：{label_of(DISFLUENCY_OPTIONS, provider().disfluency)}",
            children=tuple(
                _option_entries(
                    provider, DISFLUENCY_OPTIONS, callbacks.on_select_disfluency,
                    lambda state: state.disfluency,
                )
            ),
        ),
        Entry(
            label="开机自启",
            action=callbacks.on_toggle_autostart,
            checked=_tick(provider, lambda state: state.autostart),
        ),
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
    arrow = ROUTE_SHORT.get(state.translate_route, "中→英")
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
        parts.append(ROUTE_SHORT.get(state.translate_route, "翻译"))
    if state.overlay:
        parts.append(OUTPUT_SHORT.get(state.overlay_output, "字幕"))
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

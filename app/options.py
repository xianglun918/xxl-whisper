"""Menu option tables — the single source of truth for every choice the UI offers.

Each table pairs the **config value** (what gets written to ``config.toml`` and
what the app compares against) with the **label** shown in the tray. Before this
module existed the menu carried its own hand-copied tables, which is how a group
ends up rendering with no tick: the label table, the comparison value and the
config literal drift apart and nothing fails loudly.

The keys here are checked against the config's own literals by
``tests/test_options_consistency.py`` — if the config vocabulary changes and
these tables are not updated, that test fails instead of the user finding a dead
menu row.

Pure standard library on purpose: importing ``app.config`` pulls in the platform
layer, which cannot be imported on a headless test machine.
"""

from __future__ import annotations

from dataclasses import dataclass

from app import keycodes


@dataclass(frozen=True, slots=True)
class Option:
    """One selectable value: the config value plus its human label."""

    key: str
    label: str


#: Human labels for every preset name in either platform key table.
_HOTKEY_LABELS: dict[str, str] = {
    "caps_lock": "CapsLock",
    "f2": "F2",
    "f4": "F4",
    "f6": "F6",
    "f8": "F8",
    "scroll_lock": "Scroll Lock",
    "mouse_x1": "鼠标侧键 X1（后退键）",
    "mouse_x2": "鼠标侧键 X2（前进键）",
    "right_cmd": "右 Command",
}

#: Push-to-talk key presets, built **from the active platform table**
#: (:mod:`app.keycodes`) instead of a hard-coded list. A custom recorded key is
#: stored as a raw keycode integer. Deriving the menu from the same table the
#: hook arms is what stops the two from drifting: macOS used to offer CapsLock /
#: Scroll Lock / mouse side buttons it cannot observe, while hiding its own
#: default 右 Command.
HOTKEY_OPTIONS: tuple[Option, ...] = tuple(
    Option(key, _HOTKEY_LABELS[key]) for key in keycodes.PRESET_KEYCODES
)

#: Recogniser model, mirroring ``config.ModelKind``.
MODEL_OPTIONS: tuple[Option, ...] = (
    Option("sensevoice", "SenseVoice-Small（默认 · 230MB · 快 · 中日英韩粤）"),
    Option("funasr_nano", "Fun-ASR-Nano（中文更准 · 约 1GB · 日语偏弱）"),
    Option("whisper_turbo", "Whisper large-v3-turbo（日语最强 · 1.04GB · 较慢）"),
)

#: Filler-word handling, mirroring ``config.Disfluency``.
DISFLUENCY_OPTIONS: tuple[Option, ...] = (
    Option("verbatim", "逐字原文（保留嗯、呃、重复）"),
    Option("smooth", "语义顺滑（去掉语气词与重复）"),
)

#: Translation direction, mirroring ``config.TranslateRoute``.
ROUTE_OPTIONS: tuple[Option, ...] = (
    Option("zh-en", "中 → 英（说中文出英文）"),
    Option("en-zh", "英 → 中（说英文出中文）"),
    Option("ja-zh", "日 → 中（经英文中转，两段）"),
    Option("ja-en", "日 → 英（一段直出，最快最省内存）"),
)

#: Where the output goes, mirroring ``config.OverlayOutput``.
OUTPUT_OPTIONS: tuple[Option, ...] = (
    Option("screen", "只上屏（不打字）"),
    Option("both", "上屏 + 打字到光标"),
    Option("type", "只打字（不上屏）"),
)

#: Compact route name for parent-row labels ("翻译与字幕：日→中").
ROUTE_SHORT: dict[str, str] = {
    "zh-en": "中→英",
    "en-zh": "英→中",
    "ja-zh": "日→中",
    "ja-en": "日→英",
}

#: Compact output name for parent-row labels ("… · 上屏+打字").
OUTPUT_SHORT: dict[str, str] = {
    "screen": "只上屏",
    "both": "上屏+打字",
    "type": "只打字不上屏",
}

#: Compact labels used in the status line for the two filler modes.
DISFLUENCY_SHORT: dict[str, str] = {"verbatim": "逐字", "smooth": "顺滑"}


def label_of(options: tuple[Option, ...], key: object) -> str:
    """Return the label for *key*, falling back to its string form.

    A custom hotkey is stored as a VK integer, so it never matches a preset;
    it still has to render as something a human can read.
    """
    for option in options:
        if option.key == key:
            return option.label
    if isinstance(key, int):
        return f"自定义按键（VK 0x{key:02X}）"
    return str(key)

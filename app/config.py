"""Config loading/saving: TOML at %LOCALAPPDATA%/xxl-whisper/config.toml.

The file crosses the trust boundary (user-editable), so parsing is total:
either a valid frozen :class:`Config` or a typed :class:`ConfigError`.
"""

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import NoReturn

APP_DIR_NAME: str = "xxl-whisper"

#: Virtual-key codes for hotkeys we allow. Keys that are dangerous to
#: remap (modifiers, typing keys) are deliberately absent.
HOTKEY_VK: Mapping[str, int] = MappingProxyType(
    {
        "caps_lock": 0x14,
        "f2": 0x71,
        "f4": 0x73,
        "f6": 0x75,
        "f8": 0x77,
        "scroll_lock": 0x91,
    }
)

_LANGUAGES: Mapping[str, None] = MappingProxyType(
    {"zh": None, "auto": None, "en": None, "ja": None, "ko": None, "yue": None}
)


class ConfigError(Exception):
    """Raised when the config file cannot be parsed into a valid Config."""

    def __init__(self, reason: str, path: Path) -> None:
        super().__init__(f"{path}: {reason}")
        self.reason = reason
        self.path = path


@dataclass(frozen=True, slots=True)
class Config:
    hotkey: str
    hold_threshold_ms: int
    mic: str
    num_threads: int
    language: str
    restore_clipboard: bool
    paste_delay_ms: int


def default_config() -> Config:
    """First-run defaults: CapsLock push-to-talk, Chinese, 250 ms click cutoff."""
    return Config(
        hotkey="caps_lock",
        hold_threshold_ms=250,
        mic="",
        num_threads=2,
        language="zh",
        restore_clipboard=True,
        paste_delay_ms=200,
    )


def config_dir() -> Path:
    """Per-user data root: %LOCALAPPDATA%/xxl-whisper."""
    base = os.environ.get("LOCALAPPDATA")
    root = Path(base) if base else Path.home() / "AppData" / "Local"
    return root / APP_DIR_NAME


def config_path() -> Path:
    """Location of the user-editable TOML config."""
    return config_dir() / "config.toml"


def model_dir() -> Path:
    """Where the SenseVoice model artifacts live."""
    return config_dir() / "models" / "sensevoice"


def load_config(path: Path) -> Config:
    """Parse a config file into a Config; missing file yields defaults."""
    if not path.exists():
        return default_config()
    try:
        raw: dict[str, object] = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(reason=f"malformed TOML ({exc})", path=path) from exc
    p = _Parser(raw=raw, path=path)
    return Config(
        hotkey=p.choice("hotkey", "caps_lock", HOTKEY_VK),
        hold_threshold_ms=p.int_in("hold_threshold_ms", 250, 10, 2_000),
        mic=p.text("mic", ""),
        num_threads=p.int_in("num_threads", 2, 1, 16),
        language=p.choice("language", "zh", _LANGUAGES),
        restore_clipboard=p.bool_flag("restore_clipboard", True),
        paste_delay_ms=p.int_in("paste_delay_ms", 200, 50, 2_000),
    )


def save_config(path: Path, config: Config) -> None:
    """Persist a Config as simple key = value TOML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"hotkey = {_quote(config.hotkey)}",
        f"hold_threshold_ms = {config.hold_threshold_ms}",
        f"mic = {_quote(config.mic)}",
        f"num_threads = {config.num_threads}",
        f"language = {_quote(config.language)}",
        f"restore_clipboard = {str(config.restore_clipboard).lower()}",
        f"paste_delay_ms = {config.paste_delay_ms}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class _Parser:
    """Field validators sharing the raw table and error path."""

    __slots__ = ("_path", "_raw")

    def __init__(self, raw: dict[str, object], path: Path) -> None:
        self._raw = raw
        self._path = path

    def int_in(self, name: str, default: int, lo: int, hi: int) -> int:
        value = self._raw.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int):
            self._fail(f"{name} must be an integer")
        if not lo <= value <= hi:
            self._fail(f"{name} out of range [{lo}, {hi}]")
        return value

    def bool_flag(self, name: str, default: bool) -> bool:
        value = self._raw.get(name, default)
        if not isinstance(value, bool):
            self._fail(f"{name} must be a boolean")
        return value

    def choice(self, name: str, default: str, choices: Mapping[str, object]) -> str:
        value = self._raw.get(name, default)
        if not isinstance(value, str) or value not in choices:
            allowed = ", ".join(sorted(choices))
            self._fail(f"{name} must be one of: {allowed}")
        return value

    def text(self, name: str, default: str) -> str:
        value = self._raw.get(name, default)
        if not isinstance(value, str):
            self._fail(f"{name} must be a string")
        return value

    def _fail(self, reason: str) -> NoReturn:
        raise ConfigError(reason=reason, path=self._path)


def _quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'

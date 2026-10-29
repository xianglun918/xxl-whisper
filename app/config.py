"""Config loading/saving: TOML at %LOCALAPPDATA%/xxl-whisper/config.toml.

The file crosses the trust boundary (user-editable), so parsing is total:
either a valid frozen :class:`Config` or a typed :class:`ConfigError`.
"""

import dataclasses
import logging
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal, NoReturn, cast

from app import native

APP_DIR_NAME: str = "xxl-whisper"

log = logging.getLogger(__name__)

type ModelKind = Literal["sensevoice", "funasr_nano", "whisper_turbo"]

#: Disfluency handling for the LLM-decoder model (Fun-ASR-Nano). verbatim keeps
#: fillers (嗯/呃/啊), repetitions and false starts; smooth asks the decoder to
#: clean them for chat-style output.
type Disfluency = Literal["verbatim", "smooth"]

#: Translation target language. "en" means Chinese speech -> English text
#: (zh-en model); "zh" is the reverse (en-zh model).
type TranslateTarget = Literal["en", "zh"]

#: Spoken language fed into the translator. Japanese has no direct ja→zh
#: weight, so "ja" is chained through English (see ``TRANSLATE_ROUTE_MODELS``).
type TranslateSource = Literal["zh", "en", "ja"]

#: The supported (source, target) combinations, as one canonical key.
type TranslateRoute = Literal["zh-en", "en-zh", "ja-zh", "ja-en"]

#: What gets transcribed: the microphone, or the system audio mix (WASAPI
#: loopback on Windows; a virtual device such as BlackHole on macOS).
type AudioSource = Literal["mic", "system"]

#: What finally reaches the cursor: only the translation, or source + translation.
type TranslateMode = Literal["replace", "bilingual"]

#: What the subtitle overlay shows: only the translation, only what was said,
#: or both. Mirrors ``app.overlay.OverlayMode`` (kept local so this parser never
#: imports the window module).
type OverlayMode = Literal["target", "source", "bilingual"]

#: Where an utterance goes: only to the cursor, only to the subtitle overlay, or
#: both. ``screen`` is the read-only "don't type into my meeting notes" mode.
type OverlayOutput = Literal["type", "screen", "both"]

#: Preset hotkey names -> platform keycodes (owned by the platform hook module).
#: Any other keycode (1..254) is also accepted as a raw integer for
#: user-defined keys. Keys that are dangerous to remap (modifiers) are
#: rejected at capture time, not here.
HOTKEY_VK: Mapping[str, int] = native.hotkey.PRESET_KEYCODES

#: Buttons that live on the mouse, watched by the mouse LL hook instead of the
#: keyboard LL hook (empty on macOS).
MOUSE_VKS: frozenset[int] = native.hotkey.MOUSE_KEYCODES

_MODELS: Mapping[str, None] = MappingProxyType(
    {"sensevoice": None, "funasr_nano": None, "whisper_turbo": None}
)

_DISFLUENCY: Mapping[str, None] = MappingProxyType(
    {"verbatim": None, "smooth": None}
)

_TRANSLATE_TARGETS: Mapping[str, None] = MappingProxyType({"en": None, "zh": None})

_TRANSLATE_SOURCES: Mapping[str, None] = MappingProxyType(
    {"zh": None, "en": None, "ja": None}
)

#: Only these combinations have models behind them; anything else is a config
#: error the user must fix rather than a silent no-op.
_TRANSLATE_ROUTES: Mapping[str, None] = MappingProxyType(
    {"zh-en": None, "en-zh": None, "ja-zh": None, "ja-en": None}
)

_AUDIO_SOURCES: Mapping[str, None] = MappingProxyType({"mic": None, "system": None})

_TRANSLATE_MODES: Mapping[str, None] = MappingProxyType(
    {"replace": None, "bilingual": None}
)

_OVERLAY_MODES: Mapping[str, None] = MappingProxyType(
    {"target": None, "source": None, "bilingual": None}
)

_OVERLAY_OUTPUTS: Mapping[str, None] = MappingProxyType(
    {"type": None, "screen": None, "both": None}
)

_MIN_CUSTOM_VK: int = 1
_MAX_CUSTOM_VK: int = 254

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
    hotkey: str | int
    hold_threshold_ms: int
    mic: str
    audio_source: AudioSource
    num_threads: int
    language: str
    restore_clipboard: bool
    paste_delay_ms: int
    check_updates: bool
    model: ModelKind
    proxy: str
    disfluency: Disfluency
    continuous: bool
    vad_threshold: float
    vad_min_speech_ms: int
    live_min_silence_ms: int
    live_max_speech_ms: int
    typing_min_silence_ms: int
    typing_max_speech_ms: int
    typing_merge_ms: int
    typing_merge_chars: int
    translate: bool
    translate_target: TranslateTarget
    translate_source: TranslateSource
    translate_mode: TranslateMode
    overlay_enabled: bool
    overlay_output: OverlayOutput
    overlay_mode: OverlayMode
    overlay_font_size: int
    overlay_opacity: float
    overlay_width_ratio: float
    overlay_bottom_margin: int
    overlay_hold_seconds: float
    overlay_max_lines: int
    partial_subtitle: bool
    partial_subtitle_ms: int
    overlay_click_through: bool


def default_config() -> Config:
    """First-run defaults: platform push-to-talk key, Chinese, 400 ms cutoff."""
    return Config(
        hotkey=native.hotkey.DEFAULT_HOTKEY,
        hold_threshold_ms=400,
        mic="",
        audio_source="mic",
        num_threads=2,
        language="zh",
        restore_clipboard=True,
        paste_delay_ms=200,
        check_updates=True,
        model="sensevoice",
        proxy="",
        disfluency="verbatim",
        continuous=False,
        vad_threshold=0.5,
        vad_min_speech_ms=250,
        # Continuous dictation ("live"): the subtitle bar needs the line while
        # it is still being talked about, so sentences are cut briskly. Typing
        # does not suffer from this — app.type_batcher glues the cut pieces back
        # together and pastes one whole sentence. Push-to-talk never uses these
        # (a held key is one utterance).
        live_min_silence_ms=350,
        live_max_speech_ms=4_000,
        typing_min_silence_ms=500,
        typing_max_speech_ms=20_000,
        #: How long a force-cut batch may stay open before it is typed anyway.
        typing_merge_ms=12_000,
        #: Character budget that closes a force-cut batch early.
        typing_merge_chars=80,
        translate=False,
        translate_target="en",
        translate_source="zh",
        translate_mode="bilingual",
        overlay_enabled=False,
        overlay_output="both",
        overlay_mode="target",
        overlay_font_size=32,
        overlay_opacity=0.85,
        overlay_width_ratio=0.72,
        overlay_bottom_margin=80,
        overlay_hold_seconds=8.0,
        overlay_max_lines=6,
        partial_subtitle=True,
        partial_subtitle_ms=800,
        overlay_click_through=True,
    )


def config_dir() -> Path:
    """Per-user data root (%LOCALAPPDATA% on Windows, Application Support on macOS)."""
    return native.data_root()


def config_path() -> Path:
    """Location of the user-editable TOML config."""
    return config_dir() / "config.toml"


def model_dir() -> Path:
    """Where the SenseVoice model artifacts live."""
    return models_root() / "sensevoice"


def models_root() -> Path:
    """Where model artifacts live (%LOCALAPPDATA%/xxl-whisper/models)."""
    return config_dir() / "models"


def updates_dir() -> Path:
    """Where downloaded release builds are staged (%LOCALAPPDATA%/xxl-whisper/updates)."""
    return config_dir() / "updates"


def hotkey_vk(hotkey: str | int) -> int:
    """Resolve a preset name or raw VK into the virtual-key code."""
    if isinstance(hotkey, int):
        return hotkey
    return HOTKEY_VK[hotkey]


def translate_route(source: str, target: str) -> TranslateRoute | None:
    """Canonical route key ("ja-zh"), or None when the pair has no models."""
    key = f"{source}-{target}"
    return cast("TranslateRoute", key) if key in _TRANSLATE_ROUTES else None


def segmenter_limits(config: Config) -> tuple[int, int]:
    """Return ``(min_silence_ms, max_speech_ms)`` for a continuous session.

    Two very different jobs share one VAD:

    * **subtitles** — the bar must fill while someone is still talking, so a
      monologue is force-cut every few seconds (``live_*``);
    * **typing only** — the text lands in a document, where a cut in the middle of
      a word corrupts the sentence. Long, complete utterances win, so the loose
      ``typing_*`` limits (the pre-subtitle behaviour) apply.

    Cutting at 4 s while typing would also be a silent regression for users who
    never enabled subtitles at all.
    """
    if config.overlay_output == "type" or not config.overlay_enabled:
        return config.typing_min_silence_ms, config.typing_max_speech_ms
    return config.live_min_silence_ms, config.live_max_speech_ms


def with_overlay_mode(config: Config, mode: OverlayOutput) -> Config:
    """Return *config* with *mode* applied; the subtitle bar follows the mode.

    "只打字" means no bar, so the two cannot disagree. Keeping this in one
    function is what stops the menu from offering a contradiction such as
    "字幕已开启" + "只打字（无字幕）".
    """
    return dataclasses.replace(
        config, overlay_output=mode, overlay_enabled=mode != "type"
    )


def parse_translate_route(route: str) -> TranslateRoute | None:
    """Validate a "source-target" key from the tray; None when unsupported."""
    source, sep, target = route.partition("-")
    if not sep:
        return None
    return translate_route(source, target)


def route_of(config: Config) -> TranslateRoute | None:
    """Route key for a loaded config (None only for a hand-edited bad pair)."""
    return translate_route(config.translate_source, config.translate_target)


def load_config(path: Path) -> Config:
    """Parse a config file into a Config; missing file yields defaults."""
    if not path.exists():
        return default_config()
    try:
        raw: dict[str, object] = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(reason=f"malformed TOML ({exc})", path=path) from exc
    p = _Parser(raw=raw, path=path)
    target = p.translate_target("translate_target", "en")
    # Configs written before translate_source existed recorded only the target,
    # and 中→英 / 英→中 were the only routes, so infer the source from it.
    inferred: TranslateSource = "en" if target == "zh" else "zh"
    source = p.translate_source("translate_source", inferred)
    if translate_route(source, target) is None:
        # Forgiving on purpose: a hand-edited pair we have no models for must
        # not stop dictation from starting. Fall back to the route the target
        # implies and say so in the log.
        log.warning(
            "unsupported translate direction %s->%s; using %s->%s",
            source,
            target,
            inferred,
            target,
        )
        source = inferred
    return Config(
        hotkey=p.hotkey("hotkey", native.hotkey.DEFAULT_HOTKEY),
        hold_threshold_ms=p.int_in("hold_threshold_ms", 400, 10, 2_000),
        mic=p.text("mic", ""),
        audio_source=p.audio_source("audio_source", "mic"),
        num_threads=p.int_in("num_threads", 2, 1, 16),
        language=p.choice("language", "zh", _LANGUAGES),
        restore_clipboard=p.bool_flag("restore_clipboard", True),
        paste_delay_ms=p.int_in("paste_delay_ms", 200, 50, 2_000),
        check_updates=p.bool_flag("check_updates", True),
        model=p.model_kind("model", "sensevoice"),
        proxy=p.text("proxy", ""),
        disfluency=p.disfluency_kind("disfluency", "verbatim"),
        continuous=p.bool_flag("continuous", False),
        vad_threshold=p.float_in("vad_threshold", 0.5, 0.0, 1.0),
        vad_min_speech_ms=p.int_in("vad_min_speech_ms", 250, 50, 5_000),
        live_min_silence_ms=p.int_in(
            "live_min_silence_ms",
            p.legacy_default("vad_min_silence_ms", 500, 50, 10_000, 350),
            50,
            10_000,
        ),
        live_max_speech_ms=p.int_in(
            "live_max_speech_ms",
            p.legacy_default("vad_max_speech_ms", 20_000, 1_000, 60_000, 4_000),
            1_000,
            60_000,
        ),
        typing_min_silence_ms=p.int_in("typing_min_silence_ms", 500, 50, 10_000),
        typing_max_speech_ms=p.int_in("typing_max_speech_ms", 20_000, 1_000, 60_000),
        typing_merge_ms=p.int_in("typing_merge_ms", 12_000, 0, 60_000),
        typing_merge_chars=p.int_in("typing_merge_chars", 80, 10, 500),
        translate=p.bool_flag("translate", False),
        translate_target=target,
        translate_source=source,
        translate_mode=p.translate_mode("translate_mode", "bilingual"),
        # The bar exists iff text is not typing-only: one decision, not two.
        overlay_enabled=(
            p.bool_flag("overlay_enabled", False)
            and p.overlay_output("overlay_output", "both") != "type"
        ),
        overlay_output=p.overlay_output("overlay_output", "both"),
        overlay_mode=p.overlay_mode("overlay_mode", "target"),
        overlay_font_size=p.int_in("overlay_font_size", 32, 10, 72),
        overlay_opacity=p.float_in("overlay_opacity", 0.85, 0.2, 1.0),
        overlay_width_ratio=p.float_in("overlay_width_ratio", 0.72, 0.2, 1.0),
        overlay_bottom_margin=p.int_in("overlay_bottom_margin", 80, 0, 600),
        overlay_hold_seconds=p.float_in("overlay_hold_seconds", 8.0, 0.5, 60.0),
        overlay_max_lines=p.int_in("overlay_max_lines", 6, 1, 12),
        partial_subtitle=p.bool_flag("partial_subtitle", True),
        partial_subtitle_ms=p.int_in("partial_subtitle_ms", 800, 200, 5_000),
        overlay_click_through=p.bool_flag("overlay_click_through", True),
    )


def save_config(path: Path, config: Config) -> None:
    """Persist a Config as simple key = value TOML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"hotkey = {config.hotkey if isinstance(config.hotkey, int) else _quote(config.hotkey)}",
        f"hold_threshold_ms = {config.hold_threshold_ms}",
        f"mic = {_quote(config.mic)}",
        f"audio_source = {_quote(config.audio_source)}",
        f"num_threads = {config.num_threads}",
        f"language = {_quote(config.language)}",
        f"restore_clipboard = {str(config.restore_clipboard).lower()}",
        f"paste_delay_ms = {config.paste_delay_ms}",
        f"check_updates = {str(config.check_updates).lower()}",
        f"model = {_quote(config.model)}",
        f"proxy = {_quote(config.proxy)}",
        f"disfluency = {_quote(config.disfluency)}",
        f"continuous = {str(config.continuous).lower()}",
        f"vad_threshold = {config.vad_threshold}",
        f"vad_min_speech_ms = {config.vad_min_speech_ms}",
        f"live_min_silence_ms = {config.live_min_silence_ms}",
        f"live_max_speech_ms = {config.live_max_speech_ms}",
        f"typing_min_silence_ms = {config.typing_min_silence_ms}",
        f"typing_max_speech_ms = {config.typing_max_speech_ms}",
        f"typing_merge_ms = {config.typing_merge_ms}",
        f"typing_merge_chars = {config.typing_merge_chars}",
        f"translate = {str(config.translate).lower()}",
        f"translate_target = {_quote(config.translate_target)}",
        f"translate_source = {_quote(config.translate_source)}",
        f"translate_mode = {_quote(config.translate_mode)}",
        f"overlay_enabled = {str(config.overlay_enabled).lower()}",
        f"overlay_output = {_quote(config.overlay_output)}",
        f"overlay_mode = {_quote(config.overlay_mode)}",
        f"overlay_font_size = {config.overlay_font_size}",
        f"overlay_opacity = {config.overlay_opacity}",
        f"overlay_width_ratio = {config.overlay_width_ratio}",
        f"overlay_bottom_margin = {config.overlay_bottom_margin}",
        f"overlay_hold_seconds = {config.overlay_hold_seconds}",
        f"overlay_max_lines = {config.overlay_max_lines}",
        f"partial_subtitle = {str(config.partial_subtitle).lower()}",
        f"partial_subtitle_ms = {config.partial_subtitle_ms}",
        f"overlay_click_through = {str(config.overlay_click_through).lower()}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class _Parser:
    """Field validators sharing the raw table and error path."""

    __slots__ = ("_path", "_raw")

    def __init__(self, raw: dict[str, object], path: Path) -> None:
        self._raw = raw
        self._path = path

    def legacy_default(
        self, name: str, legacy_default: int, lo: int, hi: int, new_default: int
    ) -> int:
        """Use an old key's value only when the user actually changed it.

        A config written by an older build always contains ``vad_*`` at their
        *then* defaults. Inheriting those values blindly would pin an upgraded
        user to the old 20 s / 500 ms behaviour for ever, which is exactly the
        "waits until the whole speech ends" bug. Only a value that differs from
        the old default counts as a deliberate choice.
        """
        value = self.int_in(name, legacy_default, lo, hi)
        return new_default if value == legacy_default else value

    def int_in(self, name: str, default: int, lo: int, hi: int) -> int:
        value = self._raw.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int):
            self._fail(f"{name} must be an integer")
        if not lo <= value <= hi:
            self._fail(f"{name} out of range [{lo}, {hi}]")
        return value

    def float_in(self, name: str, default: float, lo: float, hi: float) -> float:
        value = self._raw.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            self._fail(f"{name} must be a number")
        number = float(value)
        if not lo <= number <= hi:
            self._fail(f"{name} out of range [{lo}, {hi}]")
        return number

    def bool_flag(self, name: str, default: bool) -> bool:
        value = self._raw.get(name, default)
        if not isinstance(value, bool):
            self._fail(f"{name} must be a boolean")
        return value

    def choice(self, name: str, default: str, choices: Mapping[str, object]) -> str:
        value = self._raw.get(name, default)
        if not isinstance(value, str) or value not in choices:
            self._fail(f"{name} must be one of: {', '.join(sorted(choices))}")
        return value

    def model_kind(self, name: str, default: ModelKind) -> ModelKind:
        value = self._raw.get(name, default)
        if not isinstance(value, str) or value not in _MODELS:
            self._fail(f"{name} must be one of: {', '.join(sorted(_MODELS))}")
        if value == "sensevoice":
            return "sensevoice"
        return "funasr_nano"

    def audio_source(self, name: str, default: AudioSource) -> AudioSource:
        value = self.choice(name, default, _AUDIO_SOURCES)
        return "system" if value == "system" else "mic"

    def disfluency_kind(self, name: str, default: Disfluency) -> Disfluency:
        value = self._raw.get(name, default)
        if not isinstance(value, str) or value not in _DISFLUENCY:
            self._fail(f"{name} must be one of: {', '.join(sorted(_DISFLUENCY))}")
        if value == "verbatim":
            return "verbatim"
        return "smooth"

    def translate_target(self, name: str, default: TranslateTarget) -> TranslateTarget:
        value = self.choice(name, default, _TRANSLATE_TARGETS)
        return "en" if value == "en" else "zh"

    def translate_source(self, name: str, default: TranslateSource) -> TranslateSource:
        value = self.choice(name, default, _TRANSLATE_SOURCES)
        match value:
            case "ja":
                return "ja"
            case "en":
                return "en"
            case _:
                return "zh"

    def translate_mode(self, name: str, default: TranslateMode) -> TranslateMode:
        value = self.choice(name, default, _TRANSLATE_MODES)
        return "replace" if value == "replace" else "bilingual"

    def overlay_mode(self, name: str, default: OverlayMode) -> OverlayMode:
        value = self.choice(name, default, _OVERLAY_MODES)
        match value:
            case "source":
                return "source"
            case "bilingual":
                return "bilingual"
            case _:
                return "target"

    def overlay_output(self, name: str, default: OverlayOutput) -> OverlayOutput:
        value = self.choice(name, default, _OVERLAY_OUTPUTS)
        match value:
            case "screen":
                return "screen"
            case "both":
                return "both"
            case _:
                return "type"

    def hotkey(self, name: str, default: str | int) -> str | int:
        """Accept a preset name or a raw virtual-key code (1..254)."""
        value = self._raw.get(name, default)
        if isinstance(value, str):
            if value not in HOTKEY_VK:
                self._fail(f"{name} must be a preset name or a VK integer")
            return value
        if isinstance(value, bool) or not isinstance(value, int):
            self._fail(f"{name} must be a preset name or a VK integer")
        if not _MIN_CUSTOM_VK <= value <= _MAX_CUSTOM_VK:
            self._fail(f"{name} VK out of range [{_MIN_CUSTOM_VK}, {_MAX_CUSTOM_VK}]")
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

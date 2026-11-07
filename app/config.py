"""Config loading/saving: TOML at %LOCALAPPDATA%/xxl-whisper/config.toml.

The file crosses the trust boundary (user-editable), so parsing is total:
either a valid frozen :class:`Config` or a typed :class:`ConfigError`.
"""

from __future__ import annotations

import dataclasses
import logging
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, NoReturn, cast

from app import native
from app.keycodes import DEFAULT_HOTKEY, MOUSE_KEYCODES, PRESET_KEYCODES
from app.routes import ROUTES, pin_language, route_keys

APP_DIR_NAME: str = "xxl-whisper"

log = logging.getLogger(__name__)

#: On-disk config schema version. Bump it whenever a *stored* key changes
#: meaning, and add the step to :func:`_migrate`. Absent in files written by
#: builds that predate the stamp, which therefore read as version 1.
#:
#: Deliberately **not** bumped by the overlay_mode spelling rename
#: (bilingual → stacked): the parser accepts both, so an old file keeps
#: working, and a bump would rewrite every v2 file on first launch (losing
#: comments and non-scalar keys) and make downgrading fatal — an older build
#: reading "stacked" exits with a config error. The new spelling is written
#: by the next ordinary save.
CONFIG_VERSION: int = 2

#: Keys older builds wrote that this build no longer writes, with what they map
#: to. Their *values* were already honoured at load time (``legacy_default``);
#: the rewrite is what finally removes the stale lines.
RENAMED_KEYS: Mapping[str, str] = MappingProxyType(
    {
        "vad_min_silence_ms": "live_min_silence_ms",
        "vad_max_speech_ms": "live_max_speech_ms",
    }
)

type ModelKind = Literal["sensevoice", "funasr_nano", "whisper_turbo"]

#: Disfluency handling for the LLM-decoder model (Fun-ASR-Nano). verbatim keeps
#: fillers (嗯/呃/啊), repetitions and false starts; smooth asks the decoder to
#: clean them for chat-style output.
type Disfluency = Literal["verbatim", "smooth"]

#: Translation target language. "en" means Chinese speech -> English text
#: (zh-en model); "zh" is the reverse (en-zh model).
type TranslateTarget = Literal["en", "zh"]

#: Spoken language fed into the translator. ja→zh ships one direct model; the
#: ja→en→zh chain is only the fallback (see :data:`app.routes.ROUTES`).
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
type OverlayMode = Literal["target", "source", "stacked"]

#: Where an utterance goes: only to the cursor, only to the subtitle overlay, or
#: both. ``screen`` is the read-only "don't type into my meeting notes" mode.
type OverlayOutput = Literal["type", "screen", "both"]

#: Preset hotkey names -> platform keycodes (data owned by :mod:`app.keycodes`).
#: Any other keycode (1..254) is also accepted as a raw integer for
#: user-defined keys. Keys that are dangerous to remap (modifiers) are
#: rejected at capture time, not here.
HOTKEY_VK: Mapping[str, int] = PRESET_KEYCODES

#: Buttons that live on the mouse, watched by the mouse LL hook instead of the
#: keyboard LL hook (empty on macOS).
MOUSE_VKS: frozenset[int] = MOUSE_KEYCODES

_MODELS: Mapping[str, None] = MappingProxyType(
    {"sensevoice": None, "funasr_nano": None, "whisper_turbo": None}
)

_DISFLUENCY: Mapping[str, None] = MappingProxyType(
    {"verbatim": None, "smooth": None}
)

#: Runtime whitelists, projected from :data:`app.routes.ROUTES`. A route added
#: there is accepted here automatically; a hand-written copy would reject the
#: very direction the menu just offered ("translate_target must be one of …").
_TRANSLATE_TARGETS: Mapping[str, None] = MappingProxyType(
    dict.fromkeys(spec.target for spec in ROUTES)
)

_TRANSLATE_SOURCES: Mapping[str, None] = MappingProxyType(
    dict.fromkeys(spec.source for spec in ROUTES)
)

#: Only these combinations have models behind them; anything else is a config
#: error the user must fix rather than a silent no-op. The list itself lives in
#: :mod:`app.routes` — this is only the membership test.
_TRANSLATE_ROUTES: Mapping[str, None] = MappingProxyType(dict.fromkeys(route_keys()))

_AUDIO_SOURCES: Mapping[str, None] = MappingProxyType({"mic": None, "system": None})

_TRANSLATE_MODES: Mapping[str, None] = MappingProxyType(
    {"replace": None, "bilingual": None}
)

_OVERLAY_MODES: Mapping[str, None] = MappingProxyType(
    {"target": None, "source": None, "stacked": None}
)

#: Spellings an older build wrote for a current value. Accepted on read (a stored
#: file must keep starting) but never offered back: the error message and the
#: saved file only ever mention :data:`_OVERLAY_MODES`.
_OVERLAY_MODE_LEGACY: Mapping[str, str] = MappingProxyType({"bilingual": "stacked"})

_OVERLAY_OUTPUTS: Mapping[str, None] = MappingProxyType(
    {"type": None, "screen": None, "both": None}
)

_MIN_CUSTOM_VK: int = 1
_MAX_CUSTOM_VK: int = 254

_LANGUAGES: Mapping[str, None] = MappingProxyType(
    {"zh": None, "auto": None, "en": None, "ja": None, "ko": None, "yue": None}
)


#: How a field's value is rendered back into the file (parsing is per row).
type _Kind = Literal["str", "int", "float", "bool", "hotkey"]


@dataclass(frozen=True, slots=True)
class _Field:
    """One row of the config schema.

    Before this table existed, adding a setting meant editing four places and
    keeping their defaults in step by hand — the dataclass, ``default_config()``,
    the ``Config(...)`` literal inside ``load_config()`` and the line list in
    ``save_config()`` — and two of them had already drifted. A field is now one
    row: name, default, how it is parsed, and how it is written back.
    """

    name: str
    default: object
    kind: _Kind
    parse: Callable[[_Parser], object]


def _parse(method: str, name: str, *args: object) -> Callable[[_Parser], object]:
    """Bind one :class:`_Parser` validator by name (the table predates a parser)."""
    return lambda p: cast("Callable[..., object]", getattr(p, method))(name, *args)


def _text(name: str, default: str) -> _Field:
    return _Field(name, default, "str", _parse("text", name, default))


def _flag(name: str, default: bool) -> _Field:
    return _Field(name, default, "bool", _parse("bool_flag", name, default))


def _int(name: str, default: int, lo: int, hi: int) -> _Field:
    return _Field(name, default, "int", _parse("int_in", name, default, lo, hi))


def _float(name: str, default: float, lo: float, hi: float) -> _Field:
    return _Field(name, default, "float", _parse("float_in", name, default, lo, hi))


def _enum(
    name: str, default: str, method: str, choices: Mapping[str, object] | None = None
) -> _Field:
    """Build a string field validated by a named ``*_kind`` parser method."""
    if choices is None:
        return _Field(name, default, "str", _parse(method, name, default))
    return _Field(name, default, "str", _parse(method, name, default, choices))


def _legacy_int(  # noqa: PLR0913, PLR0917 - a schema row is a row
    name: str, default: int, legacy_name: str, legacy_default: int, lo: int, hi: int
) -> _Field:
    """Build an int whose pre-rename key counts only when the user changed it."""

    def parse(p: _Parser) -> object:
        inherited = p.legacy_default(legacy_name, legacy_default, lo, hi, default)
        return p.int_in(name, inherited, lo, hi)

    return _Field(name, default, "int", parse)


def _translate_source_field() -> _Field:
    """``translate_source`` defaults to whatever the stored target implies.

    Configs written before the key existed recorded only the target, and 中→英 /
    英→中 were the only routes, so the old behaviour is exactly this inference.
    """

    def parse(p: _Parser) -> object:
        target = p.translate_target("translate_target", "en")
        inferred: TranslateSource = "en" if target == "zh" else "zh"
        return p.translate_source("translate_source", inferred)

    return _Field("translate_source", "zh", "str", parse)


#: The config schema, in file order. This is the only place a field's default and
#: its file representation are decided; ``Config`` stays hand-written because it
#: is the type surface, and ``tests/test_config.py`` pins the two together.
_SCHEMA: tuple[_Field, ...] = (
    _Field("hotkey", DEFAULT_HOTKEY, "hotkey", _parse("hotkey", "hotkey", DEFAULT_HOTKEY)),
    _int("hold_threshold_ms", 400, 10, 2_000),
    _text("mic", ""),
    _enum("audio_source", "mic", "audio_source"),
    _int("num_threads", 2, 1, 16),
    _enum("language", "zh", "choice", _LANGUAGES),
    _flag("restore_clipboard", True),
    _int("paste_delay_ms", 200, 50, 2_000),
    _flag("check_updates", True),
    _enum("model", "sensevoice", "model_kind"),
    _text("proxy", ""),
    _enum("disfluency", "verbatim", "disfluency_kind"),
    _flag("continuous", False),
    _float("vad_threshold", 0.5, 0.0, 1.0),
    _int("vad_min_speech_ms", 250, 50, 5_000),
    _legacy_int("live_min_silence_ms", 350, "vad_min_silence_ms", 500, 50, 10_000),
    _legacy_int("live_max_speech_ms", 4_000, "vad_max_speech_ms", 20_000, 1_000, 60_000),
    _int("typing_min_silence_ms", 500, 50, 10_000),
    _int("typing_max_speech_ms", 20_000, 1_000, 60_000),
    _int("typing_merge_ms", 12_000, 0, 60_000),
    _int("typing_merge_chars", 80, 10, 500),
    _flag("translate", False),
    _enum("translate_target", "en", "translate_target"),
    _translate_source_field(),
    _enum("translate_mode", "bilingual", "translate_mode"),
    _flag("overlay_enabled", False),
    _enum("overlay_output", "both", "overlay_output"),
    _enum("overlay_mode", "target", "overlay_mode"),
    _int("overlay_font_size", 15, 10, 72),
    _float("overlay_opacity", 0.85, 0.2, 1.0),
    _float("overlay_width_ratio", 0.72, 0.2, 1.0),
    _int("overlay_bottom_margin", 80, 0, 600),
    _float("overlay_hold_seconds", 8.0, 0.5, 60.0),
    _int("overlay_max_lines", 6, 1, 12),
    _flag("partial_subtitle", True),
    _int("partial_subtitle_ms", 800, 200, 5_000),
    _flag("overlay_click_through", True),
    _text("output_file", ""),
)

_FIELD_NAMES: frozenset[str] = frozenset(field.name for field in _SCHEMA)


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
    output_file: str


def default_config() -> Config:
    """First-run defaults, straight from :data:`_SCHEMA`."""
    return Config(**cast("Any", {field.name: field.default for field in _SCHEMA}))


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


def language_for_route(current_language: str, route: str) -> str:
    """Return the recogniser language that *route* implies.

    Choosing a 日→… route **must** pin ``ja``: SenseVoice decodes Japanese audio
    as Chinese otherwise, and the translator then faithfully translates the
    gibberish. Leaving a 日→… route restores a sensible default, but a user who
    deliberately chose ``auto`` keeps it.
    """
    pinned = pin_language(route)
    if pinned is not None:
        return pinned
    if current_language == "ja":
        return "auto"
    return current_language


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
    # The direction is parsed before the rest on purpose: a hand-edited file with
    # several bad values used to report a bad target/source first, and that is
    # the message people search for. Parsing in file order changed which error
    # they saw.
    target = p.translate_target("translate_target", "en")
    inferred = "en" if target == "zh" else "zh"
    source = p.translate_source("translate_source", inferred)
    values = {field.name: field.parse(p) for field in _SCHEMA}
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
    values["translate_target"] = target
    values["translate_source"] = source
    # The bar exists iff text is not typing-only: one decision, not two. A file
    # written by an older build may disagree, and convergence here is what stops
    # the contradiction from being resurrected on upgrade.
    values["overlay_enabled"] = (
        bool(values["overlay_enabled"]) and values["overlay_output"] != "type"
    )
    return Config(**cast("Any", values))


def save_config(path: Path, config: Config) -> None:
    """Persist a Config as simple key = value TOML.

    Keys this build does not know are kept: a user's own scalar note, or a
    setting written by a newer build, used to disappear the next time anything
    was saved (and ``migrate_config_file`` rewrites the file on the first start).
    Arrays, tables and datetimes have no writer here, so they are dropped with a
    warning rather than re-serialised wrongly. Renamed legacy keys are the other
    exception — they have been folded into their replacement, so keeping them
    would resurrect a value we just migrated.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        f"{field.name} = {_render(field, getattr(config, field.name))}"
        for field in _SCHEMA
    ]
    lines = [f"config_version = {CONFIG_VERSION}", *fields]
    lines.extend(_foreign_lines(path))
    payload = "\n".join(lines) + "\n"
    # Write-then-rename: `migrate_config_file` rewrites a user's config during
    # startup, and a crash between truncate and write used to leave a half-written
    # file — i.e. the next launch would refuse to start with a TOML error.
    staging = path.with_name(path.name + ".tmp")
    staging.write_text(payload, encoding="utf-8")
    staging.replace(path)


def _render(field: _Field, value: object) -> str:
    """Render one value the way the file has always spelled it."""
    if field.kind == "bool":
        return str(bool(value)).lower()
    if field.kind == "hotkey" and isinstance(value, int):
        return str(value)
    if field.kind in {"int", "float"}:
        return str(value)
    return _quote(str(value))


def _foreign_lines(path: Path) -> list[str]:
    """Re-emit keys that are not part of the schema, so a save never eats them."""
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return []
    skip = _FIELD_NAMES | {"config_version"} | set(RENAMED_KEYS)
    kept: list[tuple[str, str]] = []
    for key, value in raw.items():
        if key in skip:
            continue
        rendered = _render_foreign(value)
        if rendered is None:
            log.warning("config: %r has an unsupported value type; not rewriting it", key)
            continue
        kept.append((key, rendered))
    return [f"{key} = {rendered}" for key, rendered in kept]


def _render_foreign(value: object) -> str | None:
    """Render a non-schema TOML scalar, or None when it is not one."""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return _quote(value)
    return None


@dataclass(frozen=True, slots=True)
class MigrationReport:
    """What :func:`migrate_config_file` did, for the log line and the startup notice."""

    path: Path
    from_version: int
    to_version: int
    backup_path: Path | None
    renamed: tuple[tuple[str, str], ...]

    def describe(self) -> str:
        """One short Chinese sentence, safe to show in a tray/indicator flash."""
        changed = "、".join(f"{old}→{new}" for old, new in self.renamed) or "无键名变更"
        backup = f"，旧文件备份为 {self.backup_path.name}" if self.backup_path else ""
        return f"配置已从 v{self.from_version} 升级到 v{self.to_version}（{changed}{backup}）"


#: Zero or one report — a start migrates at most one file, and a stale report
#: from an earlier migration in the same process must not be shown twice. A
#: one-item list rather than a module global, so the writers below can mutate it
#: without ``global`` (which both ruff and basedpyright flag as a smell).
_pending_migration: list[MigrationReport] = []


def take_pending_migration() -> MigrationReport | None:
    """Pop the migration performed during this start, if any (read-once)."""
    return _pending_migration.pop() if _pending_migration else None


def migrate_config_file(path: Path) -> MigrationReport | None:
    """Bring an old config file up to :data:`CONFIG_VERSION`, backing it up first.

    Returns ``None`` when there is nothing to do — no file, already current, or
    the file does not parse. An *invalid* file is deliberately left untouched:
    rewriting it would either lose the user's mistake or silently paper over it,
    and the normal ``ConfigError`` path is what tells them what is wrong.

    The rewrite runs the parsed file through :func:`load_config` first, so every
    legacy inference (``vad_*`` → ``live_*``, translate source, overlay/output
    convergence) is applied by the same code that runs on every start, exactly
    once, and the resulting file is a plain current-schema config.
    """
    if not path.exists():
        return None
    try:
        original = path.read_text(encoding="utf-8")
        raw = tomllib.loads(original)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        log.info("config migration: skipped, cannot read %s (%s)", path, exc)
        return None
    from_version = _declared_version(raw)
    if from_version >= CONFIG_VERSION:
        return None
    try:
        config = load_config(path)
    except ConfigError as exc:
        log.warning("config migration: skipped, %s is not valid (%s)", path, exc.reason)
        return None
    renamed = tuple((old, new) for old, new in RENAMED_KEYS.items() if old in raw)
    try:
        backup = _backup_config(path, from_version, original)
        save_config(path, config)
    except OSError as exc:
        log.warning(
            "config migration: could not rewrite %s (%s); left the old file in place", path, exc
        )
        return None
    report = MigrationReport(
        path=path,
        from_version=from_version,
        to_version=CONFIG_VERSION,
        backup_path=backup,
        renamed=renamed,
    )
    _pending_migration.clear()
    _pending_migration.append(report)
    log.info("config: %s", report.describe())
    return report


def _declared_version(raw: Mapping[str, object]) -> int:
    """Return the schema version recorded in *raw*; 1 when absent (pre-stamp builds)."""
    value = raw.get("config_version", 1)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return 1
    return value


def _backup_config(path: Path, from_version: int, original: str) -> Path:
    """Write *original* beside the config as ``<name>.v<n>.bak``; never overwrite one."""
    candidate = path.with_name(f"{path.name}.v{from_version}.bak")
    suffix = 2
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.v{from_version}.{suffix}.bak")
        suffix += 1
    candidate.write_text(original, encoding="utf-8")
    return candidate


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
        # Hand back the validated value verbatim: whitelisting the names and
        # then mapping everything but "sensevoice" to "funasr_nano" silently
        # turned a saved "whisper_turbo" into Fun-ASR-Nano on the next start.
        return cast("ModelKind", value)

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
        """Parse the subtitle layout, honouring the older ``bilingual`` spelling."""
        raw = self._raw.get(name, default)
        if isinstance(raw, str) and raw in _OVERLAY_MODE_LEGACY:
            replacement = _OVERLAY_MODE_LEGACY[raw]
            log.info("config: %s = %r is the old spelling of %r", name, raw, replacement)
            return cast("OverlayMode", replacement)
        value = self.choice(name, default, _OVERLAY_MODES)
        match value:
            case "source":
                return "source"
            case "stacked":
                return "stacked"
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


#: Characters TOML basic strings cannot contain literally.
_QUOTE_ESCAPES: Mapping[str, str] = MappingProxyType(
    {
        "\\": "\\\\",
        '"': '\\"',
        "\n": "\\n",
        "\r": "\\r",
        "\t": "\\t",
        "\b": "\\b",
        "\f": "\\f",
    }
)

#: Control characters need a \u escape; everything printable is written as is.
_FIRST_CONTROL: int = 0x20
_DELETE: int = 0x7F


def _quote(value: str) -> str:
    """Quote a string for the file, escaping what a basic string forbids.

    A raw newline here used to be unreachable — no string we wrote could contain
    one. Now that keys we do not own are re-emitted, a user's multi-line string
    reaches this function, and one unescaped newline makes the whole file
    unparseable: the next launch dies with ``ConfigError`` instead of starting.
    """
    out: list[str] = []
    for char in value:
        escape = _QUOTE_ESCAPES.get(char)
        if escape is not None:
            out.append(escape)
        elif ord(char) < _FIRST_CONTROL or ord(char) == _DELETE:
            out.append(f"\\u{ord(char):04X}")
        else:
            out.append(char)
    return '"' + "".join(out) + '"'

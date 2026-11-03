"""Config file parsing, defaults, and round-trip."""

from dataclasses import fields, replace
from pathlib import Path
from typing import cast

import pytest
from app.asr import _FUNASR_NANO_PROMPTS
from app.config import (
    _SCHEMA,
    ConfigError,
    ModelKind,
    default_config,
    hotkey_vk,
    load_config,
    migrate_config_file,
    save_config,
)
from app.controls import disfluency_for_model

from app import controls, keycodes


def test_missing_file_returns_defaults(tmp_path: Path) -> None:
    config = load_config(tmp_path / "absent.toml")
    assert config.hotkey == keycodes.DEFAULT_HOTKEY
    assert config.hold_threshold_ms == 400
    assert config.mic == ""
    assert config.language == "zh"
    assert config.restore_clipboard is True
    assert config.check_updates is True
    assert config.model == "sensevoice"
    assert config.disfluency == "verbatim"
    assert config.continuous is False
    assert config.vad_threshold == 0.5
    assert config.vad_min_speech_ms == 250
    assert config.live_min_silence_ms == 350
    assert config.live_max_speech_ms == 4_000
    assert config.typing_merge_ms == 12_000
    assert config.translate is False
    assert config.translate_target == "en"
    assert config.translate_mode == "bilingual"


def test_partial_file_merges_with_defaults(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('mic = "USB Microphone"\nhold_threshold_ms = 400\n', encoding="utf-8")
    config = load_config(path)
    assert config.mic == "USB Microphone"
    assert config.hold_threshold_ms == 400
    assert config.hotkey == keycodes.DEFAULT_HOTKEY


def test_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    original = replace(
        default_config(),
        hotkey="f2",
        hold_threshold_ms=300,
        mic="Mic",
        num_threads=4,
        language="auto",
        restore_clipboard=False,
        paste_delay_ms=100,
        check_updates=False,
        model="funasr_nano",
        proxy="http://proxy:7890",
        disfluency="smooth",
        continuous=True,
        vad_threshold=0.6,
        vad_min_speech_ms=200,
        live_min_silence_ms=600,
        live_max_speech_ms=15_000,
        translate=True,
        translate_target="zh",
        translate_source="en",
        translate_mode="replace",
    )
    save_config(path, original)
    assert load_config(path) == original


def test_unknown_hotkey_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('hotkey = "volume_up"\n', encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_custom_vk_hotkey_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    original = replace(default_config(), hotkey=0x2B, hold_threshold_ms=250)
    save_config(path, original)
    assert load_config(path) == original


def test_invalid_disfluency_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('disfluency = "aggressive"\n', encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_custom_vk_hotkey_out_of_range_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("hotkey = 255\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_hotkey_vk_resolves_names_and_ints() -> None:
    default = keycodes.DEFAULT_HOTKEY
    assert hotkey_vk(default) == keycodes.PRESET_KEYCODES[default]
    assert hotkey_vk(0x2B) == 0x2B


def test_every_model_kind_survives_a_roundtrip(tmp_path: Path) -> None:
    """Regression: ``model_kind`` silently rewrote whisper_turbo to funasr_nano.

    Nothing covered this: the only round-trip used the default (sensevoice), so
    the original bug — pick Whisper, restart, quietly get Fun-ASR-Nano and leave a
    1 GB model unused on disk — reproduced with the whole suite green.
    """
    for kind in ("sensevoice", "funasr_nano", "whisper_turbo"):
        path = tmp_path / f"{kind}.toml"
        save_config(path, replace(default_config(), model=cast("ModelKind", kind)))
        assert load_config(path).model == kind


def test_japanese_route_survives_a_roundtrip(tmp_path: Path) -> None:
    """ja→zh must come back as ja→zh.

    The old test used en→zh, which happens to equal the value inferred from the
    target alone, so a save that dropped ``translate_source`` still passed.
    """
    path = tmp_path / "ja.toml"
    original = replace(
        default_config(),
        translate=True,
        translate_target="zh",
        translate_source="ja",
    )
    save_config(path, original)
    loaded = load_config(path)
    assert (loaded.translate_source, loaded.translate_target) == ("ja", "zh")


def test_invalid_language_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('language = "klingon"\n', encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_out_of_range_threshold_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("hold_threshold_ms = 0\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_malformed_toml_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("not toml at all {{{", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_funasr_nano_prompts_differ_by_disfluency() -> None:
    assert _FUNASR_NANO_PROMPTS["verbatim"] == "语音转写:"
    assert "语气填充词" in _FUNASR_NANO_PROMPTS["smooth"]
    assert _FUNASR_NANO_PROMPTS["smooth"] != _FUNASR_NANO_PROMPTS["verbatim"]


def test_disfluency_defaults_smooth_for_funasr_nano() -> None:
    assert disfluency_for_model("funasr_nano", "verbatim") == "smooth"
    assert disfluency_for_model("funasr_nano", "smooth") == "smooth"
    assert disfluency_for_model("sensevoice", "verbatim") == "verbatim"
    assert disfluency_for_model("sensevoice", "smooth") == "smooth"  # preserve user choice


def test_continuous_keys_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    original = replace(
        default_config(),
        continuous=True,
        vad_threshold=0.35,
        vad_min_speech_ms=150,
        live_min_silence_ms=800,
        live_max_speech_ms=30_000,
    )
    save_config(path, original)
    loaded = load_config(path)
    assert loaded == original
    assert loaded.continuous is True
    assert loaded.vad_threshold == 0.35
    assert loaded.vad_min_speech_ms == 150
    assert loaded.live_min_silence_ms == 800
    assert loaded.live_max_speech_ms == 30_000


def test_vad_threshold_accepts_integer(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("vad_threshold = 0\n", encoding="utf-8")
    assert load_config(path).vad_threshold == 0.0


def test_vad_threshold_rejects_non_number(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('vad_threshold = "loud"\n', encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_vad_threshold_clamped(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    for value in ("1.5", "-0.1"):
        path.write_text(f"vad_threshold = {value}\n", encoding="utf-8")
        with pytest.raises(ConfigError):
            load_config(path)


def test_vad_ms_keys_clamped(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    for key, value in (
        ("vad_min_speech_ms", 10),
        ("vad_min_speech_ms", 9_999),
        ("live_min_silence_ms", 10),
        ("live_min_silence_ms", 20_000),
        ("vad_max_speech_ms", 500),
        ("vad_max_speech_ms", 120_000),
    ):
        path.write_text(f"{key} = {value}\n", encoding="utf-8")
        with pytest.raises(ConfigError):
            load_config(path)


def test_continuous_rejects_non_bool(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('continuous = "yes"\n', encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_translate_fields_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        'translate = true\ntranslate_target = "zh"\ntranslate_mode = "replace"\n',
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.translate is True
    assert config.translate_target == "zh"
    assert config.translate_mode == "replace"


def test_invalid_translate_mode_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('translate_mode = "both"\n', encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_invalid_translate_target_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('translate_target = "fr"\n', encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


# -- the schema table ---------------------------------------------------------


def test_schema_matches_the_config_dataclass() -> None:
    """One row per field, same order: the table is the file's layout."""
    names = [field.name for field in fields(default_config())]
    assert [row.name for row in _SCHEMA] == names


def test_schema_defaults_are_the_first_run_defaults() -> None:
    """A default lives in one place; default_config() must not disagree."""
    config = default_config()
    for row in _SCHEMA:
        assert getattr(config, row.name) == row.default, row.name


def test_every_field_survives_a_save_and_load(tmp_path: Path) -> None:
    """The table drives parsing and rendering: a mismatch would lose a value."""
    path = tmp_path / "config.toml"
    original = default_config()
    save_config(path, original)

    assert load_config(path) == original


def test_unknown_keys_survive_a_save(tmp_path: Path) -> None:
    """A user's own note must not vanish because a switch was toggled."""
    path = tmp_path / "config.toml"
    path.write_text(
        'my_own_note = "keep me"\nhold_threshold_ms = 500\n', encoding="utf-8"
    )
    config = load_config(path)
    save_config(path, replace(config, hold_threshold_ms=600))

    text = path.read_text(encoding="utf-8")
    assert 'my_own_note = "keep me"' in text
    assert "hold_threshold_ms = 600" in text


def test_unknown_keys_survive_a_migration(tmp_path: Path) -> None:
    """The migration rewrites the whole file — it must carry foreign keys over."""
    path = tmp_path / "config.toml"
    path.write_text(
        'my_own_note = "keep me"\nvad_max_speech_ms = 20000\n', encoding="utf-8"
    )
    report = migrate_config_file(path)

    assert report is not None
    text = path.read_text(encoding="utf-8")
    assert 'my_own_note = "keep me"' in text
    # A renamed legacy key has been folded into live_max_speech_ms: keeping it
    # would resurrect the value the migration just replaced.
    assert "vad_max_speech_ms" not in text


def test_controls_never_writes_the_config_itself() -> None:
    """One write entry: the tray actions change the config, `_set_config` saves it.

    Two writers wrote the same file twice per tray toggle, and the second write
    could only ever agree with the first — or be the bug.
    """
    source = Path(controls.__file__).read_text(encoding="utf-8")
    assert "save_config" not in source

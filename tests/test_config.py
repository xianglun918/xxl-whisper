"""Config file parsing, defaults, and round-trip."""

from pathlib import Path

import pytest
from app.asr import _FUNASR_NANO_PROMPTS
from app.config import Config, ConfigError, hotkey_vk, load_config, save_config
from app.controls import disfluency_for_model

from app import native


def test_missing_file_returns_defaults(tmp_path: Path) -> None:
    config = load_config(tmp_path / "absent.toml")
    assert config.hotkey == native.hotkey.DEFAULT_HOTKEY
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
    assert config.vad_min_silence_ms == 500
    assert config.vad_max_speech_ms == 20_000


def test_partial_file_merges_with_defaults(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('mic = "USB Microphone"\nhold_threshold_ms = 400\n', encoding="utf-8")
    config = load_config(path)
    assert config.mic == "USB Microphone"
    assert config.hold_threshold_ms == 400
    assert config.hotkey == native.hotkey.DEFAULT_HOTKEY


def test_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    original = Config(hotkey="f2", hold_threshold_ms=300, mic="Mic", num_threads=4,
                      language="auto", restore_clipboard=False, paste_delay_ms=100,
                      check_updates=False, model="funasr_nano", proxy="http://proxy:7890",
                      disfluency="smooth", continuous=True, vad_threshold=0.6,
                      vad_min_speech_ms=200, vad_min_silence_ms=600, vad_max_speech_ms=15_000)
    save_config(path, original)
    assert load_config(path) == original


def test_unknown_hotkey_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('hotkey = "volume_up"\n', encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_custom_vk_hotkey_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    original = Config(hotkey=0x2B, hold_threshold_ms=250, mic="", num_threads=2,
                      language="zh", restore_clipboard=True, paste_delay_ms=200,
                      check_updates=True, model="sensevoice", proxy="", disfluency="verbatim",
                      continuous=False, vad_threshold=0.5, vad_min_speech_ms=250,
                      vad_min_silence_ms=500, vad_max_speech_ms=20_000)
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
    default = native.hotkey.DEFAULT_HOTKEY
    assert hotkey_vk(default) == native.hotkey.PRESET_KEYCODES[default]
    assert hotkey_vk(0x2B) == 0x2B


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
    original = Config(hotkey=native.hotkey.DEFAULT_HOTKEY, hold_threshold_ms=400, mic="",
                      num_threads=2, language="zh", restore_clipboard=True, paste_delay_ms=200,
                      check_updates=True, model="sensevoice", proxy="", disfluency="verbatim",
                      continuous=True, vad_threshold=0.35, vad_min_speech_ms=150,
                      vad_min_silence_ms=800, vad_max_speech_ms=30_000)
    save_config(path, original)
    loaded = load_config(path)
    assert loaded == original
    assert loaded.continuous is True
    assert loaded.vad_threshold == 0.35
    assert loaded.vad_min_speech_ms == 150
    assert loaded.vad_min_silence_ms == 800
    assert loaded.vad_max_speech_ms == 30_000


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
        ("vad_min_silence_ms", 10),
        ("vad_min_silence_ms", 20_000),
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

"""Config file parsing, defaults, and round-trip."""

from pathlib import Path

import pytest
from app.config import Config, ConfigError, hotkey_vk, load_config, save_config


def test_missing_file_returns_defaults(tmp_path: Path) -> None:
    config = load_config(tmp_path / "absent.toml")
    assert config.hotkey == "caps_lock"
    assert config.hold_threshold_ms == 400
    assert config.mic == ""
    assert config.language == "zh"
    assert config.restore_clipboard is True
    assert config.check_updates is True
    assert config.model == "sensevoice"


def test_partial_file_merges_with_defaults(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('mic = "USB Microphone"\nhold_threshold_ms = 400\n', encoding="utf-8")
    config = load_config(path)
    assert config.mic == "USB Microphone"
    assert config.hold_threshold_ms == 400
    assert config.hotkey == "caps_lock"


def test_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    original = Config(hotkey="f2", hold_threshold_ms=300, mic="Mic", num_threads=4,
                      language="auto", restore_clipboard=False, paste_delay_ms=100,
                      check_updates=False, model="funasr_nano")
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
                      check_updates=True, model="sensevoice")
    save_config(path, original)
    assert load_config(path) == original


def test_custom_vk_hotkey_out_of_range_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("hotkey = 255\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_hotkey_vk_resolves_names_and_ints() -> None:
    assert hotkey_vk("caps_lock") == 0x14
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

"""Which VAD limits a continuous session uses (subtitle mode vs plain typing).

The rule the user cares about: force-splitting a long monologue is a *subtitle*
feature. Holding the push-to-talk key must never be cut — and indeed the hold
path never builds a Segmenter at all — while plain continuous typing keeps the
looser defaults so a long dictation stays one sentence.
"""

import importlib
import sys
import types
from pathlib import Path

import pytest


def _config_module(tmp_path: Path):
    """Import app.config with the platform layer stubbed out."""
    fake_native = types.ModuleType("app.native")
    fake_native.hotkey = types.SimpleNamespace(  # type: ignore[attr-defined]
        PRESET_KEYCODES={"f2": 0x71}, MOUSE_KEYCODES=frozenset(), DEFAULT_HOTKEY="f2"
    )
    fake_native.data_root = lambda: tmp_path  # type: ignore[attr-defined]
    sys.modules["app.native"] = fake_native
    sys.modules.pop("app.config", None)
    return importlib.import_module("app.config")


def test_subtitle_mode_cuts_early(tmp_path: Path) -> None:
    config_mod = _config_module(tmp_path)
    config = config_mod.default_config()
    assert config_mod.segmenter_limits(config, subtitles=True) == (400, 6_000)


def test_plain_continuous_typing_keeps_the_loose_limits(tmp_path: Path) -> None:
    """A long dictation must not be chopped into 6 s pieces."""
    config_mod = _config_module(tmp_path)
    config = config_mod.default_config()
    assert config_mod.segmenter_limits(config, subtitles=False) == (500, 20_000)


def test_limits_follow_the_user_settings(tmp_path: Path) -> None:
    config_mod = _config_module(tmp_path)
    path = tmp_path / "config.toml"
    path.write_text(
        "subtitle_min_silence_ms = 250\nsubtitle_max_speech_ms = 3000\n"
        "vad_min_silence_ms = 800\nvad_max_speech_ms = 30000\n",
        encoding="utf-8",
    )
    config = config_mod.load_config(path)
    assert config_mod.segmenter_limits(config, subtitles=True) == (250, 3_000)
    assert config_mod.segmenter_limits(config, subtitles=False) == (800, 30_000)


def test_subtitle_limits_are_clamped(tmp_path: Path) -> None:
    config_mod = _config_module(tmp_path)
    path = tmp_path / "config.toml"
    path.write_text("subtitle_max_speech_ms = 100\n", encoding="utf-8")
    with pytest.raises(config_mod.ConfigError):
        config_mod.load_config(path)

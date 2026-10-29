"""Merging force-cut ASR pieces before typing, and the config that drives it.

The behaviour under test is the one the two-channels design rests on: a segment
that ended on silence is a finished sentence and is typed immediately, while a
segment cut by the max-speech cap means the speaker is still going and must be
joined with the next piece instead of being pasted as a fragment.
"""

import importlib
import sys
import types
from pathlib import Path

import pytest
from app.type_batcher import TypeBatcher


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _batcher(**kwargs: object) -> tuple[TypeBatcher, FakeClock]:
    clock = FakeClock()
    return TypeBatcher(clock=clock, **kwargs), clock  # type: ignore[arg-type]


def test_sentence_ending_on_silence_is_typed_immediately() -> None:
    batcher, _clock = _batcher()
    assert batcher.add("今天天气不错。", forced=False) == "今天天气不错。"
    assert len(batcher) == 0


def test_force_cut_piece_waits_for_the_rest() -> None:
    """A max-speech cut means the sentence is not over: never paste a fragment."""
    batcher, _clock = _batcher()
    assert batcher.add("我们先把接口定下来", forced=True) is None
    assert len(batcher) == 1
    assert batcher.add("然后再写测试。", forced=False) == "我们先把接口定下来然后再写测试。"


def test_two_force_cuts_join_without_a_space_in_chinese() -> None:
    batcher, _clock = _batcher()
    batcher.add("第一段", forced=True)
    assert batcher.add("第二段", forced=False) == "第一段第二段"


def test_latin_pieces_keep_a_space() -> None:
    batcher, _clock = _batcher()
    batcher.add("we should finish", forced=True)
    assert batcher.add("this today.", forced=False) == "we should finish this today."


def test_character_budget_closes_a_long_force_cut_run() -> None:
    batcher, _clock = _batcher(max_chars=10)
    batcher.add("一二三四五", forced=True)
    out = batcher.add("六七八九十", forced=True)
    assert out == "一二三四五六七八九十"
    assert len(batcher) == 0


def test_window_flushes_a_still_open_batch() -> None:
    batcher, clock = _batcher(window_s=12.0)
    batcher.add("说到这里先停一下", forced=True)
    assert batcher.should_flush() is False
    clock.advance(12.0)
    assert batcher.should_flush() is True
    assert batcher.flush() == "说到这里先停一下"
    assert batcher.should_flush() is False  # nothing left


def test_flush_on_empty_batch_is_a_no_op() -> None:
    batcher, _clock = _batcher()
    assert batcher.flush() is None


def test_blank_pieces_are_ignored() -> None:
    batcher, _clock = _batcher()
    assert batcher.add("   ", forced=False) is None
    assert len(batcher) == 0


def test_pending_shows_what_would_be_typed() -> None:
    batcher, _clock = _batcher()
    batcher.add("半句话", forced=True)
    assert batcher.pending == "半句话"
    assert batcher.pending_chars == 3
    assert len(batcher) == 1


# -- config side ---------------------------------------------------------------


def _config_module(tmp_path: Path):
    fake_native = types.ModuleType("app.native")
    fake_native.hotkey = types.SimpleNamespace(  # type: ignore[attr-defined]
        PRESET_KEYCODES={"f2": 0x71}, MOUSE_KEYCODES=frozenset(), DEFAULT_HOTKEY="f2"
    )
    fake_native.data_root = lambda: tmp_path  # type: ignore[attr-defined]
    sys.modules["app.native"] = fake_native
    sys.modules.pop("app.config", None)
    return importlib.import_module("app.config")


def test_live_defaults_are_subtitle_friendly(tmp_path: Path) -> None:
    config_mod = _config_module(tmp_path)
    config = config_mod.default_config()
    assert (config.live_min_silence_ms, config.live_max_speech_ms) == (400, 6_000)
    assert (config.typing_merge_ms, config.typing_merge_chars) == (12_000, 80)


def test_old_vad_keys_still_work_as_a_fallback(tmp_path: Path) -> None:
    """A config written before ``live_*`` existed keeps its tuned values."""
    config_mod = _config_module(tmp_path)
    path = tmp_path / "config.toml"
    path.write_text(
        "vad_min_silence_ms = 800\nvad_max_speech_ms = 30000\n", encoding="utf-8"
    )
    config = config_mod.load_config(path)
    assert (config.live_min_silence_ms, config.live_max_speech_ms) == (800, 30_000)


def test_live_keys_win_over_the_legacy_fallback(tmp_path: Path) -> None:
    config_mod = _config_module(tmp_path)
    path = tmp_path / "config.toml"
    path.write_text(
        "vad_min_silence_ms = 800\nvad_max_speech_ms = 30000\n"
        "live_min_silence_ms = 300\nlive_max_speech_ms = 5000\n",
        encoding="utf-8",
    )
    config = config_mod.load_config(path)
    assert (config.live_min_silence_ms, config.live_max_speech_ms) == (300, 5_000)


def test_merge_budget_is_validated(tmp_path: Path) -> None:
    config_mod = _config_module(tmp_path)
    path = tmp_path / "config.toml"
    path.write_text("typing_merge_chars = 5\n", encoding="utf-8")
    with pytest.raises(config_mod.ConfigError):
        config_mod.load_config(path)

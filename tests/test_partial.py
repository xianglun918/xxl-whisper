"""Pure unit tests for live partial-transcription helpers (no model, no I/O)."""

import numpy as np
import pytest
from app.partial import (
    PARTIAL_DISPLAY_CHARS,
    SubtitleGate,
    level_from_block,
    new_text_or_none,
    truncate_partial,
)


def test_truncate_keeps_short_text_verbatim() -> None:
    assert truncate_partial("你好") == "你好"


def test_truncate_keeps_text_at_the_limit() -> None:
    text = "a" * PARTIAL_DISPLAY_CHARS
    assert truncate_partial(text) == text


def test_truncate_keeps_tail_with_leading_ellipsis() -> None:
    text = "0123456789" * 3  # 30 chars
    assert truncate_partial(text) == "…" + text[-PARTIAL_DISPLAY_CHARS:]


def test_truncate_honours_custom_limit() -> None:
    assert truncate_partial("abcdef", limit=2) == "…ef"


def test_new_text_or_none_skips_an_unchanged_caption() -> None:
    assert new_text_or_none("你好", "你好") is None


def test_new_text_or_none_returns_a_changed_caption() -> None:
    assert new_text_or_none("你好", "你好世界") == "你好世界"


def test_level_from_block_is_zero_for_silence_and_empty() -> None:
    assert level_from_block(np.zeros(8, dtype=np.float32)) == 0.0
    assert level_from_block(np.zeros(0, dtype=np.float32)) == 0.0


def test_level_from_block_scales_quiet_speech() -> None:
    assert level_from_block(np.full(8, 0.125, dtype=np.float32)) == pytest.approx(0.5)


def test_level_from_block_saturates_at_one() -> None:
    assert level_from_block(np.full(8, 0.5, dtype=np.float32)) == 1.0


# -- the subtitle refresh gate ------------------------------------------------


def test_gate_shows_the_first_partial() -> None:
    gate = SubtitleGate(min_interval_s=0.8, min_growth=2)
    assert gate.offer("今天我们来聊", 1.0) == "今天我们来聊"


def test_gate_skips_identical_text() -> None:
    gate = SubtitleGate(min_interval_s=0.0, min_growth=0)
    gate.offer("同一句", 1.0)
    assert gate.offer("同一句", 2.0) is None


def test_gate_skips_a_bare_punctuation_refresh() -> None:
    """One extra character is not worth a translation pass or a flicker."""
    gate = SubtitleGate(min_interval_s=0.0, min_growth=2)
    gate.offer("今天我们来聊", 1.0)
    assert gate.offer("今天我们来聊啊", 2.0) is None


def test_gate_allows_enough_growth() -> None:
    gate = SubtitleGate(min_interval_s=0.0, min_growth=2)
    gate.offer("今天", 1.0)
    assert gate.offer("今天我们来聊一聊", 2.0) == "今天我们来聊一聊"


def test_gate_respects_the_interval() -> None:
    gate = SubtitleGate(min_interval_s=0.8, min_growth=2)
    gate.offer("第一句", 10.0)
    assert gate.offer("第一句加上更多字", 10.5) is None   # too soon
    assert gate.offer("第一句加上更多字", 10.9) is not None  # interval elapsed


def test_gate_still_rate_limits_a_rewrite() -> None:
    """A rewrite costs a translation pass like any other refresh: interval applies."""
    gate = SubtitleGate(min_interval_s=0.8, min_growth=2)
    gate.offer("今天我们", 5.0)
    assert gate.offer("换个说法完全不同", 5.1) is None
    assert gate.offer("换个说法完全不同", 6.0) == "换个说法完全不同"


def test_gate_ignores_empty_decodes() -> None:
    gate = SubtitleGate(min_interval_s=0.0)
    assert gate.offer("   ", 1.0) is None


def test_gate_reset_lets_the_next_sentence_through_at_once() -> None:
    gate = SubtitleGate(min_interval_s=0.8, min_growth=2)
    gate.offer("这一句", 1.0)
    gate.reset()
    assert gate.offer("下一句", 1.1) == "下一句"

"""Pure unit tests for live partial-transcription helpers (no model, no I/O)."""

import numpy as np
import pytest
from app.partial import (
    PARTIAL_DISPLAY_CHARS,
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

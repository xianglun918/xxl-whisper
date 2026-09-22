"""Pure unit tests for live partial-transcription helpers (no model, no I/O)."""

import numpy as np
import pytest
from app.partial import (
    PARTIAL_DISPLAY_CHARS,
    PartialBuffer,
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


def test_empty_buffer_has_no_samples() -> None:
    assert PartialBuffer().samples() is None


def test_push_accumulates_in_order() -> None:
    buffer = PartialBuffer()
    buffer = buffer.push(np.array([1.0, 2.0], dtype=np.float32))
    buffer = buffer.push(np.array([3.0], dtype=np.float32))
    samples = buffer.samples()
    assert samples is not None
    assert np.array_equal(samples, np.array([1.0, 2.0, 3.0], dtype=np.float32))


def test_push_flattens_a_column_block() -> None:
    samples = PartialBuffer().push(np.zeros((4, 1), dtype=np.float32)).samples()
    assert samples is not None
    assert samples.shape == (4,)


def test_push_is_immutable() -> None:
    original = PartialBuffer().push(np.ones(2, dtype=np.float32))
    grown = original.push(np.zeros(2, dtype=np.float32))
    original_samples = original.samples()
    grown_samples = grown.samples()
    assert original_samples is not None
    assert grown_samples is not None
    assert original_samples.shape == (2,)  # the original is untouched
    assert grown_samples.shape == (4,)


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

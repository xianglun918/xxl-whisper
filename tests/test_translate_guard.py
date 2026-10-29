"""The MT degeneration guard.

One of eight test sentences (`いただきます。`) made the model emit
``给你吃。。」」」等。」等 喔。」.。。。``. An empty string keeps the source sentence on
screen, which is always better than that.
"""

from __future__ import annotations

from app.translate_guard import looks_degenerate, looks_truncated


def test_real_collapse_from_the_field_is_caught() -> None:
    assert looks_degenerate("给你吃。。」」」等。」等 喔。」.。。。")


def test_normal_chinese_passes() -> None:
    for text in (
        "你好,今天天气不错吧。",
        "他生活在东京。",
        "各位客户,请大家稍等,请稍等。",
        "这事,你得先告诉你的老板,然后再讨论。",
    ):
        assert not looks_degenerate(text), text


def test_decode_that_never_ends_counts_as_collapsed() -> None:
    """The field failure had no repeated n-gram; it just never emitted EOS."""
    assert looks_truncated(64, 64)
    assert looks_truncated(65, 64)
    assert not looks_truncated(12, 64)


def test_empty_and_short_are_not_flagged() -> None:
    assert not looks_degenerate("")
    assert not looks_degenerate("是。")
    assert not looks_degenerate("谢谢")


def test_ellipsis_and_quotes_alone_are_fine() -> None:
    """Legitimate punctuation must not trip the guard."""
    assert not looks_degenerate("他说：“明天再来。”")
    assert not looks_degenerate("等一下……")

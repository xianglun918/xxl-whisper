"""Hold/click discrimination for the push-to-talk hotkey."""

from app.hotkey_logic import (
    Click,
    EndHold,
    HoldClickDetector,
    Press,
    Release,
    StartHold,
    loading_hint_visible,
    push_to_talk_active,
)


def test_press_starts_hold() -> None:
    detector = HoldClickDetector(threshold_ms=250)
    result = detector.feed(Press(timestamp_ms=1_000))
    assert result == StartHold()


def test_quick_release_is_click() -> None:
    detector = HoldClickDetector(threshold_ms=250)
    detector.feed(Press(timestamp_ms=1_000))
    result = detector.feed(Release(timestamp_ms=1_100))
    assert result == Click()


def test_long_release_ends_hold() -> None:
    detector = HoldClickDetector(threshold_ms=250)
    detector.feed(Press(timestamp_ms=1_000))
    result = detector.feed(Release(timestamp_ms=3_500))
    assert result == EndHold(duration_ms=2_500)


def test_release_exactly_at_threshold_is_hold() -> None:
    detector = HoldClickDetector(threshold_ms=250)
    detector.feed(Press(timestamp_ms=1_000))
    result = detector.feed(Release(timestamp_ms=1_250))
    assert result == EndHold(duration_ms=250)


def test_auto_repeat_press_is_ignored() -> None:
    detector = HoldClickDetector(threshold_ms=250)
    detector.feed(Press(timestamp_ms=1_000))
    assert detector.feed(Press(timestamp_ms=1_400)) is None
    # The real release still resolves against the FIRST press.
    assert detector.feed(Release(timestamp_ms=2_000)) == EndHold(duration_ms=1_000)


def test_release_without_press_is_ignored() -> None:
    detector = HoldClickDetector(threshold_ms=250)
    assert detector.feed(Release(timestamp_ms=1_000)) is None


def test_cycle_resets_state() -> None:
    detector = HoldClickDetector(threshold_ms=250)
    detector.feed(Press(timestamp_ms=1_000))
    detector.feed(Release(timestamp_ms=1_100))  # Click
    assert detector.feed(Press(timestamp_ms=2_000)) == StartHold()
    assert detector.feed(Release(timestamp_ms=5_000)) == EndHold(duration_ms=3_000)


def test_push_to_talk_is_inert_while_continuous_owns_the_mic() -> None:
    """The two modes are mutually exclusive: a press must not start a recording.

    The key stays native on purpose — CapsLock must still toggle caps — so only
    the "hold to talk" behaviour is dropped.
    """
    assert push_to_talk_active(continuous=False) is True
    assert push_to_talk_active(continuous=True) is False


def test_loading_hint_is_silent_during_continuous() -> None:
    """Nothing is loading while continuous runs, and hiding it would wipe the pill."""
    assert loading_hint_visible(continuous=False) is True
    assert loading_hint_visible(continuous=True) is False

"""Menu callbacks: never raise into the platform loop, never refresh too early.

Both rules come from a real regression: the menu rebuilt itself inside the very
action pystray was invoking, which wedged the menu thread so every later click
did nothing (the "开关全都不生效" report).
"""

from __future__ import annotations

import threading
import time

from app.menu_refresh import DEFAULT_REFRESH_DELAY_S, guarded, schedule_refresh


def test_guarded_calls_the_action() -> None:
    calls: list[str] = []
    wrapper = guarded(lambda: calls.append("hit"), lambda _reason: None)
    assert wrapper is not None
    wrapper()
    assert calls == ["hit"]


def test_guarded_absorbs_pystray_arguments() -> None:
    """pystray may pass (icon, item), (icon) or nothing, depending on backend."""
    calls: list[str] = []
    wrapper = guarded(lambda: calls.append("hit"), lambda _reason: None)
    assert wrapper is not None
    wrapper("icon")
    wrapper("icon", "item")
    wrapper()
    assert calls == ["hit", "hit", "hit"]


def test_guarded_swallows_exceptions_and_reports() -> None:
    reasons: list[str] = []

    def boom() -> None:
        reason = "kaboom"
        raise RuntimeError(reason)

    wrapper = guarded(boom, reasons.append)
    assert wrapper is not None
    wrapper()  # must not raise
    assert reasons and "kaboom" in reasons[0]


def test_guarded_none_stays_none() -> None:
    assert guarded(None, lambda _reason: None) is None


def test_schedule_refresh_runs_after_the_delay() -> None:
    done = threading.Event()
    schedule_refresh(done.set, lambda _reason: None, delay=0.01)
    assert done.wait(timeout=2.0), "refresh never ran"


def test_schedule_refresh_never_raises() -> None:
    reasons: list[str] = []

    def boom() -> None:
        reason = "refresh boom"
        raise RuntimeError(reason)

    schedule_refresh(boom, reasons.append, delay=0.01)
    for _ in range(200):
        if reasons:
            break
        time.sleep(0.01)
    assert reasons and "refresh boom" in reasons[0]


def test_default_delay_is_not_zero() -> None:
    """Zero delay is exactly the re-entrancy bug this guards against."""
    assert DEFAULT_REFRESH_DELAY_S >= 0.05


def test_custom_timer_factory_is_used() -> None:
    created: list[float] = []

    class FakeTimer:
        def __init__(self, delay: float, action) -> None:
            created.append(delay)
            self.daemon = False
            self._action = action

        def start(self) -> None:
            return None

    schedule_refresh(lambda: None, lambda _reason: None, delay=0.5, timer_factory=FakeTimer)
    assert created == [0.5]

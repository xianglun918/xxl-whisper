"""Pure unit tests for the continuous decode scheduling helpers.

The continuous mode decouples a real-time VAD consumer from a slower decoder so
a heavy model cannot stall endpointing. The priority decision and the adaptive
partial cadence are pure functions, and the latest-snapshot holder is the only
piece of mutable state — all three are pinned here without threads or a model.
"""

import numpy as np
from app.decode_scheduler import (
    DecodeKind,
    FinishedBatch,
    LatestSlot,
    next_decode,
    partial_interval,
)

_BASE_S = 0.35


def test_next_decode_prefers_a_finished_sentence_over_a_due_partial() -> None:
    assert next_decode(has_finished=True, partial_due=True) is DecodeKind.FINISHED


def test_next_decode_picks_a_partial_when_due_and_nothing_is_finished() -> None:
    assert next_decode(has_finished=False, partial_due=True) is DecodeKind.PARTIAL


def test_next_decode_idles_when_nothing_is_due() -> None:
    assert next_decode(has_finished=False, partial_due=False) is DecodeKind.IDLE


def test_partial_interval_keeps_the_base_before_any_decode() -> None:
    assert partial_interval(_BASE_S, None) == _BASE_S


def test_partial_interval_keeps_the_base_for_a_fast_decode() -> None:
    assert partial_interval(_BASE_S, 0.1) == _BASE_S


def test_partial_interval_keeps_the_base_at_the_threshold() -> None:
    assert partial_interval(_BASE_S, _BASE_S) == _BASE_S


def test_partial_interval_backs_off_for_a_slow_decode() -> None:
    assert partial_interval(_BASE_S, 1.0) == 2.0


def test_partial_interval_caps_the_backoff() -> None:
    assert partial_interval(_BASE_S, 3.0) == 4.0


def test_latest_slot_is_empty_before_any_publish() -> None:
    assert LatestSlot[int]().take() is None


def test_latest_slot_publish_then_take_returns_the_value() -> None:
    slot: LatestSlot[int] = LatestSlot()
    slot.publish(7)
    assert slot.take() == 7


def test_latest_slot_take_clears_the_slot() -> None:
    slot: LatestSlot[int] = LatestSlot()
    slot.publish(7)
    assert slot.take() == 7
    assert slot.take() is None


def test_latest_slot_publish_reports_a_dropped_stale_value() -> None:
    slot: LatestSlot[int] = LatestSlot()
    assert slot.publish(1) is False  # nothing was pending
    assert slot.publish(2) is True  # the pending 1 is superseded
    assert slot.take() == 2  # only the newest survives


def test_latest_slot_clear_drops_the_pending_value() -> None:
    slot: LatestSlot[int] = LatestSlot()
    slot.publish(1)
    slot.clear()
    assert slot.take() is None


def test_finished_batch_carries_the_endpoint_time_and_segments() -> None:
    segment = np.ones(8, dtype=np.float32)
    batch = FinishedBatch(endpoint_ts=12.5, segments=(segment,))
    assert batch.endpoint_ts == 12.5
    assert batch.segments == (segment,)

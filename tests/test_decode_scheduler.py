"""Pure unit tests for the continuous decode scheduling helpers.

The continuous mode decouples a real-time VAD consumer from two decoders (a
finished-sentence one that always wins and a live-partial one on its own
thread). The adaptive cadence, the utterance-length ceiling and the publish
gate are pure or trivially testable, and the latest-snapshot holder is the only
piece of mutable state — all are pinned here without threads or a model.
"""

import numpy as np
from app.decode_scheduler import (
    FinishedBatch,
    LatestSlot,
    PartialCadence,
    partial_interval,
    within_partial_ceiling,
)

_BASE_S = 0.35


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


def test_within_partial_ceiling_allows_short_and_boundary_utterances() -> None:
    assert within_partial_ceiling(0.0)
    assert within_partial_ceiling(6.0)  # the default ceiling is inclusive


def test_within_partial_ceiling_rejects_a_long_utterance() -> None:
    assert not within_partial_ceiling(6.001)


def test_within_partial_ceiling_honours_a_custom_ceiling() -> None:
    assert within_partial_ceiling(10.0, max_s=20.0)
    assert not within_partial_ceiling(10.0, max_s=5.0)


def test_cadence_publishes_when_nothing_has_run_yet() -> None:
    cadence = PartialCadence(_BASE_S)
    assert cadence.should_publish(100.0, paused=False) is True


def test_cadence_never_publishes_while_paused() -> None:
    cadence = PartialCadence(_BASE_S)
    assert cadence.should_publish(100.0, paused=True) is False


def test_cadence_blocks_a_publish_while_a_decode_is_in_flight() -> None:
    cadence = PartialCadence(_BASE_S)
    cadence.mark_start(100.0)
    assert cadence.should_publish(100.5, paused=False) is False


def test_cadence_waits_out_the_base_gap_after_a_fast_decode() -> None:
    cadence = PartialCadence(_BASE_S)
    cadence.mark_start(100.0)
    cadence.mark_end(0.1)
    assert cadence.should_publish(100.1, paused=False) is False
    assert cadence.should_publish(100.4, paused=False) is True


def test_cadence_backs_off_proportionally_to_a_slow_decode() -> None:
    cadence = PartialCadence(_BASE_S)
    cadence.mark_start(100.0)
    cadence.mark_end(1.0)  # 1.0 s > base -> interval stretches to 2.0 s
    assert cadence.should_publish(101.9, paused=False) is False
    assert cadence.should_publish(102.0, paused=False) is True


def test_cadence_reset_restarts_the_gap_and_bumps_the_generation() -> None:
    cadence = PartialCadence(_BASE_S)
    cadence.mark_start(100.0)
    cadence.mark_end(1.0)
    generation = cadence.generation()
    cadence.reset(200.0)
    assert cadence.generation() == generation + 1
    assert cadence.should_publish(200.1, paused=False) is False
    assert cadence.should_publish(200.4, paused=False) is True


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


def test_latest_slot_has_value_tracks_pending_state() -> None:
    slot: LatestSlot[int] = LatestSlot()
    assert slot.has_value() is False
    slot.publish(1)
    assert slot.has_value() is True
    slot.take()
    assert slot.has_value() is False


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

"""Audio-flow recovery policy for continuous dictation.

The bug this covers: with 持续听写 left ON at exit, the next launch showed the
switch as ON but transcribed nothing until it was toggled off and on again. The
stream can fail to open while the audio endpoint is still initialising, and a
dead stream is indistinguishable from a working one without a proof-of-life
counter.
"""

from __future__ import annotations

from app.audio_flow import MAX_REOPENS, plan_recovery


def test_any_delivered_block_counts_as_alive() -> None:
    assert plan_recovery(blocks=11, baseline=10, reopens_done=0) == "ok"
    assert plan_recovery(blocks=11, baseline=10, reopens_done=MAX_REOPENS) == "ok"


def test_silence_triggers_one_reopen_then_gives_up() -> None:
    assert plan_recovery(blocks=10, baseline=10, reopens_done=0) == "reopen"
    assert plan_recovery(blocks=10, baseline=10, reopens_done=MAX_REOPENS) == "give-up"


def test_give_up_is_terminal_so_the_switch_stops_lying() -> None:
    """After the reopen budget is spent the caller turns the mode off."""
    verdicts = [
        plan_recovery(blocks=10, baseline=10, reopens_done=n)
        for n in range(MAX_REOPENS + 1)
    ]
    assert verdicts[-1] == "give-up"
    assert "ok" not in verdicts

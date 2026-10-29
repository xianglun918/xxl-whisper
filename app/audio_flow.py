"""Policy for "the continuous stream never delivered a block".

Kept as pure data-in/verdict-out so the rule can be tested without audio hardware:
the recovery attempt is the part that was silently missing before, which is why a
restart with continuous dictation enabled looked healthy but transcribed nothing.
"""

from __future__ import annotations

from typing import Literal

Verdict = Literal["ok", "reopen", "give-up"]

#: How many times the stream may be reopened before giving up on it.
MAX_REOPENS: int = 1


def plan_recovery(*, blocks: int, baseline: int, reopens_done: int) -> Verdict:
    """Decide what to do when no audio has arrived since *baseline*.

    ``blocks`` is the recorder's monotonic delivered-block counter, so any
    increase means the stream is alive — even if the VAD found no speech yet.
    """
    if blocks != baseline:
        return "ok"
    if reopens_done < MAX_REOPENS:
        return "reopen"
    return "give-up"

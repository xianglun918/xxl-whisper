"""Live partial-transcription helpers: trim the caption, gate its updates.

The VAD hands over the in-progress utterance's own audio (see
:meth:`app.vad.Segmenter.current_samples`), so the continuous loop needs no
separate audio accumulator; these pure helpers only shape the decoded text and
the pill's breath. All are pure (numpy/text only): the continuous loop stays the
only place that touches the recognizer and the indicator.
"""

from __future__ import annotations

import numpy as np

#: How many trailing characters of a long partial to keep on the pill.
PARTIAL_DISPLAY_CHARS: int = 20
_ELLIPSIS: str = "…"

#: Block RMS that maps to a full-brightness breath; speech is typically
#: ~0.05-0.30 RMS, so 0.25 puts normal speech near the top of the range.
_LEVEL_FULL_SCALE_RMS: float = 0.25


def truncate_partial(text: str, limit: int = PARTIAL_DISPLAY_CHARS) -> str:
    """Keep the tail of a live partial so the pill stays small.

    Returns *text* unchanged when it fits; otherwise the last *limit*
    characters prefixed with an ellipsis.
    """
    if len(text) <= limit:
        return text
    return f"{_ELLIPSIS}{text[-limit:]}"


def new_text_or_none(previous: str, current: str) -> str | None:
    """Return *current* only when it differs from *previous*, else ``None``.

    Lets the continuous loop skip redundant indicator updates, so the pill
    caption never flickers while a partial decode returns the same text.
    """
    return current if current != previous else None


def level_from_block(block: np.ndarray) -> float:
    """Map one audio block to a 0..1 breath level (RMS, full scale ~0.25).

    Used to make the listening pill breathe with the voice; silence (or an
    empty block) maps to 0.0.
    """
    if block.size == 0:
        return 0.0
    rms = float(np.sqrt(np.mean(np.square(block, dtype=np.float32))))
    return min(1.0, rms / _LEVEL_FULL_SCALE_RMS)

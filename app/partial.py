"""Live partial-transcription helpers: rebuild in-progress audio, trim the text.

The VAD only hands over a finished segment at the endpoint, so the live
"partial" display must rebuild the utterance from the raw blocks fed while
speech is active. Both helpers are pure (numpy/text only): the continuous loop
stays the only place that touches the recognizer and the indicator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: How many trailing characters of a long partial to keep on the pill.
PARTIAL_DISPLAY_CHARS: int = 20
_ELLIPSIS: str = "…"


@dataclass(frozen=True, slots=True)
class PartialBuffer:
    """Immutable accumulator of the raw blocks spoken so far.

    ``push`` returns a new buffer (the caller rebinds), ``samples`` returns the
    buffered audio concatenated (``None`` when empty). Immutability keeps it
    trivially testable and lets the loop hold a plain local binding.
    """

    _blocks: tuple[np.ndarray, ...] = ()

    def push(self, block: np.ndarray) -> PartialBuffer:
        """Return a new buffer with *block* (flattened float32) appended."""
        flat = block.reshape(-1).astype(np.float32, copy=False)
        return PartialBuffer((*self._blocks, flat))

    def samples(self) -> np.ndarray | None:
        """All buffered samples concatenated, or ``None`` when empty."""
        if not self._blocks:
            return None
        return np.concatenate(self._blocks)


def truncate_partial(text: str, limit: int = PARTIAL_DISPLAY_CHARS) -> str:
    """Keep the tail of a live partial so the pill stays small.

    Returns *text* unchanged when it fits; otherwise the last *limit*
    characters prefixed with an ellipsis.
    """
    if len(text) <= limit:
        return text
    return f"{_ELLIPSIS}{text[-limit:]}"

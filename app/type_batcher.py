"""Merging ASR pieces back into whole sentences before typing.

Subtitle mode is allowed to cut a monologue every few seconds, but the typed
text must not be shredded: pasting twice a second breaks IME state, floods the
undo stack and leaves fragments in the document. So the two channels get
different cadences from the *same* transcript:

* subtitle: shown as soon as the piece is translated (see ``app.overlay``);
* typing: accumulated here and pasted once, as one sentence.

The signal used to decide is free: the VAD already tells us why a segment ended.
A segment closed by **silence** is a finished sentence — type it now. A segment
cut by the **max_speech cap** means the speaker is still going — keep collecting
until a natural end, a character budget, or a time cap.

:class:`TypeBatcher` is pure logic (injected clock, no audio, no Tk), so the
merge rules are unit-tested rather than discovered on a real machine.
"""

import time
from collections.abc import Callable

__all__ = ["TypeBatcher"]

#: Characters after which a piece is glued on without a space (CJK text has no
#: word spaces, and an inserted space between two Chinese characters looks wrong).
_CJK_PUNCT: str = "。！？，、；：…）】」』》"
_SENTENCE_END: str = "。！？.!?"
_ASCII_WORD_END: str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


#: Unicode blocks whose scripts do not separate words with spaces.
_CJK_RANGES: tuple[tuple[int, int], ...] = (
    (0x3040, 0x30FF),  # kana
    (0x3400, 0x4DBF),  # CJK extension A
    (0x4E00, 0x9FFF),  # CJK
    (0xAC00, 0xD7AF),  # hangul
    (0xF900, 0xFAFF),  # CJK compatibility
)


def _is_cjk(char: str) -> bool:
    """Return True for Han, kana and hangul (scripts without word spaces)."""
    code = ord(char)
    return any(low <= code <= high for low, high in _CJK_RANGES)


def _separator(left: str, right: str) -> str:
    """Return "" or " " for the seam between two pieces."""
    if not left or not right:
        return ""
    last, first = left[-1], right[0]
    if last in _CJK_PUNCT or _is_cjk(last) or _is_cjk(first):
        return ""
    if first in _CJK_PUNCT:
        return ""
    if last in _ASCII_WORD_END and first in _ASCII_WORD_END:
        return " "  # two latin words: "hello" + "world" -> "hello world"
    return ""


class TypeBatcher:
    """Collects ASR pieces and hands back whole sentences for typing."""

    def __init__(
        self,
        *,
        window_s: float = 12.0,
        max_chars: int = 80,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._window_s = window_s
        self._max_chars = max_chars
        self._clock = clock
        self._pieces: list[str] = []
        self._first_at = 0.0

    def add(self, text: str, *, forced: bool) -> str | None:
        """Add one ASR piece; returns the sentence to type when it is complete.

        ``forced`` says the segment was cut by the max-speech cap (the speaker
        is mid-sentence), so it is kept for later. A segment that ended on
        silence closes the sentence and flushes immediately.
        """
        piece = text.strip()
        if not piece:
            return None
        if not self._pieces:
            self._first_at = self._clock()
        self._pieces.append(piece)
        if not forced:
            return self.flush()
        if self.pending_chars >= self._max_chars:
            return self.flush()
        return None

    def should_flush(self, now: float | None = None) -> bool:
        """Return True when a forced-open batch has waited long enough to be typed.

        Only matters while the speaker keeps going: a batch that is already a
        whole sentence was flushed by :meth:`add`.
        """
        if not self._pieces:
            return False
        elapsed = (self._clock() if now is None else now) - self._first_at
        return elapsed >= self._window_s

    def flush(self) -> str | None:
        """Join and clear the batch; None when there was nothing pending."""
        if not self._pieces:
            return None
        joined = self._pieces[0]
        for piece in self._pieces[1:]:
            joined += _separator(joined, piece) + piece
        self._pieces = []
        self._first_at = 0.0
        return joined

    @property
    def pending_chars(self) -> int:
        """Characters currently waiting (separators excluded)."""
        return sum(len(piece) for piece in self._pieces)

    @property
    def pending(self) -> str:
        """What the batch would flush to right now, without clearing it."""
        if not self._pieces:
            return ""
        joined = self._pieces[0]
        for piece in self._pieces[1:]:
            joined += _separator(joined, piece) + piece
        return joined

    def __len__(self) -> int:
        """Return how many pieces are waiting."""
        return len(self._pieces)

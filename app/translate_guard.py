"""Reject a collapsed MT decode instead of typing it.

Greedy Marian decodes sometimes collapse into punctuation soup
(``给你吃。。」」」等。」等 喔。」.。。。``) — 1 of 8 test sentences did exactly that on
``いただきます``. An empty result makes the caller fall back to the source text,
which is always better than garbage on screen.
"""

from __future__ import annotations

#: Doubling any of these is a collapse; ``…`` and ``—`` are legitimate repeats.
_REPEAT_PUNCT_CHARS = "。、，！？；：」』）)，,;:"

#: Below this length a high punctuation share means "short sentence", not "garbage".
_MIN_CHARS_FOR_RATIO = 12


def looks_truncated(produced: int, max_new: int) -> bool:
    """True when a decode never emitted EOS and ran into the token budget.

    The field failure (``I'll I' After consulting We'll decide Then…``) had no
    repeated n-gram — it simply never ended. Running to the budget is the most
    reliable single signal of a collapsed greedy decode.
    """
    return max_new > 0 and produced >= max_new


def looks_degenerate(text: str) -> bool:
    """Return True when *text* looks like a collapsed decode, not a translation."""
    stripped = text.strip()
    if not stripped:
        return False

    for index in range(len(stripped) - 1):
        char = stripped[index]
        if char == stripped[index + 1] and char in _REPEAT_PUNCT_CHARS:
            return True

    words = stripped.split()
    if len(words) >= 6:
        pairs: dict[tuple[str, str], int] = {}
        singles: dict[str, int] = {}
        for index in range(len(words) - 1):
            pairs[(words[index], words[index + 1])] = (
                pairs.get((words[index], words[index + 1]), 0) + 1
            )
        for word in words:
            if len(word) >= 2:
                singles[word] = singles.get(word, 0) + 1
        if any(count >= 3 for count in pairs.values()):
            return True
        if any(count >= 3 for count in singles.values()):
            return True

    if len(stripped) >= _MIN_CHARS_FOR_RATIO:
        punctuation = sum(1 for char in stripped if not char.isalnum() and not char.isspace())
        return punctuation > len(stripped) / 3
    return False

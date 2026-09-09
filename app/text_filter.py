"""Pure hesitation-filler text post-processing (no backend imports)."""

import re

#: Characters that delimit clauses. A filler run is only removed when it opens
#: a clause (utterance start or after one of these), so mid-phrase particles
#: like "好啊" or "他是那个经理" are never eaten.
_CLAUSE_DELIMS = "，。！？；：,.!?;:、"


def _filler_alternation(words: str) -> str:
    """Regex alternation of the configured filler words, longest word first.

    Longest-first matters: a multi-char filler ("那个") must match before its
    single-char pieces ("那") so removing it never leaves stray characters.
    """
    ordered = sorted({w for w in words.split(",") if w}, key=len, reverse=True)
    return "|".join(map(re.escape, ordered))


def filter_fillers(text: str, words: str) -> str:
    """Remove hesitation fillers at the start of each clause; keep the rest.

    A run of filler words is treated as hesitation only when it opens a clause
    (utterance start or right after clause punctuation). Mid-phrase occurrences
    keep their meaning, so "好啊" and "他是那个部门的经理" survive untouched.
    """
    if not text.strip():
        return text.strip()
    alternation = _filler_alternation(words)
    if not alternation:
        return text
    leading = re.compile(rf"^(?:{alternation})+")
    # Split on punctuation with a capturing group so delimiters are kept.
    parts = re.split(f"([{re.escape(_CLAUSE_DELIMS)}])", text)
    out: list[str] = []
    previous_removed = False
    for part in parts:
        if not part:
            continue
        if len(part) == 1 and part in _CLAUSE_DELIMS:
            if not previous_removed:
                out.append(part)
            previous_removed = False
            continue
        stripped = leading.sub("", part)
        if stripped:
            out.append(stripped)
            previous_removed = False
        else:
            previous_removed = True
    return "".join(out)

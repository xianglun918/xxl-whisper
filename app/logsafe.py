"""Log-safe descriptions of user text.

The product promises "纯本地、隐私无忧" and the docs tell users to send
``app.log`` to the maintainer when something breaks. Logging the transcript
itself contradicted both: every dictated sentence, and every translation, was
written to disk in plain text. These helpers keep the diagnostic value (length,
shape) without the content.
"""


def describe_text(text: str) -> str:
    """Return a log-safe stand-in for *text*: its length, never the content."""
    if not text:
        return "empty"
    return f"{len(text)} chars"


def describe_pair(source: str, target: str) -> str:
    """Describe an ASR/translation pair without either side's content."""
    return f"{describe_text(source)} -> {describe_text(target)}"

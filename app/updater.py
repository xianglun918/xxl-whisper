"""GitHub Release based update detection (pure fetch + parse, no UI)."""

import json
import urllib.request
from dataclasses import dataclass

_API_URL: str = "https://api.github.com/repos/xianglun918/xxl-whisper/releases/latest"
_NOTES_LIMIT: int = 200


class UpdateCheckError(Exception):
    """Raised when the latest release cannot be fetched or parsed."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class ReleaseInfo:
    tag: str
    version: tuple[int, int, int]
    url: str
    notes: str


_VERSION_PARTS: int = 3


def parse_version(text: str) -> tuple[int, int, int]:
    """Parse 'vX.Y.Z' / 'X.Y.Z' into a comparable triple."""
    cleaned = text.strip().lstrip("vV")
    parts = cleaned.split(".")
    if len(parts) != _VERSION_PARTS or not all(part.isdigit() for part in parts):
        raise UpdateCheckError(reason=f"invalid version: {text!r}")
    return (int(parts[0]), int(parts[1]), int(parts[2]))


def is_newer(candidate: tuple[int, int, int], current: tuple[int, int, int]) -> bool:
    """Numeric release comparison; equal versions are not newer."""
    return candidate > current


def fetch_latest_release(timeout_s: float = 10.0) -> ReleaseInfo:
    """Query the GitHub API for the latest published release.

    Network/JSON/shape failures all surface as UpdateCheckError so callers
    can degrade silently on unstable connections.
    """
    request = urllib.request.Request(_API_URL, headers={"User-Agent": "xxl-whisper"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310
            raw: object = json.load(response)
    except (OSError, ValueError) as exc:
        raise UpdateCheckError(reason=f"fetch failed: {exc}") from exc
    if not isinstance(raw, dict):
        raise UpdateCheckError(reason="payload is not an object")
    tag = raw.get("tag_name")
    url = raw.get("html_url")
    body = raw.get("body")
    if not isinstance(tag, str) or not isinstance(url, str):
        raise UpdateCheckError(reason="payload missing tag_name/html_url")
    notes = body if isinstance(body, str) else ""
    return ReleaseInfo(
        tag=tag,
        version=parse_version(tag),
        url=url,
        notes=notes.strip()[:_NOTES_LIMIT],
    )

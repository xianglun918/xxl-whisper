"""GitHub Release based update detection (pure fetch + parse, no UI)."""

import json
import urllib.request
from dataclasses import dataclass

_API_URL: str = "https://api.github.com/repos/xianglun918/xxl-whisper/releases/latest"
_NOTES_LIMIT: int = 200

#: The Windows onefile asset and the checksum companion published beside it.
_WINDOWS_EXE_ASSET: str = "xxl-whisper.exe"
_CHECKSUM_SUFFIX: str = ".sha256"
_SHA256_HEX_LEN: int = 64
_HEX_DIGITS: frozenset[str] = frozenset("0123456789abcdef")


class UpdateCheckError(Exception):
    """Raised when the latest release cannot be fetched or parsed."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class AssetInfo:
    """One downloadable release asset (name, URL, and size when GitHub reports it)."""

    name: str
    url: str
    size: int | None


@dataclass(frozen=True, slots=True)
class ReleaseInfo:
    tag: str
    version: tuple[int, int, int]
    url: str
    notes: str
    assets: tuple[AssetInfo, ...]


@dataclass(frozen=True, slots=True)
class UpdateAssets:
    """The Windows exe plus its checksum companion needed for a one-click update."""

    exe: AssetInfo
    checksum: AssetInfo


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
        assets=_parse_assets(raw.get("assets")),
    )


def _parse_assets(raw: object) -> tuple[AssetInfo, ...]:
    """Parse the release ``assets`` array, skipping malformed entries."""
    if not isinstance(raw, list):
        return ()
    assets: list[AssetInfo] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        url = item.get("browser_download_url")
        size = item.get("size")
        if isinstance(name, str) and isinstance(url, str):
            assets.append(
                AssetInfo(name=name, url=url, size=size if isinstance(size, int) else None)
            )
    return tuple(assets)


def find_update_assets(release: ReleaseInfo) -> UpdateAssets | None:
    """Locate the Windows exe and its checksum asset in a release.

    ``None`` means this release cannot be applied in place (e.g. an older
    release published before checksums existed), so the caller falls back to
    opening the download page.
    """
    by_name = {asset.name: asset for asset in release.assets}
    exe = by_name.get(_WINDOWS_EXE_ASSET)
    checksum = by_name.get(f"{_WINDOWS_EXE_ASSET}{_CHECKSUM_SUFFIX}")
    if exe is None or checksum is None:
        return None
    return UpdateAssets(exe=exe, checksum=checksum)


def parse_checksum(text: str) -> str:
    """Return the lowercase SHA256 hex digest in *text*.

    Accepts either a bare digest or ``sha256sum`` output (``<digest>  <file>``).
    """
    tokens = text.split()
    digest = tokens[0].lower() if tokens else ""
    if len(digest) != _SHA256_HEX_LEN or not all(ch in _HEX_DIGITS for ch in digest):
        raise UpdateCheckError(reason=f"invalid sha256: {text.strip()!r}")
    return digest


def fetch_checksum(url: str, timeout_s: float = 10.0) -> str:
    """Download and parse a release ``.sha256`` companion file."""
    request = urllib.request.Request(url, headers={"User-Agent": "xxl-whisper"})  # noqa: S310
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310
            raw = response.read()
    except OSError as exc:
        raise UpdateCheckError(reason=f"checksum fetch failed: {exc}") from exc
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise UpdateCheckError(reason="checksum is not valid utf-8") from exc
    return parse_checksum(text)

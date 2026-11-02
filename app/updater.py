"""GitHub Release based update detection (pure fetch + parse, no UI)."""

import json
import re
import urllib.request
from dataclasses import dataclass

#: The *list* endpoint, not ``/releases/latest``. The repo also publishes model
#: bundles (``mt-ja-zh-v1`` …) as GitHub Releases, and GitHub's "latest" is just
#: the most recent non-prerelease release — so it returned a model bundle, whose
#: tag is not ``X.Y.Z``, and every update check died in ``parse_version``
#: (silently, by design). Listing releases and picking the newest tag that
#: *parses* as a version keeps model bundles out of the comparison.
_API_URL: str = "https://api.github.com/repos/xianglun918/xxl-whisper/releases?per_page=100"
_RELEASES_PAGE_LIMIT: int = 100
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


#: ``X.Y.Z`` with an optional PEP 440 pre-release suffix (``0.8.0b4``,
#: ``v0.8.0-beta.4``, ``0.9.0rc1``). The suffix is *parsed* (so a beta build can
#: talk about its own version at all — 0.8.0-beta.4 died at start-up with
#: ``invalid version: '0.8.0b4'``) but the triple is what gets compared.
_VERSION_RE: re.Pattern[str] = re.compile(
    r"^[vV]?(\d+)\.(\d+)\.(\d+)((?:[-_.]?(?:a|alpha|b|beta|rc|pre|preview)[-_.]?\d*)?)$",
    re.IGNORECASE,
)


def parse_version(text: str) -> tuple[int, int, int]:
    """Parse 'vX.Y.Z' (optionally with a pre-release suffix) into a comparable triple."""
    match = _VERSION_RE.fullmatch(text.strip())
    if match is None:
        raise UpdateCheckError(reason=f"invalid version: {text!r}")
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def is_prerelease(text: str) -> bool:
    """Return whether *text* carries a pre-release suffix (``0.8.0b4``, ``v0.9.0-rc1``)."""
    match = _VERSION_RE.fullmatch(text.strip())
    return bool(match and match.group(4))


def is_upgrade(
    candidate: ReleaseInfo,
    current: tuple[int, int, int],
    *,
    current_is_prerelease: bool,
) -> bool:
    """Return whether a build reporting *current* should offer *candidate*.

    A higher triple always wins. Equal triples only count when we are running a
    pre-release and the candidate is a full release — otherwise a beta install
    would sit on 0.8.0b4 for ever and never be told that 0.8.0 shipped.
    """
    if is_newer(candidate.version, current):
        # A pre-release is never an "upgrade" anybody gets told about: betas are
        # opt-in by downloading them, never pushed by the update check.
        return not is_prerelease(candidate.tag)
    return (
        current_is_prerelease
        and candidate.version == current
        and not is_prerelease(candidate.tag)
    )


def is_newer(candidate: tuple[int, int, int], current: tuple[int, int, int]) -> bool:
    """Numeric release comparison; equal versions are not newer."""
    return candidate > current


def fetch_latest_release(timeout_s: float = 10.0) -> ReleaseInfo:
    """Query the GitHub API for the newest release whose tag is a version.

    Network/JSON/shape failures all surface as UpdateCheckError so callers
    can degrade silently on unstable connections.
    """
    request = urllib.request.Request(_API_URL, headers={"User-Agent": "xxl-whisper"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310
            raw: object = json.load(response)
    except (OSError, ValueError) as exc:
        raise UpdateCheckError(reason=f"fetch failed: {exc}") from exc
    chosen = _newest_versioned_release(raw)
    tag = chosen.get("tag_name")
    url = chosen.get("html_url")
    body = chosen.get("body")
    if not isinstance(tag, str) or not isinstance(url, str):
        raise UpdateCheckError(reason="payload missing tag_name/html_url")
    notes = body if isinstance(body, str) else ""
    return ReleaseInfo(
        tag=tag,
        version=parse_version(tag),
        url=url,
        notes=notes.strip()[:_NOTES_LIMIT],
        assets=_parse_assets(chosen.get("assets")),
    )


def _newest_versioned_release(raw: object) -> dict[str, object]:
    """Pick the highest-versioned, published, non-prerelease entry of the list.

    Drafts, prereleases and non-version tags (model bundles such as
    ``mt-ja-zh-v1``) are skipped rather than raising: one unrelated release
    must not disable update checks for everybody.
    """
    if not isinstance(raw, list):
        raise UpdateCheckError(reason="payload is not a list of releases")
    best: dict[str, object] | None = None
    best_version: tuple[int, int, int] | None = None
    for item in raw[:_RELEASES_PAGE_LIMIT]:
        if not isinstance(item, dict):
            continue
        if item.get("draft") is True or item.get("prerelease") is True:
            continue
        tag = item.get("tag_name")
        if not isinstance(tag, str):
            continue
        # Both checks are needed. `prerelease` is a flag a human can forget to
        # tick when publishing by hand; the tag itself never lies. Without this,
        # a "v0.9.0-beta.1" release published without the flag would parse as
        # (0, 9, 0) and be offered to every stable 0.8.x user as "the new version".
        if is_prerelease(tag):
            continue
        try:
            version = parse_version(tag)
        except UpdateCheckError:
            continue
        if best_version is None or version > best_version:
            best, best_version = item, version
    if best is None:
        raise UpdateCheckError(reason="no versioned release found")
    return best


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

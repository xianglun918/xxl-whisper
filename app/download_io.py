"""Resumable, retrying, hash-verified byte transfer for model artifacts.

This module owns *how* bytes move reliably: it streams to a ``.part`` staging
file, resumes with an HTTP ``Range`` request when a partial exists, retries
transient failures a bounded number of times, and verifies the SHA256 of every
completed artifact before it is moved into place. The model catalog — *what* to
fetch and where it goes — lives in :mod:`app.downloader`.
"""

import hashlib
import http.client
import logging
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

ProgressFn = Callable[[str, int, int], None]  # (filename, downloaded_bytes, total_bytes)

#: Retry policy for a single source: up to _MAX_ATTEMPTS tries with a short
#: linear backoff between them, then the caller falls through to the next source.
_MAX_ATTEMPTS: int = 3
_RETRY_BACKOFF_BASE_S: float = 0.5
#: HTTP 206 Partial Content — the server honoured our Range request.
_HTTP_PARTIAL_CONTENT: int = 206
#: Failures worth retrying: socket/OS errors and malformed HTTP responses
#: (e.g. a connection dropped mid-body raises http.client.IncompleteRead).
_RETRYABLE: tuple[type[Exception], ...] = (OSError, http.client.HTTPException)
_TIMEOUT_S: int = 60
_CHUNK_SIZE: int = 1 << 20


class DownloadError(Exception):
    """Raised when model files cannot be fetched from any source."""

    def __init__(self, source: str, reason: str) -> None:
        super().__init__(f"{source}: {reason}")
        self.source = source
        self.reason = reason


@dataclass(frozen=True, slots=True)
class FileSpec:
    """One artifact to fetch: where from, where to, and how to verify it."""

    url: str
    dest: Path
    expected_size: int | None  # None = skip the post-download size check
    expected_sha256: frozenset[str]  # empty = skip the post-download hash check


def download(spec: FileSpec, progress: ProgressFn, *, proxy: str = "") -> None:
    """Stream one URL to dest via a resumable, retried ``.part`` file.

    The stream is retried up to ``_MAX_ATTEMPTS`` times (resuming each time),
    then the completed file is checked against the expected size and SHA256
    before it replaces the destination. A hash mismatch deletes the file and
    raises; it is not retried because a bad digest is not a transient error.
    """
    url, dest, expected_size, expected_sha256 = (
        spec.url,
        spec.dest,
        spec.expected_size,
        spec.expected_sha256,
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    opener = build_opener(proxy)
    _stream_with_retry(opener, spec, part, progress)
    actual = part.stat().st_size
    if expected_size is not None and actual != expected_size:
        part.unlink(missing_ok=True)
        raise DownloadError(source=url, reason=f"size mismatch: {actual}")
    digest = verify_sha256(part, expected_sha256, source=url)
    log.info(
        "download complete: %s (%d bytes, sha256 %s)",
        dest.name,
        actual,
        digest or "unchecked",
    )
    part.replace(dest)


def _stream_with_retry(
    opener: urllib.request.OpenerDirector,
    spec: FileSpec,
    part: Path,
    progress: ProgressFn,
) -> None:
    """Fetch *spec* up to ``_MAX_ATTEMPTS`` times, then give up.

    Transient network errors and truncated transfers (a size mismatch) are
    retried with a short backoff. The partially written ``.part`` is kept on
    failure so the next attempt -- or the next app run -- resumes instead of
    restarting from zero.
    """
    last_reason = "download failed"
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            _stream_once(opener, spec, part, progress)
            if _size_ok(spec, part):
                return
            last_reason = f"size mismatch: {part.stat().st_size}"
        except _RETRYABLE as exc:
            last_reason = str(exc)
        if attempt < _MAX_ATTEMPTS:
            delay = _backoff_seconds(attempt)
            log.info(
                "%s attempt %d/%d failed (%s); retrying in %.1fs",
                spec.dest.name,
                attempt,
                _MAX_ATTEMPTS,
                last_reason,
                delay,
            )
            _sleep(delay)
    raise DownloadError(source=spec.url, reason=last_reason)


def _stream_once(
    opener: urllib.request.OpenerDirector,
    spec: FileSpec,
    part: Path,
    progress: ProgressFn,
) -> None:
    """One HTTP attempt, resuming from an existing ``.part`` when possible."""
    display_name = spec.dest.name
    offset = _resume_offset(
        part.stat().st_size if part.exists() else 0, spec.expected_size
    )
    request = urllib.request.Request(spec.url)  # noqa: S310 - module-level https constants
    range_header = _range_header(offset)
    if range_header is not None:
        request.add_header("Range", range_header)
    # URLs are module-level https constants, not user input.
    with opener.open(request, timeout=_TIMEOUT_S) as response:
        status = response.status
        if offset > 0 and status == _HTTP_PARTIAL_CONTENT:
            mode, start = "ab", offset
            log.info("%s: resuming at byte %d", display_name, offset)
        else:
            mode, start = "wb", 0
            if offset > 0:
                log.info("%s: server ignored range, restarting from zero", display_name)
        header_size = response.headers.get("Content-Length")
        content_length = int(header_size) if header_size else 0
        total = start + content_length if content_length else (spec.expected_size or 0)
        with part.open(mode) as out:
            downloaded = start
            while chunk := response.read(_CHUNK_SIZE):
                out.write(chunk)
                downloaded += len(chunk)
                progress(display_name, downloaded, total)


def _resume_offset(part_size: int, expected_size: int | None) -> int:
    """Bytes of an existing ``.part`` to keep (0 = start the file clean).

    A partial that is already at least the expected size is stale or oversized,
    so it is restarted rather than resumed.
    """
    if part_size <= 0:
        return 0
    if expected_size is not None and part_size >= expected_size:
        return 0
    return part_size


def _range_header(offset: int) -> str | None:
    """``Range`` header value for a resume, or None for a fresh download."""
    return f"bytes={offset}-" if offset > 0 else None


def _size_ok(spec: FileSpec, part: Path) -> bool:
    """Whether the staged file matches the expected size (when one is known)."""
    if spec.expected_size is None:
        return True
    return part.stat().st_size == spec.expected_size


def _backoff_seconds(attempt: int) -> float:
    """Short linear backoff: 0.5s after the first failure, 1.0s after the second."""
    return _RETRY_BACKOFF_BASE_S * attempt


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def sha256_of(path: Path) -> str:
    """Streaming SHA256 of a file, so a ~1GB model never loads into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: Path, expected: frozenset[str], *, source: str) -> str:
    """Return the file's SHA256, or delete it and raise on a mismatch.

    An empty *expected* set means no hash was recorded for this artifact (e.g.
    the tarball container itself), so verification is skipped.
    """
    if not expected:
        return ""
    digest = sha256_of(path)
    if digest not in expected:
        path.unlink(missing_ok=True)
        wanted = "/".join(sorted(expected))
        raise DownloadError(
            source=source,
            reason=f"sha256 mismatch: computed {digest}, expected {wanted}",
        )
    return digest


def build_opener(proxy: str) -> urllib.request.OpenerDirector:
    """Build a urllib opener with an optional HTTP/HTTPS proxy.

    When *proxy* is non-empty, both HTTP and HTTPS requests go through it.
    Otherwise falls back to the default urllib behaviour (which honours
    HTTP_PROXY / HTTPS_PROXY env vars if set).
    """
    if proxy:
        handler = urllib.request.ProxyHandler({"http": proxy, "https": proxy})
    else:
        handler = urllib.request.ProxyHandler()  # respect env vars
    return urllib.request.build_opener(handler)

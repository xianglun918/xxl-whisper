"""Update detection: version parsing, comparison, and release fetch."""

import io
import json

import app.updater as upd
import pytest
from app.updater import UpdateCheckError, fetch_latest_release, is_newer, parse_version


def test_parse_version_accepts_v_prefix() -> None:
    assert parse_version("v0.2.0") == (0, 2, 0)
    assert parse_version("1.10.3") == (1, 10, 3)


def test_parse_version_rejects_garbage() -> None:
    for bad in ("", "v1", "1.2.x", "v1.2.3.4"):
        with pytest.raises(UpdateCheckError):
            parse_version(bad)


def test_is_newer_compares_numerically() -> None:
    assert is_newer((0, 2, 0), (0, 1, 9)) is True
    assert is_newer((1, 0, 0), (0, 9, 9)) is True
    assert is_newer((0, 1, 0), (0, 1, 0)) is False
    assert is_newer((0, 1, 0), (0, 2, 0)) is False


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._stream = io.BytesIO(payload)

    def read(self, n: int = -1) -> bytes:
        return self._stream.read(n)

    def __enter__(self) -> "_FakeResponse":  # noqa: PYI034 — stub mirrors urlopen shape
        return self

    def __exit__(self, *_args: object) -> None:
        return None


def test_fetch_latest_release_parses_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = json.dumps(
        {
            "tag_name": "v0.2.0",
            "html_url": "https://github.com/xianglun918/xxl-whisper/releases/tag/v0.2.0",
            "body": "支持更新检测与自动提示。\n第二行被截断。",
        }
    ).encode("utf-8")

    captured: list[object] = []

    def fake_urlopen(request: object, timeout: float) -> _FakeResponse:
        captured.append((request, timeout))
        return _FakeResponse(payload)

    monkeypatch.setattr(upd.urllib.request, "urlopen", fake_urlopen)
    release = fetch_latest_release(timeout_s=7.5)

    assert release.tag == "v0.2.0"
    assert release.version == (0, 2, 0)
    assert release.url.endswith("v0.2.0")
    assert "更新检测" in release.notes
    assert captured[0][1] == 7.5


def test_fetch_latest_release_wraps_network_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reason = "connection reset"

    def fake_urlopen(request: object, timeout: float) -> _FakeResponse:
        raise OSError(reason)

    monkeypatch.setattr(upd.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(UpdateCheckError):
        fetch_latest_release()


def test_fetch_latest_release_rejects_bad_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(request: object, timeout: float) -> _FakeResponse:
        return _FakeResponse(b'{"tag_name": "not-a-version"}')

    monkeypatch.setattr(upd.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(UpdateCheckError):
        fetch_latest_release()

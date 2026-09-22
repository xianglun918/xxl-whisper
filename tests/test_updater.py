"""Update detection: version parsing, comparison, and release fetch."""

import io
import json

import app.updater as upd
import pytest
from app.updater import (
    AssetInfo,
    ReleaseInfo,
    UpdateCheckError,
    fetch_checksum,
    fetch_latest_release,
    find_update_assets,
    is_newer,
    parse_checksum,
    parse_version,
)


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


def test_fetch_latest_release_parses_assets(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = json.dumps(
        {
            "tag_name": "v0.3.0",
            "html_url": "https://example.invalid/v0.3.0",
            "body": "",
            "assets": [
                {
                    "name": "xxl-whisper.exe",
                    "browser_download_url": "https://example.invalid/x.exe",
                    "size": 1234,
                },
                {
                    "name": "xxl-whisper.exe.sha256",
                    "browser_download_url": "https://example.invalid/x.sha256",
                },
                42,
                "not-a-dict",
            ],
        }
    ).encode("utf-8")

    def fake_urlopen(request: object, timeout: float) -> _FakeResponse:
        return _FakeResponse(payload)

    monkeypatch.setattr(upd.urllib.request, "urlopen", fake_urlopen)
    release = fetch_latest_release()

    assert [asset.name for asset in release.assets] == [
        "xxl-whisper.exe",
        "xxl-whisper.exe.sha256",
    ]
    assert release.assets[0].size == 1234
    assert release.assets[1].size is None


def _release_with(assets: list[AssetInfo]) -> ReleaseInfo:
    return ReleaseInfo(
        tag="v0.3.0", version=(0, 3, 0), url="u", notes="", assets=tuple(assets)
    )


def test_find_update_assets_requires_exe_and_checksum() -> None:
    exe = AssetInfo(name="xxl-whisper.exe", url="u1", size=10)
    sum_asset = AssetInfo(name="xxl-whisper.exe.sha256", url="u2", size=None)

    found = find_update_assets(_release_with([exe, sum_asset]))
    assert found is not None
    assert found.exe == exe
    assert found.checksum == sum_asset

    assert find_update_assets(_release_with([exe])) is None
    assert find_update_assets(_release_with([sum_asset])) is None
    assert find_update_assets(_release_with([])) is None


def test_parse_checksum_accepts_bare_and_sha256sum_output() -> None:
    digest = "a" * 64
    assert parse_checksum(digest) == digest
    assert parse_checksum(f"{digest}  xxl-whisper.exe") == digest
    assert parse_checksum(f"  {digest.upper()}\n") == digest


def test_parse_checksum_rejects_garbage() -> None:
    for bad in ("", "abc", "z" * 64, "a" * 63, "a" * 65):
        with pytest.raises(UpdateCheckError):
            parse_checksum(bad)


def test_fetch_checksum_parses_response(monkeypatch: pytest.MonkeyPatch) -> None:
    digest = "b" * 64

    def fake_urlopen(request: object, timeout: float) -> _FakeResponse:
        return _FakeResponse(digest.encode())

    monkeypatch.setattr(upd.urllib.request, "urlopen", fake_urlopen)
    assert fetch_checksum("https://example.invalid/x.sha256") == digest


def test_fetch_checksum_wraps_network_error(monkeypatch: pytest.MonkeyPatch) -> None:
    reason = "connection reset"

    def boom(request: object, timeout: float) -> _FakeResponse:
        raise OSError(reason)

    monkeypatch.setattr(upd.urllib.request, "urlopen", boom)
    with pytest.raises(UpdateCheckError):
        fetch_checksum("https://example.invalid/x.sha256")

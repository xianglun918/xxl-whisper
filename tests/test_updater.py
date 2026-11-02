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
    is_prerelease,
    is_upgrade,
    parse_checksum,
    parse_version,
)


def test_parse_version_accepts_v_prefix() -> None:
    assert parse_version("v0.2.0") == (0, 2, 0)
    assert parse_version("1.10.3") == (1, 10, 3)


def test_parse_version_rejects_garbage() -> None:
    for text in ("", "0.8", "v1.2.3.4", "not-a-version", "mt-ja-zh-v1"):
        with pytest.raises(UpdateCheckError):
            parse_version(text)


def test_parse_version_accepts_prerelease_suffixes() -> None:
    """构建自己的版本号带预发布后缀（0.8.0b4），解析不了会让整个 App 起不来。"""
    assert parse_version("0.8.0b4") == (0, 8, 0)
    assert parse_version("v0.8.0-beta.4") == (0, 8, 0)
    assert parse_version("0.9.0rc1") == (0, 9, 0)
    assert parse_version("1.0.0-preview.2") == (1, 0, 0)


def test_is_prerelease_flags_suffixes_only() -> None:
    assert is_prerelease("0.8.0b4") is True
    assert is_prerelease("v0.9.0-rc1") is True
    assert is_prerelease("v0.8.0") is False
    assert is_prerelease("not-a-version") is False


def test_is_upgrade_offers_the_stable_release_to_a_beta_build() -> None:
    """跑 beta 的人必须还能收到同号正式版的通知，否则会永远卡在 beta 上。"""
    beta = ReleaseInfo(tag="v0.8.0-beta.4", version=(0, 8, 0), url="u", notes="", assets=())
    stable = ReleaseInfo(tag="v0.8.0", version=(0, 8, 0), url="u", notes="", assets=())
    older = ReleaseInfo(tag="v0.7.0", version=(0, 7, 0), url="u", notes="", assets=())
    newer_stable = ReleaseInfo(tag="v0.9.0", version=(0, 9, 0), url="u", notes="", assets=())

    assert is_upgrade(beta, (0, 8, 0), current_is_prerelease=False) is False
    assert is_upgrade(stable, (0, 8, 0), current_is_prerelease=False) is False
    assert is_upgrade(stable, (0, 8, 0), current_is_prerelease=True) is True
    assert is_upgrade(older, (0, 8, 0), current_is_prerelease=True) is False
    # beta 本身永远不推：要试新 beta 得自己下载
    assert is_upgrade(beta, (0, 7, 0), current_is_prerelease=False) is False
    assert is_upgrade(newer_stable, (0, 8, 0), current_is_prerelease=False) is True


def test_parse_version_rejects_garbage_legacy() -> None:
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
        [
            {
                "tag_name": "v0.2.0",
                "html_url": "https://github.com/xianglun918/xxl-whisper/releases/tag/v0.2.0",
                "body": "支持更新检测与自动提示。\n第二行被截断。",
            }
        ]
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
        return _FakeResponse(b'[{"tag_name": "not-a-version"}]')

    monkeypatch.setattr(upd.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(UpdateCheckError):
        fetch_latest_release()


def test_fetch_latest_release_parses_assets(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = json.dumps(
        [
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
        ]
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


def test_fetch_latest_release_skips_model_bundles_and_prereleases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """模型包与预发布都不能阻止真·新版本的发现。

    线上真实情况：`/releases/latest` 返回的是模型包 `mt-ja-zh-v1`，解析版本时抛错，
    于是所有人都收不到更新提示（静默）。列表 + 过滤才是稳的形状。
    """
    payload = json.dumps(
        [
            {"tag_name": "mt-ja-zh-v1", "html_url": "https://example.invalid/model"},
            {
                "tag_name": "v0.9.0-beta.1",
                "html_url": "https://example.invalid/beta",
                "prerelease": True,
            },
            {"tag_name": "v0.8.3", "html_url": "https://example.invalid/083"},
            {"tag_name": "v0.8.0", "html_url": "https://example.invalid/080"},
            {"tag_name": "v0.9.0", "html_url": "https://example.invalid/draft", "draft": True},
        ]
    ).encode("utf-8")

    monkeypatch.setattr(
        upd.urllib.request, "urlopen", lambda *_args, **_kwargs: _FakeResponse(payload)
    )
    release = fetch_latest_release()

    assert release.tag == "v0.8.3"
    assert release.version == (0, 8, 3)


def test_a_prerelease_tag_is_skipped_even_without_the_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """手发 Release 时忘勾 prerelease 也不能把 beta 当正式版推给稳定用户。

    Review 2026-11-02 指出的回归：旧代码里 beta tag 会因解析失败被跳过，
    parse_version 放宽之后它反而能被选中，再由 is_upgrade 推给所有人。
    """
    payload = json.dumps(
        [
            {"tag_name": "v0.9.0-beta.1", "html_url": "https://example.invalid/beta-noflag"},
            {"tag_name": "v0.8.3", "html_url": "https://example.invalid/083"},
        ]
    ).encode("utf-8")

    monkeypatch.setattr(
        upd.urllib.request, "urlopen", lambda *_args, **_kwargs: _FakeResponse(payload)
    )

    assert fetch_latest_release().tag == "v0.8.3"


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

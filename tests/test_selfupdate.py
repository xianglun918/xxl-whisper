"""Self-update helper: argument parsing, replace-retry, wait, checksum wiring."""

import hashlib
from pathlib import Path

import pytest
from app.download_io import DownloadError, verify_sha256
from app.selfupdate import (
    APPLY_FLAG,
    ApplyRequest,
    ReplaceOptions,
    build_update_spec,
    download_filename,
    install_new_exe,
    parse_update_args,
    should_retry_replace,
    supports_self_update,
    wait_until_released,
)
from app.updater import AssetInfo


def test_parse_update_args_recognizes_helper_invocation() -> None:
    request = parse_update_args([APPLY_FLAG, r"C:\apps\app.exe", r"C:\tmp\new.exe"])
    assert request == ApplyRequest(
        target=Path(r"C:\apps\app.exe"), source=Path(r"C:\tmp\new.exe")
    )


def test_parse_update_args_ignores_normal_launch() -> None:
    assert parse_update_args([]) is None
    assert parse_update_args(["--other"]) is None
    assert parse_update_args([APPLY_FLAG, "only-one-path"]) is None
    assert parse_update_args([APPLY_FLAG, "a", "b", "extra"]) is None
    assert parse_update_args(["a", "b", "c"]) is None


def test_download_filename_strips_v_prefix() -> None:
    assert download_filename("v0.6.2") == "xxl-whisper-0.6.2.exe"
    assert download_filename("0.6.2") == "xxl-whisper-0.6.2.exe"
    assert download_filename(" V1.0.0 ") == "xxl-whisper-1.0.0.exe"


def test_build_update_spec_carries_size_and_checksum(tmp_path: Path) -> None:
    asset = AssetInfo(name="xxl-whisper.exe", url="https://example.invalid/x.exe", size=321)
    dest = tmp_path / "x.exe"
    spec = build_update_spec(asset, "a" * 64, dest)
    assert spec.url == asset.url
    assert spec.dest == dest
    assert spec.expected_size == 321
    assert spec.expected_sha256 == frozenset({"a" * 64})


def test_should_retry_replace_retries_permission_error_while_attempts_remain() -> None:
    locked = PermissionError(5, "still mapped")
    assert should_retry_replace(attempt=1, attempts=5, exc=locked) is True
    assert should_retry_replace(attempt=4, attempts=5, exc=locked) is True


def test_should_retry_replace_stops_on_last_attempt() -> None:
    locked = PermissionError(5, "still mapped")
    assert should_retry_replace(attempt=5, attempts=5, exc=locked) is False


def test_should_retry_replace_does_not_retry_other_oserror() -> None:
    missing = FileNotFoundError(2, "no such file")
    assert should_retry_replace(attempt=1, attempts=5, exc=missing) is False


def test_wait_until_released_returns_true_when_probe_clears() -> None:
    states = iter([True, True, False])
    assert (
        wait_until_released(
            lambda: next(states), timeout_s=10.0, sleep=lambda _s: None
        )
        is True
    )


def test_wait_until_released_times_out() -> None:
    assert (
        wait_until_released(lambda: True, timeout_s=0.0, sleep=lambda _s: None) is False
    )


def test_install_new_exe_retries_locked_target_then_succeeds(tmp_path: Path) -> None:
    source = tmp_path / "new.exe"
    source.write_bytes(b"new")
    target = tmp_path / "app.exe"
    target.write_bytes(b"old")
    attempts: list[int] = []

    def fake_replace(src: Path, dst: Path) -> None:
        attempts.append(1)
        if len(attempts) < 3:
            raise PermissionError(5, "locked")
        dst.write_bytes(src.read_bytes())

    install_new_exe(
        source,
        target,
        options=ReplaceOptions(
            attempts=5, interval_s=0.0, replace=fake_replace, sleep=lambda _s: None
        ),
    )
    assert len(attempts) == 3
    assert target.read_bytes() == b"new"


def test_install_new_exe_real_replace_swaps_file(tmp_path: Path) -> None:
    source = tmp_path / "new.exe"
    source.write_bytes(b"new")
    target = tmp_path / "app.exe"
    target.write_bytes(b"old")

    install_new_exe(source, target, options=ReplaceOptions(attempts=3, interval_s=0.0))

    assert target.read_bytes() == b"new"
    assert source.read_bytes() == b"new"  # the download is copied, not consumed
    assert [p for p in tmp_path.iterdir() if p.name.startswith(".app.exe.new")] == []


def test_install_new_exe_cleans_staging_on_persistent_failure(tmp_path: Path) -> None:
    source = tmp_path / "new.exe"
    source.write_bytes(b"new")
    target = tmp_path / "app.exe"
    target.write_bytes(b"old")

    def always_locked(_src: Path, _dst: Path) -> None:
        raise PermissionError(5, "locked")

    with pytest.raises(PermissionError):
        install_new_exe(
            source,
            target,
            options=ReplaceOptions(
                attempts=3, interval_s=0.0, replace=always_locked, sleep=lambda _s: None
            ),
        )
    assert target.read_bytes() == b"old"
    assert [p for p in tmp_path.iterdir() if p.name.startswith(".app.exe.new")] == []


def test_checksum_wiring_verifies_then_rejects(tmp_path: Path) -> None:
    payload = b"new exe bytes"
    digest = hashlib.sha256(payload).hexdigest()
    asset = AssetInfo(
        name="xxl-whisper.exe", url="https://example.invalid/x.exe", size=len(payload)
    )
    dest = tmp_path / "x.exe"
    dest.write_bytes(payload)
    spec = build_update_spec(asset, digest, dest)

    assert verify_sha256(dest, spec.expected_sha256, source="test") == digest

    dest.write_bytes(b"tampered")
    with pytest.raises(DownloadError):
        verify_sha256(dest, spec.expected_sha256, source="test")
    assert not dest.exists()


def test_supports_self_update_is_false_for_unfrozen_build() -> None:
    assert supports_self_update() is False

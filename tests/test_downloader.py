"""Model download logic: reuse, tarball fallback, and size guards."""

import io
import tarfile
from pathlib import Path

import app.downloader as dl
import pytest
from app.downloader import DownloadError, ModelFiles, ensure_model


def _fake_progress(_name: str, _done: int, _total: int) -> None:
    pass


def test_ensure_model_reuses_complete_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dl, "MODEL_SIZE", 4)
    monkeypatch.setattr(dl, "TOKENS_SIZE", 2)
    (tmp_path / "model.onnx").write_bytes(b"mode")  # exactly MODEL_SIZE bytes
    (tmp_path / "tokens.txt").write_bytes(b"tk")

    files = ensure_model(tmp_path, progress=_fake_progress)

    assert files == ModelFiles(model=tmp_path / "model.onnx", tokens=tmp_path / "tokens.txt")


def test_tarball_fallback_extracts_missing_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_bytes = b"M" * 8
    tokens_bytes = b"T" * 4
    tar_bytes = io.BytesIO()
    with tarfile.open(fileobj=tar_bytes, mode="w:bz2") as tar:
        for name, payload in (
            ("pkg/model.int8.onnx", model_bytes),
            ("pkg/tokens.txt", tokens_bytes),
        ):
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    synthetic_tar = tmp_path / "synthetic.tar.bz2"
    synthetic_tar.write_bytes(tar_bytes.getvalue())

    monkeypatch.setattr(dl, "MODEL_SIZE", len(model_bytes))
    monkeypatch.setattr(dl, "TOKENS_SIZE", len(tokens_bytes))
    monkeypatch.setattr(dl, "_HF_BASE", "https://127.0.0.1:1/nope")  # force direct failure

    def fake_download(
        url: str, dest: Path, display_name: str, progress: dl.ProgressFn, expected_size: int | None
    ) -> None:
        if url != dl._GH_TARBALL:
            raise DownloadError(source=url, reason="primary down in test")
        dest.write_bytes(synthetic_tar.read_bytes())

    monkeypatch.setattr(dl, "_download", fake_download)

    files = ensure_model(tmp_path, progress=_fake_progress)

    assert files.model.read_bytes() == model_bytes
    assert files.tokens.read_bytes() == tokens_bytes


def test_all_sources_failing_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dl, "MODEL_SIZE", 4)
    monkeypatch.setattr(dl, "TOKENS_SIZE", 2)

    def always_fail(
        url: str, dest: Path, display_name: str, progress: dl.ProgressFn, expected_size: int | None
    ) -> None:
        raise DownloadError(source=url, reason="down")

    monkeypatch.setattr(dl, "_download", always_fail)

    with pytest.raises(DownloadError):
        ensure_model(tmp_path, progress=_fake_progress)

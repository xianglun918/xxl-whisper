"""Model download logic: reuse, tarball fallback, and size guards."""

import io
import tarfile
from pathlib import Path

import app.downloader as dl
import pytest
from app.downloader import DownloadError, ModelFiles, ensure_model, manual_download_guide


def _fake_progress(_name: str, _done: int, _total: int) -> None:
    pass


def _spec_map(kind: str) -> dict[str, int]:
    return {dest: size for _url, dest, size in dl._MODEL_FILES[kind]}


def test_ensure_model_reuses_complete_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir = tmp_path / "sensevoice"
    model_dir.mkdir()
    for dest, size in _spec_map("sensevoice").items():
        p = model_dir / dest
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x" * size)
    def _noop_fetch(_spec: dl._FileSpec, _progress: dl.ProgressFn) -> None:
        pass

    def _forbid_tarball(*_args: object, **_kwargs: object) -> None:
        pytest.fail("should not fetch tarball")

    monkeypatch.setattr(dl, "_fetch", _noop_fetch)
    monkeypatch.setattr(dl, "_fetch_tarball_fallback", _forbid_tarball)
    files = ensure_model("sensevoice", tmp_path, progress=_fake_progress)
    assert files == ModelFiles(kind="sensevoice", directory=model_dir)


def test_tarball_fallback_extracts_missing_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nano_specs = _spec_map("funasr_nano")
    contents = {
        dest: (f"payload-{dest}" * 10).encode()[:size].ljust(size, b"0")
        for dest, size in nano_specs.items()
    }
    tar_bytes = io.BytesIO()
    with tarfile.open(fileobj=tar_bytes, mode="w:bz2") as tar:
        for name, payload in contents.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    synthetic_tar = tmp_path / "synthetic.tar.bz2"
    synthetic_tar.write_bytes(tar_bytes.getvalue())

    monkeypatch.setattr(
        dl,
        "_MODEL_FILES",
        {
            "funasr_nano": tuple(
                (f"https://127.0.0.1:1/{dest}", dest, size)
                for dest, size in nano_specs.items()
            )
        },
    )

    def fake_download(
        spec: dl._FileSpec, progress: dl.ProgressFn, *, proxy: str = ""
    ) -> None:
        if spec.url not in dl._MODEL_TARBALLS["funasr_nano"]:
            raise DownloadError(source=spec.url, reason="primary down in test")
        spec.dest.write_bytes(synthetic_tar.read_bytes())

    monkeypatch.setattr(dl, "_download", fake_download)

    files = ensure_model("funasr_nano", tmp_path, progress=_fake_progress)

    for dest, payload in contents.items():
        assert (files.directory / dest).read_bytes() == payload


def test_all_sources_failing_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail_fetch(spec: dl._FileSpec, _progress: dl.ProgressFn, *, proxy: str = "") -> None:
        raise DownloadError(source=spec.url, reason="down")

    monkeypatch.setattr(dl, "_fetch", _fail_fetch)

    def fail_tarball(
        model_dir: Path, specs: list, progress: dl.ProgressFn, *, proxy: str = ""
    ) -> None:
        raise DownloadError(source="tarball", reason="down")

    monkeypatch.setattr(dl, "_fetch_tarball_fallback", fail_tarball)

    with pytest.raises(DownloadError):
        ensure_model("sensevoice", tmp_path, progress=_fake_progress)

def test_manual_download_guide_lists_files_and_urls() -> None:
    guide = manual_download_guide("funasr_nano", Path("C:/models"))
    assert "funasr_nano" in guide
    assert "llm.int8.onnx" in guide
    assert "Qwen3-0.6B" in guide
    assert "https://hf-mirror.com/" in guide

def test_member_to_spec_handles_sensevoice_alias() -> None:
    specs = [
        dl._FileSpec(url="x", dest=Path("model.onnx"), expected_size=1),
        dl._FileSpec(url="y", dest=Path("tokens.txt"), expected_size=1),
    ]
    wanted = dl._member_to_spec(specs, "sensevoice")
    # k2-fsa ships model.int8.onnx; our backup ships model.onnx — both must map.
    assert wanted["model.int8.onnx"] is wanted["model.onnx"]
    assert wanted["tokens.txt"] is specs[1]

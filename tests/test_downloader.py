"""Model download logic: reuse, tarball fallback, resume, retry, and integrity."""

import hashlib
import io
import tarfile
from pathlib import Path
from typing import Self

import app.download_io as dio
import app.downloader as dl
import pytest
from app.downloader import DownloadError, ModelFiles, ensure_model, manual_download_guide


def _fake_progress(_name: str, _done: int, _total: int) -> None:
    pass


def _spec_map(kind: str) -> dict[str, int]:
    return {dest: size for _url, dest, size, _sha256 in dl._MODEL_FILES[kind]}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class _FakeResponse:
    def __init__(self, status: int, body: bytes, content_length: int | None = None) -> None:
        self.status = status
        self._body = body
        self._pos = 0
        self.headers = {} if content_length is None else {"Content-Length": str(content_length)}

    def read(self, size: int = -1) -> bytes:
        end = len(self._body) if size < 0 else self._pos + size
        chunk = self._body[self._pos : end]
        self._pos += len(chunk)
        return chunk

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False


class _FakeOpener:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response
        self.requests: list[dio.urllib.request.Request] = []

    def open(
        self, request: dio.urllib.request.Request, **_kwargs: object
    ) -> _FakeResponse:
        self.requests.append(request)
        return self._response


def test_ensure_model_reuses_complete_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir = tmp_path / "sensevoice"
    model_dir.mkdir()
    for dest, size in _spec_map("sensevoice").items():
        p = model_dir / dest
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x" * size)

    def _noop_fetch(_spec: dio.FileSpec, _progress: dl.ProgressFn) -> None:
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
                (f"https://127.0.0.1:1/{dest}", dest, size, frozenset({_sha256(contents[dest])}))
                for dest, size in nano_specs.items()
            )
        },
    )

    def fake_download(
        spec: dio.FileSpec, progress: dl.ProgressFn, *, proxy: str = ""
    ) -> None:
        if spec.url not in dl._MODEL_TARBALLS["funasr_nano"]:
            raise DownloadError(source=spec.url, reason="primary down in test")
        spec.dest.write_bytes(synthetic_tar.read_bytes())

    monkeypatch.setattr(dl, "_download", fake_download)

    files = ensure_model("funasr_nano", tmp_path, progress=_fake_progress)

    for dest, payload in contents.items():
        assert (files.directory / dest).read_bytes() == payload


def test_tarball_fallback_rejects_wrong_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tarball member whose SHA256 is not the recorded value is discarded."""
    nano_specs = _spec_map("funasr_nano")
    contents = {dest: b"payload-" + dest.encode() for dest in nano_specs}
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
                (
                    f"https://127.0.0.1:1/{dest}",
                    dest,
                    len(contents[dest]),
                    frozenset({_sha256(b"other")}),
                )
                for dest in nano_specs
            )
        },
    )

    def fail_primary(spec: dio.FileSpec, _progress: dl.ProgressFn, *, proxy: str = "") -> None:
        raise DownloadError(source=spec.url, reason="primary down in test")

    def fake_download(
        spec: dio.FileSpec, _progress: dl.ProgressFn, *, proxy: str = ""
    ) -> None:
        spec.dest.write_bytes(synthetic_tar.read_bytes())

    monkeypatch.setattr(dl, "_fetch", fail_primary)
    monkeypatch.setattr(dl, "_download", fake_download)

    with pytest.raises(DownloadError, match="sha256 mismatch"):
        ensure_model("funasr_nano", tmp_path, progress=_fake_progress)


def test_all_sources_failing_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail_fetch(spec: dio.FileSpec, _progress: dl.ProgressFn, *, proxy: str = "") -> None:
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
        dio.FileSpec(
            url="x", dest=Path("model.onnx"), expected_size=1, expected_sha256=frozenset()
        ),
        dio.FileSpec(
            url="y", dest=Path("tokens.txt"), expected_size=1, expected_sha256=frozenset()
        ),
    ]
    wanted = dl._member_to_spec(specs, "sensevoice")
    # k2-fsa ships model.int8.onnx; our backup ships model.onnx — both must map.
    assert wanted["model.int8.onnx"] is wanted["model.onnx"]
    assert wanted["tokens.txt"] is specs[1]


# -- resume decision + Range header -------------------------------------------


def test_resume_offset_keeps_partial_and_restarts_when_full() -> None:
    assert dio._resume_offset(0, 100) == 0  # nothing staged
    assert dio._resume_offset(40, 100) == 40  # resume mid-file
    assert dio._resume_offset(100, 100) == 0  # already complete -> restart clean
    assert dio._resume_offset(120, 100) == 0  # oversized/stale -> restart clean
    assert dio._resume_offset(40, None) == 40  # unknown size -> resume


def test_range_header_only_for_a_positive_offset() -> None:
    assert dio._range_header(0) is None
    assert dio._range_header(40) == "bytes=40-"


def test_stream_once_resumes_with_range_and_appends(tmp_path: Path) -> None:
    part = tmp_path / "f.part"
    part.write_bytes(b"abc")
    spec = dio.FileSpec(
        url="https://example.test/f",
        dest=tmp_path / "f",
        expected_size=6,
        expected_sha256=frozenset(),
    )
    opener = _FakeOpener(_FakeResponse(206, b"def", content_length=3))

    dio._stream_once(opener, spec, part, _fake_progress)

    assert part.read_bytes() == b"abcdef"
    assert opener.requests[0].get_header("Range") == "bytes=3-"


def test_stream_once_restarts_when_server_ignores_range(tmp_path: Path) -> None:
    part = tmp_path / "f.part"
    part.write_bytes(b"abc")
    spec = dio.FileSpec(
        url="https://example.test/f",
        dest=tmp_path / "f",
        expected_size=6,
        expected_sha256=frozenset(),
    )
    opener = _FakeOpener(_FakeResponse(200, b"abcdef", content_length=6))

    dio._stream_once(opener, spec, part, _fake_progress)

    # The server answered 200 (full body): the stale partial must be replaced.
    assert part.read_bytes() == b"abcdef"
    assert opener.requests[0].get_header("Range") == "bytes=3-"


def test_stream_once_starts_clean_without_a_part(tmp_path: Path) -> None:
    part = tmp_path / "f.part"
    spec = dio.FileSpec(
        url="https://example.test/f",
        dest=tmp_path / "f",
        expected_size=3,
        expected_sha256=frozenset(),
    )
    opener = _FakeOpener(_FakeResponse(200, b"abc", content_length=3))

    dio._stream_once(opener, spec, part, _fake_progress)

    assert part.read_bytes() == b"abc"
    assert opener.requests[0].get_header("Range") is None


# -- retry policy -------------------------------------------------------------


def test_backoff_seconds_is_short_and_grows_linearly() -> None:
    assert dio._backoff_seconds(1) == 0.5
    assert dio._backoff_seconds(2) == 1.0
    assert dio._backoff_seconds(1) < dio._backoff_seconds(2)


def test_download_retries_transient_failures_then_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = b"payload"
    spec = dio.FileSpec(
        url="https://example.test/f",
        dest=tmp_path / "f",
        expected_size=len(body),
        expected_sha256=frozenset({_sha256(body)}),
    )
    attempts: list[int] = []

    def flaky(_opener: object, _spec: dio.FileSpec, part: Path, _progress: dl.ProgressFn) -> None:
        attempts.append(1)
        if len(attempts) < dio._MAX_ATTEMPTS:
            message = "transient blip"
            raise OSError(message)
        part.write_bytes(body)

    sleeps: list[float] = []
    monkeypatch.setattr(dio, "build_opener", lambda _proxy: object())
    monkeypatch.setattr(dio, "_stream_once", flaky)
    monkeypatch.setattr(dio, "_sleep", sleeps.append)

    dio.download(spec, _fake_progress)

    assert len(attempts) == dio._MAX_ATTEMPTS
    assert sleeps == [0.5, 1.0]
    assert (tmp_path / "f").read_bytes() == body


def test_download_retries_a_truncated_transfer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = b"complete-body"
    spec = dio.FileSpec(
        url="https://example.test/f",
        dest=tmp_path / "f",
        expected_size=len(body),
        expected_sha256=frozenset({_sha256(body)}),
    )
    calls: list[int] = []

    def truncated_once(
        _opener: object, _spec: dio.FileSpec, part: Path, _progress: dl.ProgressFn
    ) -> None:
        calls.append(1)
        part.write_bytes(body[:3] if len(calls) == 1 else body)

    monkeypatch.setattr(dio, "build_opener", lambda _proxy: object())
    monkeypatch.setattr(dio, "_stream_once", truncated_once)
    monkeypatch.setattr(dio, "_sleep", lambda _seconds: None)

    dio.download(spec, _fake_progress)

    assert len(calls) == 2
    assert (tmp_path / "f").read_bytes() == body


def test_download_gives_up_after_max_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = dio.FileSpec(
        url="https://example.test/f",
        dest=tmp_path / "f",
        expected_size=5,
        expected_sha256=frozenset({_sha256(b"hello")}),
    )

    def always_fail(*_args: object, **_kwargs: object) -> None:
        message = "down"
        raise OSError(message)

    monkeypatch.setattr(dio, "build_opener", lambda _proxy: object())
    monkeypatch.setattr(dio, "_stream_once", always_fail)
    monkeypatch.setattr(dio, "_sleep", lambda _seconds: None)

    with pytest.raises(DownloadError, match="down"):
        dio.download(spec, _fake_progress)


# -- hash verification --------------------------------------------------------


def test_verify_sha256_accepts_any_recorded_hash(tmp_path: Path) -> None:
    path = tmp_path / "f"
    path.write_bytes(b"data")
    good = _sha256(b"data")

    assert dio.verify_sha256(path, frozenset({"deadbeef", good}), source="x") == good
    assert path.exists()


def test_verify_sha256_deletes_and_raises_on_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "f"
    path.write_bytes(b"tampered")

    with pytest.raises(DownloadError, match="sha256 mismatch"):
        dio.verify_sha256(path, frozenset({_sha256(b"expected")}), source="x")

    assert not path.exists()


def test_verify_sha256_skips_when_no_hash_recorded(tmp_path: Path) -> None:
    path = tmp_path / "f"
    path.write_bytes(b"data")

    assert dio.verify_sha256(path, frozenset(), source="x") == ""
    assert path.exists()


def test_download_verifies_hash_and_replaces_part(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = b"hello world"
    spec = dio.FileSpec(
        url="https://example.test/f",
        dest=tmp_path / "f",
        expected_size=len(body),
        expected_sha256=frozenset({_sha256(body)}),
    )
    opener = _FakeOpener(_FakeResponse(200, body, content_length=len(body)))
    monkeypatch.setattr(dio, "build_opener", lambda _proxy: opener)

    dio.download(spec, _fake_progress)

    assert (tmp_path / "f").read_bytes() == body
    assert not (tmp_path / "f.part").exists()


def test_download_rejects_tampered_body_and_deletes_part(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = b"tampered payload"
    spec = dio.FileSpec(
        url="https://example.test/f",
        dest=tmp_path / "f",
        expected_size=len(body),
        expected_sha256=frozenset({_sha256(b"the real payload")}),
    )
    opener = _FakeOpener(_FakeResponse(200, body, content_length=len(body)))
    monkeypatch.setattr(dio, "build_opener", lambda _proxy: opener)

    with pytest.raises(DownloadError, match="sha256 mismatch"):
        dio.download(spec, _fake_progress)

    assert not (tmp_path / "f").exists()
    assert not (tmp_path / "f.part").exists()


def test_vad_artifact_accepts_both_known_hashes() -> None:
    assert dl._VAD_HF_SHA256 in dl._VAD_SHA256
    assert dl._VAD_GH_SHA256 in dl._VAD_SHA256
    assert len(dl._VAD_SHA256) == 2
    assert frozenset({1_807_522, 643_854}) == dl._VAD_SIZES

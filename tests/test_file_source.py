"""Offline file transcription: decode errors and the transcriber contract."""

import app.file_source as fs
import numpy as np
import pytest
from app.file_source import FileTranscribeError


class _Proc:
    returncode = 0
    stdout = b""
    stderr = b""


def test_missing_ffmpeg_raises_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _no_ffmpeg(*_args: object, **_kwargs: object) -> None:
        error = FileNotFoundError("ffmpeg")
        raise error

    monkeypatch.setattr(fs.subprocess, "run", _no_ffmpeg)
    with pytest.raises(FileTranscribeError) as excinfo:
        fs.decode_audio("clip.mp4")
    assert "ffmpeg" in excinfo.value.reason


def test_decode_returns_float32_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = np.arange(16_000, dtype=np.float32).tobytes()
    proc = _Proc()
    proc.stdout = payload
    monkeypatch.setattr(fs.subprocess, "run", lambda *_a, **_k: proc)
    samples = fs.decode_audio("clip.wav")
    assert samples.dtype == np.float32
    assert samples.size == 16_000


def test_transcribe_file_returns_stripped_text(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeRecognizer:
        def transcribe(self, _samples: np.ndarray) -> str:
            return " 你好世界 "

    proc = _Proc()
    proc.stdout = np.zeros(1_600, dtype=np.float32).tobytes()
    monkeypatch.setattr(fs.subprocess, "run", lambda *_a, **_k: proc)
    assert fs.transcribe_file("clip.wav", _FakeRecognizer()) == "你好世界"

"""Offline transcription of audio/video files (no playback).

Decodes any container ffmpeg understands to 16 kHz mono float32, then hands the
samples to the recognizer. Kept separate from the real-time recorder so the
live path never depends on an external ffmpeg binary — a missing ffmpeg only
turns this one feature off.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)

#: The recognizer expects 16 kHz mono float32.
SAMPLE_RATE: int = 16_000

#: How long ffmpeg may run before we give up (an hour-long meeting at worst).
_DECODE_TIMEOUT_S: float = 3600.0


def ffmpeg_available() -> bool:
    """Return True when the ffmpeg binary is on PATH.

    ffmpeg is a feature-time dependency (only file transcription needs it), not
    a runtime requirement: the app starts and dictates without it, and only
    asks for it right before a file transcription begins.
    """
    return shutil.which("ffmpeg") is not None


class FileTranscribeError(RuntimeError):
    """Raised when a file cannot be decoded or has nothing to transcribe."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class _Transcriber(Protocol):
    def transcribe(self, samples: np.ndarray) -> str: ...


def decode_audio(path: str | Path) -> np.ndarray:
    """Decode an audio/video file to float32 mono at 16 kHz via ffmpeg."""
    # The path is passed as a list element (never through a shell), so a
    # user-chosen filename cannot inject a command — ffmpeg only ever reads it.
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-",
    ]
    try:
        proc = subprocess.run(  # noqa: S603 — list argv, no shell, fixed binary
            cmd, capture_output=True, timeout=_DECODE_TIMEOUT_S, check=False
        )
    except FileNotFoundError as exc:
        raise FileTranscribeError(reason="未找到 ffmpeg，请先安装 ffmpeg 并加入 PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise FileTranscribeError(reason="ffmpeg 解码超时") from exc
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", "replace").strip()
        raise FileTranscribeError(reason=f"ffmpeg 解码失败：{detail or '未知错误'}")
    if not proc.stdout:
        raise FileTranscribeError(reason="文件没有可识别的音频（可能没有音轨）")
    return np.frombuffer(proc.stdout, dtype=np.float32)


def transcribe_file(path: str | Path, recognizer: _Transcriber) -> str:
    """Decode *path* and return the transcription of its full audio."""
    samples = decode_audio(path)
    if samples.size == 0:
        raise FileTranscribeError(reason="文件没有可识别的音频")
    text = recognizer.transcribe(samples)
    return text.strip()

"""Integration test: the real Silero VAD segments speech and ignores silence/noise.

Skipped when no VAD model is available (fresh checkout / CI). The model is the
same artifact ``ensure_vad_model`` installs; a local copy under the temp dir is
also accepted so the test runs before the app has downloaded it.
"""

import tempfile
import wave
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
from app.asr import Recognizer
from app.config import model_dir, models_root
from app.vad import SAMPLE_RATE, Segmenter, VadSettings

pytestmark = pytest.mark.integration


def _find_vad_model() -> Path | None:
    for candidate in (
        models_root() / "vad" / "silero_vad.onnx",
        Path(tempfile.gettempdir()) / "opencode" / "silero_vad.onnx",
    ):
        if candidate.exists():
            return candidate
    return None


def _load_samples(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    if rate != SAMPLE_RATE:
        xs = np.arange(pcm.shape[0], dtype=np.float64) * SAMPLE_RATE / rate
        pcm = np.interp(xs, np.arange(pcm.shape[0]), pcm.astype(np.float64)).astype(np.int16)
    return pcm.astype(np.float32) / 32768.0


def _settings(model: Path) -> VadSettings:
    return VadSettings(
        model_path=model,
        threshold=0.5,
        min_speech_s=0.25,
        min_silence_s=0.5,
        max_speech_s=20.0,
        num_threads=1,
    )


def _feed(segmenter: Segmenter, samples: np.ndarray, block: int) -> None:
    for start in range(0, samples.shape[0], block):
        segmenter.accept(samples[start : start + block])
    segmenter.flush()


def test_speech_segments_and_silence_noise_do_not() -> None:
    model = _find_vad_model()
    if model is None:
        pytest.skip("VAD model not downloaded")

    settings = _settings(model)

    # Real speech, fed in 1024-sample blocks (never the VAD's own 512 window) so
    # the internal framing is what makes detection work.
    speech = _load_samples(Path(__file__).parent / "assets" / "zh_test.wav")
    segmenter = Segmenter(settings)
    _feed(segmenter, speech, 1024)
    segments = segmenter.drain()
    assert len(segments) >= 1
    assert sum(s.shape[0] for s in segments) >= SAMPLE_RATE // 2  # >= 0.5 s

    # Digital silence and white noise must never look like speech.
    silence = np.zeros(SAMPLE_RATE * 4, dtype=np.float32)
    noise = (np.random.default_rng(0).standard_normal(SAMPLE_RATE * 4) * 0.1).astype(np.float32)
    for samples in (silence, noise):
        quiet = Segmenter(settings)
        _feed(quiet, samples, 700)
        assert quiet.drain() == []


def test_current_samples_grows_during_speech_and_is_none_when_quiet() -> None:
    """The live partial must come from the VAD's own in-progress segment."""
    model = _find_vad_model()
    if model is None:
        pytest.skip("VAD model not downloaded")

    settings = _settings(model)
    speech = _load_samples(Path(__file__).parent / "assets" / "zh_test.wav")
    segmenter = Segmenter(settings)
    lengths: list[int] = []
    for start in range(0, speech.shape[0], 1024):
        segmenter.accept(speech[start : start + 1024])
        current = segmenter.current_samples()
        if current is not None:
            assert current.dtype == np.float32
            lengths.append(current.shape[0])
        else:
            lengths.append(0)
    # The in-progress segment accumulates across blocks (and includes the onset,
    # which is the whole point: it is the audio that will be inserted).
    assert max(lengths) >= SAMPLE_RATE // 2
    assert any(later > earlier for earlier, later in pairwise(lengths))

    # No speech ⇒ no current segment to preview.
    silence = np.zeros(SAMPLE_RATE * 4, dtype=np.float32)
    noise = (np.random.default_rng(0).standard_normal(SAMPLE_RATE * 4) * 0.1).astype(np.float32)
    for quiet_samples in (silence, noise):
        quiet = Segmenter(settings)
        for start in range(0, quiet_samples.shape[0], 700):
            quiet.accept(quiet_samples[start : start + 700])
        assert quiet.current_samples() is None


def test_current_samples_decode_previews_the_finished_segment() -> None:
    """Decoding the VAD's current samples previews the sentence that is inserted."""
    model = _find_vad_model()
    asr_dir = model_dir()
    if model is None:
        pytest.skip("VAD model not downloaded")
    if not (asr_dir / "model.onnx").exists() or not (asr_dir / "tokens.txt").exists():
        pytest.skip("ASR model not downloaded")

    settings = _settings(model)
    speech = _load_samples(Path(__file__).parent / "assets" / "zh_test.wav")
    segmenter = Segmenter(settings)
    recognizer = Recognizer(
        kind="sensevoice", model_dir=asr_dir, num_threads=2, language="zh"
    )

    last_partial: np.ndarray | None = None
    checked = 0
    for start in range(0, speech.shape[0], 1024):
        segmenter.accept(speech[start : start + 1024])
        current = segmenter.current_samples()
        if current is not None:
            last_partial = current  # the newest preview before any endpoint
        for segment in segmenter.drain():
            assert last_partial is not None
            partial_text = recognizer.transcribe(last_partial)
            final_text = recognizer.transcribe(segment)
            assert partial_text
            assert final_text
            # Same source audio (the VAD's own segment): the preview shares the
            # inserted sentence's wording instead of diverging from it.
            assert partial_text.startswith(final_text) or final_text.startswith(partial_text)
            checked += 1
            last_partial = None
    assert checked >= 1

"""Integration test: real SenseVoice model against a known TTS utterance.

Skipped when the model has not been downloaded yet (fresh checkout / CI).
Utterance spoken in tests/assets/zh_test.wav:
"今天天气怎么样？我们一起去吃火锅吧，记得带上充电宝。"
"""

import wave
from pathlib import Path

import numpy as np
import pytest
from app.asr import SAMPLE_RATE, Recognizer
from app.config import model_dir

pytestmark = pytest.mark.integration


def _load_samples(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    if rate != SAMPLE_RATE:
        xs = np.arange(pcm.shape[0], dtype=np.float64) * SAMPLE_RATE / rate
        pcm = np.interp(xs, np.arange(pcm.shape[0]), pcm.astype(np.float64)).astype(np.int16)
    return pcm.astype(np.float32) / 32768.0


def test_transcribe_known_utterance() -> None:
    model = model_dir() / "model.onnx"
    tokens = model_dir() / "tokens.txt"
    if not model.exists() or not tokens.exists():
        pytest.skip("model not downloaded")

    recognizer = Recognizer(model=model, tokens=tokens, num_threads=2, language="zh")
    samples = _load_samples(Path(__file__).parent / "assets" / "zh_test.wav")

    text = recognizer.transcribe(samples)

    assert "今天天气" in text
    assert "火锅" in text

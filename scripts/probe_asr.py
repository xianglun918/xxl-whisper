"""One-shot probe: sherpa-onnx + SenseVoice on the production model path."""

import sys
import time
import wave
from pathlib import Path

import numpy as np
import sherpa_onnx

MODELS = Path.home() / "AppData" / "Local" / "xxl-whisper" / "models" / "sensevoice"
WAV = Path(r"C:\Users\AA\AppData\Local\Temp\opencode\zh_test.wav")

with wave.open(str(WAV), "rb") as w:
    rate = w.getframerate()
    pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)

target_rate = 16_000
if rate != target_rate:
    xs = np.arange(pcm.shape[0], dtype=np.float64) * target_rate / rate
    pcm = np.interp(xs, np.arange(pcm.shape[0]), pcm.astype(np.float64)).astype(np.int16)
samples = pcm.astype(np.float32) / 32768.0
print(f"audio: {rate} Hz -> 16 kHz, {samples.shape[0] / target_rate:.1f}s")

for language in ("zh", "auto"):
    rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=str(MODELS / "model.onnx"),
        tokens=str(MODELS / "tokens.txt"),
        num_threads=2,
        language=language,
        use_itn=True,
    )
    stream = rec.create_stream()
    stream.accept_waveform(target_rate, samples)
    t0 = time.perf_counter()
    rec.decode_stream(stream)
    dt = (time.perf_counter() - t0) * 1000
    print(f"[{language}] {dt:.0f}ms  raw={stream.result.text!r}")
sys.exit(0)

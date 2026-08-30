"""SenseVoice offline recognition via sherpa-onnx (CPU)."""

import re
from pathlib import Path

import numpy as np
import sherpa_onnx

SAMPLE_RATE: int = 16_000

#: SenseVoice emits rich-transcription tags like ``<zh>``, ``<|en|>``, ``</s>``
#: depending on the language setting; end users should never see them.
_TAG_RE = re.compile(r"<\|?/?[a-zA-Z_][a-zA-Z0-9_|]*\|?>")


class Recognizer:
    """Wraps :class:`sherpa_onnx.OfflineRecognizer` for utterance-at-a-time use."""

    def __init__(
        self, model: Path, tokens: Path, num_threads: int = 2, language: str = "zh"
    ) -> None:
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(model),
            tokens=str(tokens),
            num_threads=num_threads,
            language=language,
            use_itn=True,
        )

    def transcribe(self, samples: np.ndarray) -> str:
        """Decode one utterance of float32 mono samples at 16 kHz."""
        stream = self._recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, samples)
        self._recognizer.decode_stream(stream)
        return _TAG_RE.sub("", stream.result.text).strip()

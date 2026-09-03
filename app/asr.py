"""SenseVoice / FunASR-Nano offline recognition via sherpa-onnx (CPU)."""

import re
from pathlib import Path
from typing import assert_never

import numpy as np
import sherpa_onnx

from app.config import Disfluency, ModelKind

SAMPLE_RATE: int = 16_000

#: SenseVoice emits rich-transcription tags like ``<zh>``, ``<|en|>``, ``</s>``
#: depending on the language setting; end users should never see them.
_TAG_RE = re.compile(r"<\|?/?[a-zA-Z_][a-zA-Z0-9_|]*\|?>")

#: Decoder prompts for Fun-ASR-Nano (LLM-decoder ASR). verbatim transcribes
#: as-is; smooth asks the decoder to strip fillers/repetitions/false starts,
#: mirroring chat assistants' semantic smoothing (e.g. Doubao's DDC).
_FUNASR_NANO_PROMPTS: dict[Disfluency, str] = {
    "verbatim": "语音转写:",
    "smooth": "语音转写并去除嗯、呃、啊等语气填充词、重复和口误，输出流畅文本:",
}


class Recognizer:
    """Wraps :class:`sherpa_onnx.OfflineRecognizer` for utterance-at-a-time use."""

    def __init__(
        self,
        kind: ModelKind,
        model_dir: Path,
        num_threads: int = 2,
        language: str = "zh",
        disfluency: Disfluency = "verbatim",
    ) -> None:
        self.kind = kind
        match kind:
            case "sensevoice":
                self._recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                    model=str(model_dir / "model.onnx"),
                    tokens=str(model_dir / "tokens.txt"),
                    num_threads=num_threads,
                    language=language,
                    use_itn=True,
                )
            case "funasr_nano":
                self._recognizer = sherpa_onnx.OfflineRecognizer.from_funasr_nano(
                    encoder_adaptor=str(model_dir / "encoder_adaptor.int8.onnx"),
                    llm=str(model_dir / "llm.int8.onnx"),
                    embedding=str(model_dir / "embedding.int8.onnx"),
                    tokenizer=str(model_dir / "Qwen3-0.6B"),
                    num_threads=num_threads,
                    language=language,
                    user_prompt=_FUNASR_NANO_PROMPTS[disfluency],
                )
            case unreachable:
                assert_never(unreachable)

    def transcribe(self, samples: np.ndarray) -> str:
        """Decode one utterance of float32 mono samples at 16 kHz."""
        stream = self._recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, samples)
        self._recognizer.decode_stream(stream)
        return _TAG_RE.sub("", stream.result.text).strip()

"""SenseVoice / FunASR-Nano offline recognition via sherpa-onnx (CPU)."""

import logging
import re
from pathlib import Path
from typing import assert_never

import numpy as np
import sherpa_onnx

from app.config import Disfluency, ModelKind

log = logging.getLogger(__name__)

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

#: FunASR-Nano's ``language`` is a *prompt fragment in natural language*, not a
#: code like ``ja``. sherpa-onnx builds the model prompt from it, and it also
#: **suppresses our own ``user_prompt``** whenever it is non-empty.
_FUNASR_LANGUAGES: dict[str, str] = {
    "zh": "中文",
    "en": "英语",
    "ja": "日语",
    "auto": "",
}


def funasr_language(code: str, *, has_prompt: bool) -> str:
    """Return the value to pass as FunASR-Nano's ``language``.

    Two rules, both learned the hard way:

    * it is natural language (``日语``), not a code (``ja``) — passing a code gave
      the model a meaningless hint and it fell back to Chinese-flavoured output,
      which then translated badly;
    * a custom disfluency prompt wins only when ``language`` is empty, so a
      request for 语义顺滑 must blank the language field.
    """
    if has_prompt:
        return ""
    return _FUNASR_LANGUAGES.get(code, "")


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
                log.info("recognizer: sensevoice (disfluency=%s — no effect)", disfluency)
            case "funasr_nano":
                prompt = _FUNASR_NANO_PROMPTS[disfluency]
                funasr_lang = funasr_language(language, has_prompt=bool(prompt))
                self._recognizer = sherpa_onnx.OfflineRecognizer.from_funasr_nano(
                    encoder_adaptor=str(model_dir / "encoder_adaptor.int8.onnx"),
                    llm=str(model_dir / "llm.int8.onnx"),
                    embedding=str(model_dir / "embedding.int8.onnx"),
                    tokenizer=str(model_dir / "Qwen3-0.6B"),
                    num_threads=num_threads,
                    language=funasr_lang,
                    user_prompt=prompt,
                )
                if language == "ja":
                    log.warning(
                        "Fun-ASR-Nano 的日语支持较弱（训练以中英为主），"
                        "建议改用 SenseVoice-Small：识别文本更贴原话，翻译质量也会更好"
                    )
                log.info(
                    "recognizer: funasr_nano (disfluency=%s language=%r prompt=%r)",
                    disfluency,
                    funasr_lang,
                    prompt,
                )
            case unreachable:
                assert_never(unreachable)

    def transcribe(self, samples: np.ndarray) -> str:
        """Decode one utterance of float32 mono samples at 16 kHz."""
        stream = self._recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, samples)
        self._recognizer.decode_stream(stream)
        return _TAG_RE.sub("", stream.result.text).strip()

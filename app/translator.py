"""Offline neural machine translation of dictation output (ONNX, CPU only).

The product promise is "runs entirely on this machine", so translation is a
*local* Marian NMT model exported to ONNX (opus-mt-zh-en / opus-mt-en-zh by
default). Nothing here touches the network at inference time: the model is
downloaded once, then every sentence is translated on the CPU.

Design notes:

* **Sentence in, sentence out.** The engine is non-streaming (a full
  encoder/decoder pass per call). Callers already hand us a completed
  utterance — the hold-to-talk release or a VAD endpoint — which is exactly
  the right granularity: word-by-word streaming would garble word order
  between languages with different syntax.
* **Greedy decode, no beam search.** Beam 2+ roughly doubles CPU time for a
  marginal quality gain on dictation-length input; single-threaded greedy
  keeps the added latency in the 100-300 ms range on a laptop CPU.
* **No KV cache.** The decoder is run over the whole prefix each step. That is
  O(n²) in the number of generated tokens, but n is capped (96 by default) and
  it removes an entire class of shape/cache bugs — the merged-past exports are
  the most fragile part of the ONNX Marian graph. Swap in a cached decoder
  later if profiling says it matters.
* **Optional dependencies.** ``onnxruntime`` and ``sentencepiece`` are imported
  lazily, so unit tests (and users who never enable translation) never pay for
  them and the module still imports cleanly.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

log = logging.getLogger(__name__)

__all__ = [
    "BILINGUAL_SEPARATOR",
    "NmtSettings",
    "NullTranslator",
    "OnnxNmtTranslator",
    "TRANSLATE_MODES",
    "TRANSLATE_TARGETS",
    "TranslationError",
    "Translator",
    "compose_text",
    "has_translatable_text",
    "load_translator",
]

#: Bilingual output separator. A newline keeps the two lines visually paired in
#: every editor; chat boxes that submit on Enter are why it is configurable.
BILINGUAL_SEPARATOR: str = "\n"

#: Human labels used by the tray menu and config docs.
TRANSLATE_TARGETS: dict[str, str] = {"en": "英语", "zh": "中文"}

#: How the translated text reaches the cursor.
TRANSLATE_MODES: dict[str, str] = {
    "replace": "只上译文",
    "bilingual": "原文 + 译文",
}

#: Anything shorter than this is not worth a model pass ("嗯", "ok", "1").
_MIN_CHARS: int = 2

#: Marian/opus-mt default special ids. A ``translator.json`` next to the model
#: may override them (NLLB exports differ).
_DEFAULT_START_TOKEN_ID: int = 65_000  # == pad_token_id for opus-mt
_DEFAULT_EOS_TOKEN_ID: int = 0  # == </s> for opus-mt

#: File names to look for inside the model directory, in preference order.
_ENCODER_FILES: tuple[str, ...] = ("encoder_model.onnx", "encoder.onnx")
_DECODER_FILES: tuple[str, ...] = ("decoder_model.onnx", "decoder.onnx")

#: Input names the Marian ONNX graphs normally use; a graph that names them
#: differently is fed by position instead (see ``_feed``).
_ENC_IDS: str = "input_ids"
_ENC_MASK: str = "attention_mask"
_DEC_IDS: str = "input_ids"
_DEC_MASK: str = "encoder_attention_mask"
_DEC_HIDDEN: str = "encoder_hidden_states"


class TranslationError(Exception):
    """Raised when the local model cannot be loaded or run."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class Translator(Protocol):
    """Minimal contract the dictation pipeline depends on."""

    def translate(self, text: str) -> str: ...


class NullTranslator:
    """Translation disabled: always yields an empty string."""

    def translate(self, text: str) -> str:
        return ""


def has_translatable_text(text: str) -> bool:
    """Return False for fragments that are not worth a model pass.

    Punctuation-only output ("。", "..."), bare numbers and one-character
    interjections are skipped: translating them costs a full encoder/decoder
    pass and can only make the output worse.
    """
    stripped = text.strip()
    if len(stripped) < _MIN_CHARS:
        return False
    return any(ch.isalnum() for ch in stripped)


def compose_text(
    original: str,
    translated: str,
    mode: str,
    separator: str = BILINGUAL_SEPARATOR,
) -> str:
    """Build the text that is finally typed at the cursor.

    ``replace`` emits the translation alone; ``bilingual`` keeps the original
    above it. An empty translation never wins over the source: the caller
    passes the original through in that case, and this helper keeps that
    invariant (no blank line where a sentence used to be).
    """
    if not translated.strip():
        return original
    if mode == "replace":
        return translated
    if mode == "bilingual":
        return f"{original}{separator}{translated}"
    log.warning("unknown translate mode %r; falling back to replace", mode)
    return translated


@dataclass(frozen=True, slots=True)
class NmtSettings:
    """Everything needed to build a translator, resolved from config."""

    model_dir: Path
    target: str = "en"
    num_threads: int = 2
    max_new_tokens: int = 96
    start_token_id: int = _DEFAULT_START_TOKEN_ID
    eos_token_id: int = _DEFAULT_EOS_TOKEN_ID
    append_eos: bool = True


def _read_meta(model_dir: Path) -> dict[str, Any]:
    """Read the optional ``translator.json`` sidecar (absent -> defaults)."""
    path = model_dir / "translator.json"
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("translator.json unreadable (%s); using defaults", exc)
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _first_existing(model_dir: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        candidate = model_dir / name
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    return None


class OnnxNmtTranslator:
    """Greedy-decoding Marian/NLLB translator backed by onnxruntime.

    ``encoder``/``decoder``/``source_tokenizer``/``target_tokenizer`` can be
    injected for tests; in production they are loaded from *model_dir*.
    """

    def __init__(
        self,
        settings: NmtSettings,
        *,
        encoder: Any = None,
        decoder: Any = None,
        source_tokenizer: Any = None,
        target_tokenizer: Any = None,
    ) -> None:
        self._settings = settings
        self._meta = _read_meta(settings.model_dir)
        self._start_id = int(self._meta.get("start_token_id", settings.start_token_id))
        self._eos_id = int(self._meta.get("eos_token_id", settings.eos_token_id))
        self._max_new = int(self._meta.get("max_new_tokens", settings.max_new_tokens))
        self._append_eos = bool(self._meta.get("append_eos", settings.append_eos))
        self._encoder = encoder if encoder is not None else self._load_encoder()
        self._decoder = decoder if decoder is not None else self._load_decoder()
        self._source_spm = (
            source_tokenizer
            if source_tokenizer is not None
            else self._load_spm(str(self._meta.get("source_spm_file", "source.spm")))
        )
        self._target_spm = (
            target_tokenizer
            if target_tokenizer is not None
            else self._load_spm(str(self._meta.get("target_spm_file", "target.spm")))
        )
        log.info(
            "translator ready: dir=%s target=%s start=%d eos=%d max_new=%d",
            settings.model_dir,
            settings.target,
            self._start_id,
            self._eos_id,
            self._max_new,
        )

    # -- loading -------------------------------------------------------------

    def _session(self, path: Path) -> Any:
        try:
            import onnxruntime  # noqa: PLC0415 — optional, lazy
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise TranslationError(
                "缺少 onnxruntime（翻译引擎依赖），请重新安装依赖后再启用翻译"
            ) from exc
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = self._settings.num_threads
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        try:
            return onnxruntime.InferenceSession(
                str(path), sess_options=options, providers=["CPUExecutionProvider"]
            )
        except Exception as exc:  # noqa: BLE001 - ORT raises many types
            raise TranslationError(f"加载翻译模型失败：{path.name}: {exc}") from exc

    def _load_encoder(self) -> Any:
        path = _first_existing(self._settings.model_dir, _ENCODER_FILES)
        if path is None:
            raise TranslationError(f"缺少编码器模型：{self._settings.model_dir}")
        return self._session(path)

    def _load_decoder(self) -> Any:
        path = _first_existing(self._settings.model_dir, _DECODER_FILES)
        if path is None:
            raise TranslationError(f"缺少解码器模型：{self._settings.model_dir}")
        return self._session(path)

    def _load_spm(self, name: str) -> Any:
        path = self._settings.model_dir / name
        if not path.exists():
            raise TranslationError(f"缺少分词模型：{path}")
        try:
            import sentencepiece  # noqa: PLC0415 — optional, lazy
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise TranslationError(
                "缺少 sentencepiece（翻译分词依赖），请重新安装依赖后再启用翻译"
            ) from exc
        return sentencepiece.SentencePieceProcessor(model_file=str(path))

    # -- inference -----------------------------------------------------------

    def translate(self, text: str) -> str:
        """Translate one utterance; returns "" for empty/whitespace input."""
        stripped = text.strip()
        if not stripped:
            return ""
        ids = [int(i) for i in self._source_spm.encode(stripped, out_type=int)]
        if not ids:
            return ""
        if self._append_eos:
            ids.append(self._eos_id)
        source = np.asarray([ids], dtype=np.int64)
        mask = np.ones_like(source)

        try:
            hidden = self._encoder.run(None, {_ENC_IDS: source, _ENC_MASK: mask})[0]
        except Exception as exc:  # noqa: BLE001 - ORT raises many types
            raise TranslationError(f"编码失败：{exc}") from exc

        generated = [self._start_id]
        produced: list[int] = []
        for _ in range(max(1, self._max_new)):
            feed = self._decoder_feed(generated, hidden, mask)
            try:
                logits = self._decoder.run(None, feed)[0]
            except Exception as exc:  # noqa: BLE001
                raise TranslationError(f"解码失败：{exc}") from exc
            next_id = int(np.argmax(logits[0, -1]))
            if next_id == self._eos_id:
                break
            produced.append(next_id)
            generated.append(next_id)
        if not produced:
            return ""
        try:
            return str(self._target_spm.decode(produced)).strip()
        except Exception as exc:  # noqa: BLE001
            raise TranslationError(f"译文还原失败：{exc}") from exc

    def _decoder_feed(
        self, generated: list[int], hidden: Any, mask: Any
    ) -> dict[str, Any]:
        """Build the decoder feed from whatever inputs the graph declares.

        A Marian export asks for ``input_ids`` + ``encoder_attention_mask`` +
        ``encoder_hidden_states``; an older one may ask for a subset. Feeding by
        name (and skipping unknown names) keeps both working without branching
        on the export flavour.
        """
        ids = np.asarray([generated], dtype=np.int64)
        available = {
            _DEC_IDS: ids,
            _DEC_MASK: mask,
            _DEC_HIDDEN: hidden,
        }
        feed: dict[str, Any] = {}
        for item in self._decoder.get_inputs():
            value = available.get(item.name)
            if value is None:
                log.warning("decoder input %r unsupported; feeding zeros", item.name)
                shape = [d if isinstance(d, int) else 1 for d in item.shape]
                value = np.zeros(shape, dtype=np.int64)
            feed[item.name] = value
        return feed


def load_translator(settings: NmtSettings) -> Translator:
    """Build a translator, converting loader failures into TranslationError."""
    return OnnxNmtTranslator(settings)

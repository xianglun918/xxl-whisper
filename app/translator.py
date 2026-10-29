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
* **The tokenizer is not the SentencePiece model.** These ONNX exports use a
  65,001-entry Marian vocabulary (``</s>``=0, ``<unk>``=1, ``<pad>``=65000)
  whose ids do *not* match the 32,000-entry ``source.spm``/``target.spm`` ids.
  ``tokenizer.json`` (a Unigram table) is the authoritative piece→id map:
  SentencePiece turns text into *pieces*, that table turns pieces into ids, and
  the target model turns pieces back into text. Skipping this mapping produces
  fluent but completely unrelated English — verified the hard way.
* **Greedy decode + no-repeat guard.** Beam search doubles CPU time for a
  marginal gain on dictation-length input. A 3-gram repetition block removes
  the "world world, world" loops greedy decoding falls into, which is the only
  quality problem that actually shows up on short sentences.
* **No KV cache.** The decoder is re-run over the whole prefix each step. The
  ``decoder_with_past`` export is not usable here: this export's first-step
  decoder does not emit ``present.*.encoder.key``, so the cached graph's
  required encoder past cannot be produced (the merged graph crashes in its
  ``If`` branch). Cost is ~0.15 s/token on a 4-core Celeron-class NAS and
  roughly 2-3x faster on a laptop.
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
    "TRANSLATE_MODES",
    "TRANSLATE_ROUTES",
    "TRANSLATE_TARGETS",
    "ChainedTranslator",
    "MarianTokenizer",
    "NmtSettings",
    "NullTranslator",
    "OnnxNmtTranslator",
    "TranslationError",
    "Translator",
    "blocked_tokens",
    "compose_text",
    "has_translatable_text",
    "load_translator",
]

#: Bilingual output separator. A newline keeps the two lines visually paired in
#: every editor; chat boxes that submit on Enter are why it is configurable.
BILINGUAL_SEPARATOR: str = "\n"

#: Human labels used by the tray menu and config docs.
TRANSLATE_TARGETS: dict[str, str] = {"en": "英语", "zh": "中文"}

#: Supported routes, in tray order, with the human label shown in the menu.
#: "ja-zh" is chained through English (no official ja→zh Marian weight).
TRANSLATE_ROUTES: dict[str, str] = {
    "zh-en": "中 → 英（说中文，出英文）",
    "en-zh": "英 → 中（说英文，出中文）",
    "ja-zh": "日 → 中（说日语，出中文，经英文中转）",
}

#: How the translated text reaches the cursor.
TRANSLATE_MODES: dict[str, str] = {
    "replace": "只上译文",
    "bilingual": "原文 + 译文",
}

#: Anything shorter than this is not worth a model pass ("嗯", "ok", "1").
_MIN_CHARS: int = 2

#: Marian special ids (``</s>``, ``<unk>``, ``<pad>``) in this export's vocab.
_DEFAULT_START_TOKEN_ID: int = 65_000  # == <pad>, the decoder start token
_DEFAULT_EOS_TOKEN_ID: int = 0
_DEFAULT_UNK_TOKEN_ID: int = 1

#: Graph files, probed in order. The Xenova exports keep them under ``onnx/``;
#: a hand-assembled directory may put them at the root, so both are tried.
_ENCODER_FILES: tuple[str, ...] = (
    "onnx/encoder_model_int8.onnx",
    "onnx/encoder_model.onnx",
    "encoder_model_int8.onnx",
    "encoder_model.onnx",
    "encoder.onnx",
)
_DECODER_FILES: tuple[str, ...] = (
    "onnx/decoder_model_int8.onnx",
    "onnx/decoder_model.onnx",
    "decoder_model_int8.onnx",
    "decoder_model.onnx",
    "decoder.onnx",
)

#: Input names the Marian ONNX graphs normally use.
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


def blocked_tokens(generated: list[int], n: int) -> set[int]:
    """Token ids that would repeat an n-gram already present in *generated*.

    Greedy decoding on Marian falls into loops ("world world, world"); banning
    the token that completes an already-seen n-gram breaks them without the cost
    of beam search. Returns an empty set for n < 2 or a too-short history.
    """
    if n < 2 or len(generated) < n - 1:
        return set()
    context = tuple(generated[len(generated) - (n - 1):])
    banned: set[int] = set()
    for i in range(len(generated) - n + 1):
        if tuple(generated[i:i + n - 1]) == context:
            banned.add(generated[i + n - 1])
    return banned


class MarianTokenizer:
    """SentencePiece pieces + the model's own piece→id table.

    ``tokenizer.json`` carries the authoritative vocabulary for these exports
    (a Unigram table of 65,001 entries plus added tokens). SentencePiece is only
    used for segmentation (text ↔ pieces); ids come from, and go back through,
    that table.
    """

    def __init__(
        self,
        source_spm: Any,
        target_spm: Any,
        piece2id: dict[str, int],
        id2piece: dict[int, str],
        specials: set[int],
        unk_id: int = _DEFAULT_UNK_TOKEN_ID,
    ) -> None:
        self._source_spm = source_spm
        self._target_spm = target_spm
        self._piece2id = piece2id
        self._id2piece = id2piece
        self._specials = specials
        self._unk_id = unk_id
        self.unknown_pieces = 0

    @classmethod
    def load(cls, model_dir: Path, *, unk_id: int = _DEFAULT_UNK_TOKEN_ID) -> "MarianTokenizer":
        """Load the tokenizer from *model_dir*, or raise TranslationError."""
        try:
            import sentencepiece  # noqa: PLC0415 — optional, lazy
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise TranslationError(
                "缺少 sentencepiece（翻译分词依赖），请重新安装依赖后再启用翻译"
            ) from exc
        source = model_dir / "source.spm"
        target = model_dir / "target.spm"
        vocab_file = model_dir / "tokenizer.json"
        for path in (source, target, vocab_file):
            if not path.exists():
                raise TranslationError(f"缺少分词文件：{path.name}")
        try:
            data = json.loads(vocab_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TranslationError(f"tokenizer.json 无法解析：{exc}") from exc

        piece2id: dict[str, int] = {}
        id2piece: dict[int, str] = {}
        for index, row in enumerate(data.get("model", {}).get("vocab") or []):
            piece = row[0] if isinstance(row, list) else row
            piece2id.setdefault(str(piece), index)
            id2piece[index] = str(piece)
        specials: set[int] = set()
        for token in data.get("added_tokens") or []:
            content = str(token.get("content", ""))
            token_id = int(token.get("id", -1))
            piece2id.setdefault(content, token_id)
            id2piece[token_id] = content
            specials.add(token_id)
        if not piece2id:
            raise TranslationError("tokenizer.json 里没有词表（model.vocab 为空）")
        return cls(
            sentencepiece.SentencePieceProcessor(model_file=str(source)),
            sentencepiece.SentencePieceProcessor(model_file=str(target)),
            piece2id,
            id2piece,
            specials,
            unk_id=unk_id,
        )

    def encode(self, text: str) -> list[int]:
        """Text -> model ids (unknown pieces fall back to ``<unk>``)."""
        pieces = self._source_spm.encode(text, out_type=str)
        ids: list[int] = []
        for piece in pieces:
            token_id = self._piece2id.get(piece)
            if token_id is None:
                self.unknown_pieces += 1
                token_id = self._unk_id
            ids.append(token_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        """Model ids -> text, via pieces so SentencePiece handles spacing."""
        pieces = [
            self._id2piece[i]
            for i in ids
            if i in self._id2piece and i not in self._specials
        ]
        if not pieces:
            return ""
        return str(self._target_spm.decode(pieces)).strip()


@dataclass(frozen=True, slots=True)
class NmtSettings:
    """Everything needed to build a translator, resolved from config."""

    model_dir: Path
    target: str = "en"
    num_threads: int = 2
    max_new_tokens: int = 64
    no_repeat_ngram: int = 3
    start_token_id: int = _DEFAULT_START_TOKEN_ID
    eos_token_id: int = _DEFAULT_EOS_TOKEN_ID
    unk_token_id: int = _DEFAULT_UNK_TOKEN_ID


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
    """Greedy-decoding Marian translator backed by onnxruntime.

    ``encoder``/``decoder``/``tokenizer`` can be injected for tests; in
    production they are loaded from ``settings.model_dir``.
    """

    def __init__(
        self,
        settings: NmtSettings,
        *,
        encoder: Any = None,
        decoder: Any = None,
        tokenizer: Any = None,
    ) -> None:
        self._settings = settings
        self._meta = _read_meta(settings.model_dir)
        self._start_id = int(self._meta.get("start_token_id", settings.start_token_id))
        self._eos_id = int(self._meta.get("eos_token_id", settings.eos_token_id))
        self._max_new = int(self._meta.get("max_new_tokens", settings.max_new_tokens))
        self._no_repeat = int(self._meta.get("no_repeat_ngram", settings.no_repeat_ngram))
        self._append_eos = bool(self._meta.get("append_eos", True))
        self._encoder = encoder if encoder is not None else self._load_encoder()
        self._decoder = decoder if decoder is not None else self._load_decoder()
        self._tokenizer = (
            tokenizer
            if tokenizer is not None
            else MarianTokenizer.load(settings.model_dir, unk_id=settings.unk_token_id)
        )
        log.info(
            "translator ready: dir=%s target=%s start=%d eos=%d max_new=%d no_repeat=%d",
            settings.model_dir,
            settings.target,
            self._start_id,
            self._eos_id,
            self._max_new,
            self._no_repeat,
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
        override = self._meta.get("encoder_file")
        candidates = (str(override), *_ENCODER_FILES) if override else _ENCODER_FILES
        path = _first_existing(self._settings.model_dir, candidates)
        if path is None:
            raise TranslationError(f"缺少编码器模型：{self._settings.model_dir}")
        return self._session(path)

    def _load_decoder(self) -> Any:
        override = self._meta.get("decoder_file")
        candidates = (str(override), *_DECODER_FILES) if override else _DECODER_FILES
        path = _first_existing(self._settings.model_dir, candidates)
        if path is None:
            raise TranslationError(f"缺少解码器模型：{self._settings.model_dir}")
        return self._session(path)

    # -- inference -----------------------------------------------------------

    def translate(self, text: str) -> str:
        """Translate one utterance; returns "" for empty/whitespace input."""
        stripped = text.strip()
        if not stripped:
            return ""
        ids = self._tokenizer.encode(stripped)
        if not ids:
            return ""
        if self._append_eos:
            ids.append(self._eos_id)
        source = np.asarray([ids], dtype=np.int64)
        mask = np.ones_like(source)

        try:
            hidden = self._encoder.run(None, self._encoder_feed(source, mask))[0]
        except Exception as exc:  # noqa: BLE001 - ORT raises many types
            raise TranslationError(f"编码失败：{exc}") from exc

        sequence = [self._start_id]
        produced: list[int] = []
        for _ in range(max(1, self._max_new)):
            try:
                logits = self._decoder.run(None, self._decoder_feed(sequence, hidden, mask))[0]
            except Exception as exc:  # noqa: BLE001
                raise TranslationError(f"解码失败：{exc}") from exc
            row = np.array(logits[0, -1], dtype=np.float32, copy=True)
            for banned in blocked_tokens(produced, self._no_repeat):
                row[banned] = -np.inf
            next_id = int(np.argmax(row))
            if next_id == self._eos_id:
                break
            produced.append(next_id)
            sequence.append(next_id)
        if not produced:
            return ""
        return self._tokenizer.decode(produced)

    def _decoder_feed(self, sequence: list[int], hidden: Any, mask: Any) -> dict[str, Any]:
        """Build the decoder feed from whatever inputs the graph declares.

        A Marian export asks for ``input_ids`` + ``encoder_attention_mask`` +
        ``encoder_hidden_states``; an older one may ask for a subset. Feeding by
        name (and skipping unknown names) keeps both working, and onnxruntime
        rejects unexpected names outright.
        """
        available = {
            _DEC_IDS: np.asarray([sequence], dtype=np.int64),
            _DEC_MASK: mask,
            _DEC_HIDDEN: hidden,
        }
        return self._feed(self._decoder, available, role="decoder")

    def _encoder_feed(self, source: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
        """Build the encoder feed from the graph's declared inputs."""
        available = {_ENC_IDS: source, _ENC_MASK: mask}
        return self._feed(self._encoder, available, role="encoder")

    @staticmethod
    def _feed(session: Any, available: dict[str, Any], *, role: str) -> dict[str, Any]:
        feed: dict[str, Any] = {}
        for item in session.get_inputs():
            value = available.get(item.name)
            if value is None:
                log.warning("%s input %r unsupported; feeding zeros", role, item.name)
                shape = [d if isinstance(d, int) else 1 for d in item.shape]
                value = np.zeros(shape, dtype=np.int64)
            feed[item.name] = value
        return feed


def load_translator(settings: NmtSettings) -> Translator:
    """Build a translator, converting loader failures into TranslationError."""
    return OnnxNmtTranslator(settings)


class ChainedTranslator:
    """Translate through intermediate languages, one Marian pass per hop.

    Japanese has no official ``opus-mt`` ja→zh weight, so the route runs
    ja→en→zh: two models, two passes. Each hop is a normal
    :class:`OnnxNmtTranslator`, so this class only owns the sequencing and the
    "an empty hop means the whole translation failed" rule — an empty string
    makes the caller fall back to the original text instead of typing a
    half-translated sentence.
    """

    def __init__(self, stages: list[Translator]) -> None:
        if not stages:
            reason = "翻译链路为空"
            raise TranslationError(reason)
        self._stages = list(stages)

    @property
    def hops(self) -> int:
        """How many models this chain runs per sentence."""
        return len(self._stages)

    def translate(self, text: str) -> str:
        """Run every hop in order; returns "" when any hop produces nothing."""
        current = text
        for index, stage in enumerate(self._stages):
            current = stage.translate(current)
            if not current.strip():
                log.info("translate: hop %d/%d produced nothing", index + 1, len(self._stages))
                return ""
        return current

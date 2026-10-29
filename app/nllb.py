"""NLLB-200 distilled 600M as an *optional* direct 日→中 engine.

Why it exists: Marian has no official ja→zh weight, so the default route chains
ja→en→zh. English is a poor pivot for Japanese — pro-drop subjects get invented,
keigo gets flattened, names get transliterated twice — so the two-hop output is
noticeably worse than the same engine's zh↔en. NLLB translates jpn_Jpan→zho_Hans
in a single pass.

Cost: ~890 MB of int8 ONNX and ~1 GB resident, plus a decoder pass per token, so
it is opt-in rather than the default. The two-hop route stays available for
machines that cannot afford it.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Protocol

import numpy as np

log = logging.getLogger(__name__)

#: NLLB sentence pieces for the languages this build cares about.
SRC_JAPANESE: str = "jpn_Jpan"
TGT_CHINESE: str = "zho_Hans"

#: NLLB language tokens look like ``jpn_Jpan`` / ``zho_Hans``.
_LANG_TOKEN_RE = re.compile(r"^[a-z]{3}_[A-Z][a-z]{3}$")

_EOS_TOKEN: str = "</s>"  # noqa: S105 - sentence-piece token, not a secret


class TokenizerLike(Protocol):
    """The slice of ``tokenizers.Tokenizer`` this module uses."""

    def encode(self, text: str) -> Any: ...
    def decode(self, ids: list[int], skip_special_tokens: bool = ...) -> str: ...
    def token_to_id(self, token: str) -> int | None: ...


class NllbTranslator:
    """Greedy NLLB decoder over the quantized ONNX export (transformers.js layout).

    Two conventions were established by experiment against the real weights; both
    are easy to get wrong and neither is documented:

    1. The tokenizer's post-processor prepends ``eng_Latn`` and appends ``</s>``.
       Both must be removed before prepending the real source language token and
       one terminator — otherwise the encoder sees two language tokens and two
       end-of-sentence tokens.
    2. The decoder starts from ``</s>`` and the **first** emitted token is forced
       to the target language token (what ``generate(forced_bos_token_id=...)``
       does). Starting the decoder *with* the language token collapses the
       distribution into one repeated piece.
    """

    def __init__(
        self,
        model_dir: Path,
        *,
        source_lang: str = SRC_JAPANESE,
        target_lang: str = TGT_CHINESE,
        num_threads: int = 4,
        max_new_tokens: int = 64,
        session_factory: Any = None,
        tokenizer: TokenizerLike | None = None,
    ) -> None:  # noqa: PLR0913 - an engine needs its knobs visible
        import onnxruntime as ort  # noqa: PLC0415 - only when this engine is on
        import tokenizers  # noqa: PLC0415

        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, num_threads)
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        factory = session_factory or ort.InferenceSession

        self._encoder = factory(
            str(model_dir / "onnx" / "encoder_model_quantized.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        # Plain (non-merged) decoder: the cached export reshapes past tensors
        # unconditionally and a zero-length past fails that Reshape in
        # onnxruntime. Recomputing the prefix per token is slower but correct.
        self._decoder = factory(
            str(model_dir / "onnx" / "decoder_model_quantized.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._tokenizer = tokenizer or tokenizers.Tokenizer.from_file(
            str(model_dir / "tokenizer.json")
        )
        self._max_new_tokens = max_new_tokens
        self._src_id = self._require_token(source_lang)
        self._tgt_id = self._require_token(target_lang)
        self._eos_id = self._require_token(_EOS_TOKEN)
        log.info("nllb: ready (src=%s tgt=%s)", source_lang, target_lang)

    def _require_token(self, token: str) -> int:
        token_id = self._tokenizer.token_to_id(token)
        if token_id is None:
            reason = f"tokenizer is missing {token!r}"
            raise ValueError(reason)
        return int(token_id)

    # -- translation -------------------------------------------------------

    def translate(self, text: str) -> str:
        """Translate *text* from Japanese to Chinese in a single pass."""
        source = text.strip()
        if not source:
            return ""
        input_ids = np.asarray(
            [[self._src_id, *self._source_ids(source), self._eos_id]], dtype=np.int64
        )
        attention = np.ones_like(input_ids)
        hidden = self._encoder.run(
            None, {"input_ids": input_ids, "attention_mask": attention}
        )[0]

        produced = self._decode(hidden, attention)
        if not produced:
            return ""
        return self._tokenizer.decode(
            produced, skip_special_tokens=True
        ).strip()

    def _source_ids(self, text: str) -> list[int]:
        """Token ids for *text*, without the language token / terminator the export adds."""
        encoded = self._tokenizer.encode(text)
        ids = list(encoded.ids)
        tokens = list(getattr(encoded, "tokens", []))
        if ids and tokens and _LANG_TOKEN_RE.match(tokens[0]):
            ids = ids[1:]
        if ids and ids[-1] == self._eos_id:
            ids = ids[:-1]
        return ids

    def _decode(self, hidden: np.ndarray, attention: np.ndarray) -> list[int]:
        """Greedy decode following NLLB's own conventions (see class docstring)."""
        prefix = [self._eos_id]
        produced: list[int] = []
        for step in range(self._max_new_tokens):
            logits = self._decoder.run(
                None,
                {
                    "input_ids": np.asarray([prefix], dtype=np.int64),
                    "encoder_attention_mask": attention,
                    "encoder_hidden_states": hidden,
                },
            )[0]
            next_id = self._tgt_id if step == 0 else int(np.argmax(logits[0, -1]))
            if next_id == self._eos_id:
                break
            prefix.append(next_id)
            if step > 0:
                produced.append(next_id)
        return produced

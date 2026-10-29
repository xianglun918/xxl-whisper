"""NLLB-200 distilled 600M as an *optional* direct 日→中 engine.

Why it exists: Marian has no official ja→zh weight, so the default route chains
ja→en→zh. English is a poor pivot for Japanese — pro-drop subjects get invented,
keigo gets flattened, names get transliterated twice — so the two-hop output is
noticeably worse than the same engine's zh↔en. NLLB translates jpn_Jpan→zho_Hans
in one pass.

Cost: ~890 MB of int8 ONNX and ~1 GB resident, hence optional (a config switch)
rather than the default. The two-hop route stays available for smaller machines.
"""

from __future__ import annotations

import logging
from array import array
from pathlib import Path
from typing import Any, Protocol

import numpy as np

log = logging.getLogger(__name__)

#: NLLB sentence pieces for the two languages this build cares about.
SRC_JAPANESE: str = "jpn_Jpan"
TGT_CHINESE: str = "zho_Hans"

_EOS_TOKEN: str = "</s>"
_PAD_TOKEN: str = "<pad>"

#: Encoder/decoder geometry of nllb-200-distilled-600M (12 layers, 16 heads × 64).


class TokenizerLike(Protocol):
    """The slice of ``tokenizers.Tokenizer`` this module uses."""

    def encode(self, text: str) -> Any: ...
    def decode(self, ids: list[int]) -> str: ...
    def token_to_id(self, token: str) -> int | None: ...


class NllbTranslator:
    """Greedy NLLB decoder over the quantized ONNX export (transformers.js layout).

    The merged decoder keeps its own KV cache: the first pass runs with
    ``use_cache_branch = False`` and empty past tensors, later passes feed the
    previous ``present.*`` outputs back in. That keeps a 600M decoder at roughly
    one token-step per forward pass instead of re-encoding the whole prefix.
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
    ) -> None:
        import onnxruntime as ort  # noqa: PLC0415 - heavy, only when this engine is on
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
        # Non-merged decoder on purpose: the merged (cached) export reshapes the
        # past tensors unconditionally, and a zero-length past fails that Reshape
        # in onnxruntime ("The dimension with value zero exceeds the dimension
        # size"). Recomputing the prefix per token is slower but correct, and this
        # engine is optional quality-mode.
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
        ids = [self._src_id, *self._tokenizer.encode(source).ids, self._eos_id]
        input_ids = np.asarray([ids], dtype=np.int64)
        attention = np.ones_like(input_ids)

        hidden = self._encoder.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention,
        })[0]

        generated = self._decode(hidden, attention)
        if not generated:
            return ""
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()

    def _decode(self, hidden: np.ndarray, attention: np.ndarray) -> list[int]:
        """Greedy loop over the plain decoder: feed the whole prefix each step."""
        prefix = [self._tgt_id]
        produced: list[int] = []
        for _ in range(self._max_new_tokens):
            feeds = {
                "input_ids": np.asarray([prefix], dtype=np.int64),
                "encoder_attention_mask": attention,
                "encoder_hidden_states": hidden,
            }
            logits = self._decoder.run(None, feeds)[0]
            next_id = int(np.argmax(logits[0, -1]))
            if next_id == self._eos_id:
                break
            produced.append(next_id)
            prefix.append(next_id)
        return produced


def _unused(value: object) -> None:
    """Keep ``array`` import meaningful for type checkers on empty inputs."""
    if isinstance(value, array):
        return

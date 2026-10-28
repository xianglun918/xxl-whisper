"""Unit tests for the offline translation layer (no onnxruntime needed)."""

import json

import numpy as np
import pytest

from app.translator import (
    NmtSettings,
    OnnxNmtTranslator,
    TranslationError,
    compose_text,
    has_translatable_text,
)


# -- pure helpers -------------------------------------------------------------


@pytest.mark.parametrize("text", ["你好", "hello", "开个 PR", "ok"])
def test_translatable_text_accepts_words(text: str) -> None:
    assert has_translatable_text(text) is True


@pytest.mark.parametrize("text", ["", "   ", "。", "…", "嗯", "1", "！？"])
def test_translatable_text_rejects_noise(text: str) -> None:
    assert has_translatable_text(text) is False


def test_compose_replace_drops_source() -> None:
    assert compose_text("你好", "hello", "replace") == "hello"


def test_compose_bilingual_keeps_source_above() -> None:
    assert compose_text("你好", "hello", "bilingual") == "你好\nhello"


def test_compose_bilingual_honours_custom_separator() -> None:
    assert compose_text("你好", "hello", "bilingual", separator=" / ") == "你好 / hello"


def test_compose_never_drops_text_when_translation_empty() -> None:
    """An empty translation must not replace the user's words with nothing."""
    assert compose_text("你好", "", "replace") == "你好"
    assert compose_text("你好", "   ", "bilingual") == "你好"


def test_compose_unknown_mode_falls_back_to_translation() -> None:
    assert compose_text("你好", "hello", "sideways") == "hello"


# -- greedy decoding ----------------------------------------------------------


class _FakeNode:
    def __init__(self, name: str, shape: list[object] | None = None) -> None:
        self.name = name
        self.shape = shape or [1, "seq"]


class _FakeSession:
    """Minimal onnxruntime InferenceSession stand-in."""

    def __init__(self, inputs: list[str], outputs: list[np.ndarray] | None = None) -> None:
        self._inputs = [_FakeNode(name) for name in inputs]
        self._outputs = list(outputs or [])
        self.feeds: list[dict[str, np.ndarray]] = []

    def get_inputs(self) -> list[_FakeNode]:
        return self._inputs

    def run(self, _names: object, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.feeds.append(feed)
        if self._outputs:
            return [self._outputs.pop(0)]
        return [np.zeros((1, 1, 4), dtype=np.float32)]


class _FakeSpm:
    def __init__(self, ids: list[int] | None = None, text: str = "hello") -> None:
        self._ids = ids if ids is not None else [11, 12]
        self._text = text
        self.encoded: list[str] = []
        self.decoded: list[list[int]] = []

    def encode(self, text: str, out_type: object = int) -> list[int]:
        self.encoded.append(text)
        return list(self._ids)

    def decode(self, ids: object) -> str:
        self.decoded.append(list(ids))  # type: ignore[arg-type]
        return self._text


def _step_logits(vocab: int, winner: int) -> np.ndarray:
    logits = np.zeros((1, 1, vocab), dtype=np.float32)
    logits[0, 0, winner] = 10.0
    return logits


def _build(
    tmp_path: object, *, steps: list[int], spm_text: str = "hello", **kwargs: object
) -> tuple[OnnxNmtTranslator, _FakeSpm, _FakeSession, _FakeSession]:
    src = _FakeSpm(ids=[11, 12], text=spm_text)
    tgt = _FakeSpm(ids=[11, 12], text=spm_text)
    encoder = _FakeSession(["input_ids", "attention_mask"], [np.zeros((1, 2, 8), np.float32)])
    decoder = _FakeSession(
        ["input_ids", "encoder_attention_mask", "encoder_hidden_states"],
        [_step_logits(16, winner) for winner in steps],
    )
    settings = NmtSettings(model_dir=tmp_path, target="en")  # type: ignore[arg-type]
    translator = OnnxNmtTranslator(
        settings,
        encoder=encoder,
        decoder=decoder,
        source_tokenizer=src,
        target_tokenizer=tgt,
        **kwargs,  # type: ignore[arg-type]
    )
    return translator, src, encoder, decoder


def test_greedy_decode_stops_at_eos(tmp_path: object) -> None:
    translator, src, _encoder, decoder = _build(tmp_path, steps=[7, 8, 0, 9])

    assert translator.translate(" 你好 ") == "hello"
    assert src.encoded == ["你好"]  # whitespace trimmed before tokenising
    assert src.encode("你好") == [11, 12]
    # EOS (0) must stop decoding: the 4th programmed step never runs.
    assert len(decoder.feeds) == 3


def test_greedy_decode_appends_eos_to_source(tmp_path: object) -> None:
    translator, _src, encoder, _decoder = _build(tmp_path, steps=[7, 0])

    translator.translate("你好")
    encoder_feeds = encoder.feeds
    assert encoder_feeds[0]["input_ids"].tolist() == [[11, 12, 0]]  # eos appended
    assert encoder_feeds[0]["attention_mask"].tolist() == [[1, 1, 1]]


def test_greedy_decode_respects_max_new_tokens(tmp_path: object) -> None:
    meta = {"max_new_tokens": 2, "start_token_id": 65000, "eos_token_id": 0}
    (tmp_path / "translator.json").write_text(json.dumps(meta), encoding="utf-8")  # type: ignore[operator]
    translator, _src, _encoder, decoder = _build(tmp_path, steps=[7, 8, 9, 0])

    translator.translate("你好")
    assert len(decoder.feeds) == 2  # capped by translator.json, no EOS within budget


def test_decoder_feed_skips_unknown_inputs(tmp_path: object) -> None:
    src = _FakeSpm()
    encoder = _FakeSession(["input_ids", "attention_mask"], [np.zeros((1, 2, 8), np.float32)])
    decoder = _FakeSession(["input_ids", "mystery_input"], [_step_logits(16, 0)])
    translator = OnnxNmtTranslator(
        NmtSettings(model_dir=tmp_path),  # type: ignore[arg-type]
        encoder=encoder,
        decoder=decoder,
        source_tokenizer=src,
        target_tokenizer=src,
    )

    assert translator.translate("你好") == ""
    assert "mystery_input" in decoder.feeds[0]  # filled with zeros, not crashed


def test_empty_input_short_circuits(tmp_path: object) -> None:
    translator, src, encoder, _decoder = _build(tmp_path, steps=[0])

    assert translator.translate("   ") == ""
    assert src.encoded == []
    assert encoder.feeds == []


def test_missing_model_files_raise(tmp_path: object) -> None:
    with pytest.raises(TranslationError):
        OnnxNmtTranslator(NmtSettings(model_dir=tmp_path))  # type: ignore[arg-type]

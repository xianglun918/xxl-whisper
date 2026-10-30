"""Unit tests for the offline translation layer (no onnxruntime needed)."""

import json

import numpy as np
import pytest
from app.translator import (
    MarianTokenizer,
    NmtSettings,
    OnnxNmtTranslator,
    TranslationError,
    blocked_tokens,
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


# -- no-repeat guard ----------------------------------------------------------


def test_blocked_tokens_blocks_the_repeating_completion() -> None:
    # "a b" appeared before "a b c", so "c" is banned after the second "a b".
    generated = [10, 20, 30, 40, 10, 20]
    assert blocked_tokens(generated, 3) == {30}


def test_blocked_tokens_allows_fresh_continuations() -> None:
    assert blocked_tokens([10, 20, 30, 40], 3) == set()


def test_blocked_tokens_is_inert_for_short_history() -> None:
    assert blocked_tokens([10], 3) == set()
    assert blocked_tokens([10, 20], 2) == set()


def test_blocked_tokens_disabled_by_n_less_than_two() -> None:
    assert blocked_tokens([1, 2, 3, 1, 2], 1) == set()


# -- tokenizer ----------------------------------------------------------------


class _FakeSpm:
    def __init__(self, pieces: list[str] | None = None, decoded: str = "hello world") -> None:
        self._pieces = pieces if pieces is not None else ["▁你好", "，"]
        self._decoded = decoded
        self.decoded_pieces: list[list[str]] = []

    def encode(self, text: str, out_type: object = str) -> list[str]:
        return list(self._pieces)

    def decode(self, pieces: object) -> str:
        self.decoded_pieces.append(list(pieces))  # type: ignore[arg-type]
        return self._decoded


def _tokenizer(piece2id: dict[str, int] | None = None, decoded: str = "hello world") -> MarianTokenizer:
    pieces = {"▁你好": 5349, "，": 2, "世界": 907}
    return MarianTokenizer(
        _FakeSpm(["▁你好", "，"], decoded=decoded),
        _FakeSpm(decoded=decoded),
        piece2id if piece2id is not None else pieces,
        {5: "▁Hello", 6: "▁world", 0: "</s>", 65000: "<pad>"},
        {0, 65000},
    )


def test_tokenizer_maps_pieces_to_model_ids() -> None:
    """The spm ids (32k vocab) are NOT the model ids (65k vocab)."""
    tokenizer = _tokenizer()
    assert tokenizer.encode("你好，") == [5349, 2]


def test_tokenizer_falls_back_to_unk_for_unknown_pieces() -> None:
    tokenizer = _tokenizer(piece2id={"▁你好": 5349})
    assert tokenizer.encode("你好，") == [5349, 1]
    assert tokenizer.unknown_pieces == 1


def test_tokenizer_decode_drops_special_tokens() -> None:
    fake = _FakeSpm(decoded="Hello world")
    tokenizer = MarianTokenizer(fake, fake, {}, {5: "▁Hello", 6: "▁world", 0: "</s>"}, {0})
    assert tokenizer.decode([5, 6, 0]) == "Hello world"
    assert fake.decoded_pieces[0] == ["▁Hello", "▁world"]


def test_tokenizer_decode_returns_empty_for_only_specials() -> None:
    tokenizer = _tokenizer()
    assert tokenizer.decode([0, 65000]) == ""


def test_tokenizer_load_reports_missing_files(tmp_path: object) -> None:
    with pytest.raises(TranslationError):
        MarianTokenizer.load(tmp_path)  # type: ignore[arg-type]


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


def _step_logits(vocab: int, winner: int) -> np.ndarray:
    logits = np.zeros((1, 1, vocab), dtype=np.float32)
    logits[0, 0, winner] = 10.0
    return logits


def _build(
    tmp_path: object, *, steps: list[int], decoded: str = "hello"
) -> tuple[OnnxNmtTranslator, _FakeSession, _FakeSession]:
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
        tokenizer=_tokenizer(decoded=decoded),
    )
    return translator, encoder, decoder


def test_greedy_decode_stops_at_eos(tmp_path: object) -> None:
    translator, _encoder, decoder = _build(tmp_path, steps=[5, 6, 0, 9])

    assert translator.translate(" 你好 ") == "hello"
    # EOS (0) must stop decoding: the 4th programmed step never runs.
    assert len(decoder.feeds) == 3


def test_greedy_decode_appends_eos_to_source(tmp_path: object) -> None:
    translator, encoder, _decoder = _build(tmp_path, steps=[7, 0])

    translator.translate("你好")
    assert encoder.feeds[0]["input_ids"].tolist() == [[5349, 2, 0]]  # eos appended
    assert encoder.feeds[0]["attention_mask"].tolist() == [[1, 1, 1]]


def test_greedy_decode_respects_max_new_tokens(tmp_path: object) -> None:
    (tmp_path / "translator.json").write_text(  # type: ignore[operator]
        json.dumps({"max_new_tokens": 2, "no_repeat_ngram": 0}), encoding="utf-8"
    )
    translator, _encoder, decoder = _build(tmp_path, steps=[7, 8, 9, 0])

    translator.translate("你好")
    assert len(decoder.feeds) == 2  # capped by translator.json, no EOS within budget


def test_decoder_feed_skips_unknown_inputs(tmp_path: object) -> None:
    encoder = _FakeSession(["input_ids", "attention_mask"], [np.zeros((1, 2, 8), np.float32)])
    decoder = _FakeSession(["input_ids", "mystery_input"], [_step_logits(16, 0)])
    translator = OnnxNmtTranslator(
        NmtSettings(model_dir=tmp_path),  # type: ignore[arg-type]
        encoder=encoder,
        decoder=decoder,
        tokenizer=_tokenizer(decoded=""),
    )

    assert translator.translate("你好") == ""
    assert "mystery_input" in decoder.feeds[0]  # filled with zeros, not crashed


def test_encoder_feed_only_declared_inputs(tmp_path: object) -> None:
    """An export that declares only input_ids must not receive a mask."""
    encoder = _FakeSession(["input_ids"], [np.zeros((1, 2, 8), np.float32)])
    decoder = _FakeSession(["input_ids", "encoder_attention_mask"], [_step_logits(16, 0)])
    translator = OnnxNmtTranslator(
        NmtSettings(model_dir=tmp_path),  # type: ignore[arg-type]
        encoder=encoder,
        decoder=decoder,
        tokenizer=_tokenizer(decoded=""),
    )

    translator.translate("你好")
    assert set(encoder.feeds[0]) == {"input_ids"}


def test_empty_input_short_circuits(tmp_path: object) -> None:
    translator, encoder, _decoder = _build(tmp_path, steps=[0])

    assert translator.translate("   ") == ""
    assert encoder.feeds == []


def test_missing_model_files_raise(tmp_path: object) -> None:
    with pytest.raises(TranslationError):
        OnnxNmtTranslator(NmtSettings(model_dir=tmp_path))  # type: ignore[arg-type]


def test_first_existing_prefers_int8_under_onnx(tmp_path: object) -> None:
    """The downloader writes graphs to <model>/onnx/*_int8.onnx; find them there."""
    from app.translator import _ENCODER_FILES, _first_existing

    model_dir = tmp_path / "opus-mt-zh-en"  # type: ignore[operator]
    (model_dir / "onnx").mkdir(parents=True)
    graph = model_dir / "onnx" / "encoder_model_int8.onnx"
    graph.write_bytes(b"x" * 32)

    assert _first_existing(model_dir, _ENCODER_FILES) == graph

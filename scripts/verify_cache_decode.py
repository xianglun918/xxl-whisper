"""Phase-1 (no cache) + phase-2 (with past) decoding benchmark."""

import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import sentencepiece

ROOT = Path("/tmp/mtmodels/opus-mt-zh-en")


def load(path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.log_severity_level = 3
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


class Vocab:
    def __init__(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        self.piece2id: dict[str, int] = {}
        self.id2piece: dict[int, str] = {}
        for idx, row in enumerate(data["model"]["vocab"]):
            piece = row[0] if isinstance(row, list) else row
            self.piece2id.setdefault(piece, idx)
            self.id2piece[idx] = piece
        self.specials: set[int] = set()
        for tok in data.get("added_tokens", []):
            self.piece2id.setdefault(tok["content"], tok["id"])
            self.id2piece[tok["id"]] = tok["content"]
            self.specials.add(tok["id"])


ENC = load(ROOT / "onnx/encoder_model_int8.onnx")
DEC = load(ROOT / "onnx/decoder_model_int8.onnx")
DEC_PAST = load(ROOT / "onnx/decoder_with_past_model_int8.onnx")
SRC = sentencepiece.SentencePieceProcessor(model_file=str(ROOT / "source.spm"))
TGT = sentencepiece.SentencePieceProcessor(model_file=str(ROOT / "target.spm"))
VOCAB = Vocab(ROOT / "tokenizer.json")


def blocked(generated: list[int], n: int) -> set[int]:
    """Tokens that would repeat an n-gram already present in the output."""
    if len(generated) < n - 1:
        return set()
    ctx = tuple(generated[-(n - 1):])
    out: set[int] = set()
    for i in range(len(generated) - n + 1):
        if tuple(generated[i:i + n - 1]) == ctx:
            out.add(generated[i + n - 1])
    return out


def translate(text: str, no_repeat: int = 3, max_new: int = 96) -> tuple[str, float, int]:
    ids = [VOCAB.piece2id.get(p, 1) for p in SRC.encode(text, out_type=str)] + [0]
    source = np.asarray([ids], dtype=np.int64)
    mask = np.ones_like(source)
    t0 = time.monotonic()
    hidden = ENC.run(None, {"input_ids": source, "attention_mask": mask})[0]

    seq = [65000]
    produced: list[int] = []
    outputs = DEC.run(None, {
        "input_ids": np.asarray([seq], dtype=np.int64),
        "encoder_attention_mask": mask,
        "encoder_hidden_states": hidden,
    })
    names = [o.name for o in DEC.get_outputs()]
    past = {("past_key_values." + n[len("present."):]): v for n, v in zip(names, outputs) if n.startswith("present.")}
    logits = outputs[0]

    for _ in range(max_new):
        banned = blocked(produced, no_repeat) if no_repeat > 1 else set()
        row = logits[0, -1].copy()
        for bad in banned:
            row[bad] = -1e9
        nxt = int(np.argmax(row))
        if nxt == 0:
            break
        produced.append(nxt)
        seq.append(nxt)
        feed = {
            "input_ids": np.asarray([[nxt]], dtype=np.int64),
            "encoder_attention_mask": mask,
        }
        for item in DEC_PAST.get_inputs():
            if item.name in past:
                feed[item.name] = past[item.name]
        out2 = DEC_PAST.run(None, feed)
        names2 = [o.name for o in DEC_PAST.get_outputs()]
        past = {("past_key_values." + n[len("present."):]): v for n, v in zip(names2, out2) if n.startswith("present.")}
        logits = out2[0]

    pieces = [VOCAB.id2piece[i] for i in produced if i in VOCAB.id2piece]
    return TGT.decode(pieces), time.monotonic() - t0, len(produced)


def main() -> None:
    for text in ["你好，世界。", "今天天气不错，我们下午三点开会。", "这个功能必须完全离线运行，不能调用云端接口。", "开个 PR 我看看。"]:
        out, secs, n = translate(text)
        print(f"[{secs:.2f}s / {n} tok] {text} -> {out!r}", flush=True)


if __name__ == "__main__":
    main()

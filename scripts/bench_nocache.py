"""Benchmark no-cache greedy decoding with the correct Marian tokenizer."""

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
SRC = sentencepiece.SentencePieceProcessor(model_file=str(ROOT / "source.spm"))
TGT = sentencepiece.SentencePieceProcessor(model_file=str(ROOT / "target.spm"))
VOCAB = Vocab(ROOT / "tokenizer.json")


def blocked(generated: list[int], n: int) -> set[int]:
    if n < 2 or len(generated) < n - 1:
        return set()
    ctx = tuple(generated[-(n - 1):])
    out: set[int] = set()
    for i in range(len(generated) - n + 1):
        if tuple(generated[i:i + n - 1]) == ctx:
            out.add(generated[i + n - 1])
    return out


def translate(text: str, no_repeat: int = 3, max_new: int = 64) -> tuple[str, float, int]:
    ids = [VOCAB.piece2id.get(p, 1) for p in SRC.encode(text, out_type=str)] + [0]
    source = np.asarray([ids], dtype=np.int64)
    mask = np.ones_like(source)
    t0 = time.monotonic()
    hidden = ENC.run(None, {"input_ids": source, "attention_mask": mask})[0]

    seq = [65000]
    produced: list[int] = []
    for _ in range(max_new):
        logits = DEC.run(None, {
            "input_ids": np.asarray([seq], dtype=np.int64),
            "encoder_attention_mask": mask,
            "encoder_hidden_states": hidden,
        })[0]
        row = logits[0, -1].copy()
        for bad in blocked(produced, no_repeat):
            row[bad] = -1e9
        nxt = int(np.argmax(row))
        if nxt == 0:
            break
        produced.append(nxt)
        seq.append(nxt)

    pieces = [VOCAB.id2piece[i] for i in produced if i in VOCAB.id2piece]
    return TGT.decode(pieces), time.monotonic() - t0, len(produced)


def main() -> None:
    sentences = [
        "你好，世界。",
        "今天天气不错，我们下午三点开会。",
        "这个功能必须完全离线运行，不能调用云端接口。",
        "开个 PR 我看看。",
        "他说他明天不来公司了，让我们自己决定。",
    ]
    for text in sentences:
        out, secs, n = translate(text)
        print(f"[{secs:.2f}s / {n} tok] {text} -> {out!r}", flush=True)


if __name__ == "__main__":
    main()

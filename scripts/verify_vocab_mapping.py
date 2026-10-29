"""Verify a minimal Marian tokenizer: spm pieces <-> tokenizer.json Unigram ids."""

import json
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
        entries = data["model"]["vocab"]  # Unigram: [[piece, score], ...]
        self.piece2id: dict[str, int] = {}
        self.id2piece: dict[int, str] = {}
        for idx, row in enumerate(entries):
            piece = row[0] if isinstance(row, list) else row
            self.piece2id.setdefault(piece, idx)
            self.id2piece[idx] = piece
        for tok in data.get("added_tokens", []):
            self.piece2id.setdefault(tok["content"], tok["id"])
            self.id2piece[tok["id"]] = tok["content"]
        self.specials = {t["id"] for t in data.get("added_tokens", [])}


def main() -> None:
    enc = load(ROOT / "onnx/encoder_model_int8.onnx")
    dec = load(ROOT / "onnx/decoder_model_int8.onnx")
    spm = sentencepiece.SentencePieceProcessor(model_file=str(ROOT / "source.spm"))
    tgt_spm = sentencepiece.SentencePieceProcessor(model_file=str(ROOT / "target.spm"))
    vocab = Vocab(ROOT / "tokenizer.json")

    text = "你好，世界。"
    pieces = spm.encode(text, out_type=str)
    ids = [vocab.piece2id.get(p, 1) for p in pieces]
    print("pieces:", pieces, "ids:", ids, "unk?", 1 in ids)

    source = np.asarray([ids], dtype=np.int64)
    mask = np.ones_like(source)
    hidden = enc.run(None, {"input_ids": source, "attention_mask": mask})[0]

    seq = [65000]
    produced: list[int] = []
    for _ in range(96):
        logits = dec.run(None, {
            "input_ids": np.asarray([seq], dtype=np.int64),
            "encoder_attention_mask": mask,
            "encoder_hidden_states": hidden,
        })[0]
        nxt = int(np.argmax(logits[0, -1]))
        if nxt == 0:
            break
        produced.append(nxt)
        seq.append(nxt)

    out_pieces = [vocab.id2piece[i] for i in produced if i in vocab.id2piece and i not in vocab.specials]
    print("out pieces:", out_pieces)
    print("TEXT:", repr(tgt_spm.decode(out_pieces)))


if __name__ == "__main__":
    main()

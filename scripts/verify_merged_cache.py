"""Try the merged decoder (single 60MB int8 graph) with KV cache."""

import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import sentencepiece

ROOT = Path("/tmp/mtmodels/opus-mt-zh-en")
HEADS, DIM = 8, 64
LAYERS = 6


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
        self.specials = set()
        for tok in data.get("added_tokens", []):
            self.piece2id.setdefault(tok["content"], tok["id"])
            self.id2piece[tok["id"]] = tok["content"]
            self.specials.add(tok["id"])


def translate(text: str) -> tuple[str, float, int]:
    enc = load(ROOT / "onnx/encoder_model_int8.onnx")
    dec = load(ROOT / "onnx/decoder_model_merged_int8.onnx")
    src_spm = sentencepiece.SentencePieceProcessor(model_file=str(ROOT / "source.spm"))
    tgt_spm = sentencepiece.SentencePieceProcessor(model_file=str(ROOT / "target.spm"))
    vocab = Vocab(ROOT / "tokenizer.json")

    ids = [vocab.piece2id.get(p, 1) for p in src_spm.encode(text, out_type=str)] + [0]
    source = np.asarray([ids], dtype=np.int64)
    mask = np.ones_like(source)
    seq_len = source.shape[1]
    t0 = time.monotonic()
    hidden = enc.run(None, {"input_ids": source, "attention_mask": mask})[0]

    past = {}
    for layer in range(LAYERS):
        past[f"past_key_values.{layer}.decoder.key"] = np.zeros((1, HEADS, 0, DIM), dtype=np.float32)
        past[f"past_key_values.{layer}.decoder.value"] = np.zeros((1, HEADS, 0, DIM), dtype=np.float32)
        past[f"past_key_values.{layer}.encoder.key"] = np.zeros((1, HEADS, 0, DIM), dtype=np.float32)
        past[f"past_key_values.{layer}.encoder.value"] = np.zeros((1, HEADS, 0, DIM), dtype=np.float32)

    tokens = [65000]
    produced: list[int] = []
    use_cache = False
    for step in range(96):
        feed = {
            "input_ids": np.asarray([[tokens[-1] if use_cache else tokens[0]]], dtype=np.int64),
            "encoder_attention_mask": mask,
            "encoder_hidden_states": hidden,
            "use_cache_branch": np.asarray([use_cache]),
        }
        feed.update(past)
        names = [o.name for o in dec.get_outputs()]
        outs = dec.run(None, feed)
        logits = outs[0]
        past = {n: o for n, o in zip(names, outs) if n.startswith("present.")}
        past = {k.replace("present.", "past_key_values."): v for k, v in past.items()}
        nxt = int(np.argmax(logits[0, -1]))
        if nxt == 0:
            break
        produced.append(nxt)
        tokens.append(nxt)
        use_cache = True
    elapsed = time.monotonic() - t0
    pieces = [vocab.id2piece[i] for i in produced if i in vocab.id2piece]
    return tgt_spm.decode(pieces), elapsed, len(produced)


def main() -> None:
    for text in ["你好，世界。", "今天天气不错，我们下午三点开会。", "这个功能必须完全离线运行，不能调用云端接口。", "开个 PR 我看看。"]:
        out, secs, n = translate(text)
        print(f"[{secs:.2f}s / {n} tok] {text} -> {out!r}", flush=True)


if __name__ == "__main__":
    main()

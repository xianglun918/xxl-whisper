"""Is the decoder actually using the encoder states? Vary them and compare."""

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


def encode(enc: ort.InferenceSession, spm: sentencepiece.SentencePieceProcessor, text: str) -> tuple[np.ndarray, np.ndarray]:
    ids = spm.encode(text, out_type=int) + [0]
    source = np.asarray([ids], dtype=np.int64)
    mask = np.ones_like(source)
    hidden = enc.run(None, {"input_ids": source, "attention_mask": mask})[0]
    return hidden, mask


def greedy(dec: ort.InferenceSession, hidden: np.ndarray, mask: np.ndarray, start: int = 65000,
           steps: int = 10) -> list[int]:
    seq = [start]
    out: list[int] = []
    for _ in range(steps):
        logits = dec.run(None, {
            "input_ids": np.asarray([seq], dtype=np.int64),
            "encoder_attention_mask": mask,
            "encoder_hidden_states": hidden,
        })[0]
        nxt = int(np.argmax(logits[0, -1]))
        if nxt == 0:
            break
        out.append(nxt)
        seq.append(nxt)
    return out


def main() -> None:
    enc = load(ROOT / "onnx/encoder_model.onnx")
    dec = load(ROOT / "onnx/decoder_model.onnx")
    src = sentencepiece.SentencePieceProcessor(model_file=str(ROOT / "source.spm"))
    tgt = sentencepiece.SentencePieceProcessor(model_file=str(ROOT / "target.spm"))

    h_zh, mask_zh = encode(enc, src, "你好，世界。")
    h_other, mask_other = encode(enc, src, "我们明天下午三点开会。")
    print("hidden stats:", h_zh.shape, "mean", float(h_zh.mean()), "std", float(h_zh.std()), flush=True)

    cases = {
        "real hidden": (h_zh, mask_zh),
        "zeros hidden": (np.zeros_like(h_zh), mask_zh),
        "other sentence": (h_other, mask_other),
    }
    for tag, (h, m) in cases.items():
        ids = greedy(dec, h, m)
        print(f"{tag:15} ids={ids} text={tgt.decode(ids)!r}", flush=True)

    # decoder self-conditioning: ignore encoder entirely by feeding start twice?
    for start in (65000, 0):
        ids = greedy(dec, h_zh, mask_zh, start=start)
        print(f"start={start:<6} ids={ids} text={tgt.decode(ids)!r}", flush=True)


if __name__ == "__main__":
    main()

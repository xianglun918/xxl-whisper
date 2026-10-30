"""Debug the ONNX Marian pipeline: tokenisation, ids, and fp32 vs int8 graphs."""

import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import sentencepiece

ROOT = Path("/tmp/mtmodels/opus-mt-zh-en")
TEXT = "你好，世界。"


def load(path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.log_severity_level = 3
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def describe(sess: ort.InferenceSession, tag: str) -> None:
    print(f"[{tag}] inputs:", [(i.name, i.shape, i.type) for i in sess.get_inputs()])
    print(f"[{tag}] outputs:", [(o.name, o.shape, o.type) for o in sess.get_outputs()])


def run(enc_path: Path, dec_path: Path, text: str, start_id: int, eos_id: int, append_eos: bool, steps: int = 12) -> str:
    enc = load(enc_path)
    dec = load(dec_path)
    src_spm = sentencepiece.SentencePieceProcessor(model_file=str(ROOT / "source.spm"))
    tgt_spm = sentencepiece.SentencePieceProcessor(model_file=str(ROOT / "target.spm"))

    ids = src_spm.encode(text, out_type=int)
    if append_eos:
        ids = ids + [eos_id]
    source = np.asarray([ids], dtype=np.int64)
    mask = np.ones_like(source)
    hidden = enc.run(None, {"input_ids": source, "attention_mask": mask})[0]

    seq = [start_id]
    produced: list[int] = []
    for _ in range(steps):
        feed = {
            "input_ids": np.asarray([seq], dtype=np.int64),
            "encoder_attention_mask": mask,
            "encoder_hidden_states": hidden,
        }
        logits = dec.run(None, feed)[0]
        nxt = int(np.argmax(logits[0, -1]))
        if nxt == eos_id:
            break
        produced.append(nxt)
        seq.append(nxt)
    return f"src_ids={ids} out_ids={produced} text={tgt_spm.decode(produced)!r}"


def main() -> None:
    print("spm src piece of text:", sentencepiece.SentencePieceProcessor(model_file=str(ROOT / "source.spm")).encode(TEXT, out_type=str))

    enc_i8 = ROOT / "onnx" / "encoder_model_int8.onnx"
    dec_i8 = ROOT / "onnx" / "decoder_model_int8.onnx"
    describe(load(enc_i8), "enc-int8")
    describe(load(dec_i8), "dec-int8")

    t0 = time.monotonic()
    print("int8  :", run(enc_i8, dec_i8, TEXT, 65000, 0, True))
    print(f"  ({time.monotonic()-t0:.2f}s)")


if __name__ == "__main__":
    main()

"""Verify the ONNX Marian pipeline with the real HF tokenizer.json."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

ROOT = Path("/tmp/mtmodels/opus-mt-zh-en")
TEXT = "你好，世界。"


def load(path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.log_severity_level = 3
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def translate(text: str, enc_name: str = "encoder_model_int8.onnx", dec_name: str = "decoder_model_int8.onnx") -> str:
    enc = load(ROOT / "onnx" / enc_name)
    dec = load(ROOT / "onnx" / dec_name)
    tok = Tokenizer.from_file(str(ROOT / "tokenizer.json"))

    encoding = tok.encode(text)
    ids = list(encoding.ids)
    print("  tokenized ids:", ids, "->", tok.decode(ids), flush=True)

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
        if nxt == 0:  # </s>
            break
        produced.append(nxt)
        seq.append(nxt)
    return tok.decode(produced)


def main() -> None:
    for text in ["你好，世界。", "今天天气不错，我们下午三点开会。", "这个功能必须完全离线运行。"]:
        print(text, "->", repr(translate(text)), flush=True)


if __name__ == "__main__":
    main()

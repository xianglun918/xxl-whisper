"""Compare fp32 vs int8 graph combinations for the ONNX Marian pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from debug_translate import run

ROOT = Path("/tmp/mtmodels/opus-mt-zh-en/onnx")
TEXT = "你好，世界。"

COMBOS = [
    ("fp32enc + fp32dec", ROOT / "encoder_model.onnx", ROOT / "decoder_model.onnx"),
    ("int8enc + fp32dec", ROOT / "encoder_model_int8.onnx", ROOT / "decoder_model.onnx"),
    ("fp32enc + int8dec", ROOT / "encoder_model.onnx", ROOT / "decoder_model_int8.onnx"),
]


def main() -> None:
    for tag, enc, dec in COMBOS:
        try:
            print(tag, "->", run(enc, dec, TEXT, 65000, 0, True), flush=True)
        except Exception as exc:  # noqa: BLE001
            print(tag, "FAILED", type(exc).__name__, exc, flush=True)


if __name__ == "__main__":
    main()

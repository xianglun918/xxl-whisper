"""End-to-end check: download the local NMT model, load it, translate sentences.

    uv run python scripts/probe_translate.py [en|zh]

``en`` exercises 中→英 (opus-mt-zh-en), ``zh`` exercises 英→中 (opus-mt-en-zh).
It downloads the model if needed, prints per-sentence latency, and is the
quickest way to confirm a machine can actually run offline translation.
"""

import logging
import sys
import time
from pathlib import Path

from app.downloader import ensure_translate_model
from app.translator import NmtSettings, compose_text, load_translator

TARGET = sys.argv[1] if len(sys.argv) > 1 else "en"
ROOT = Path(f"/tmp/translate-probe-{TARGET}")

ZH_SENTENCES = [
    "你好，世界。",
    "今天天气不错，我们下午三点开会。",
    "这个功能必须完全离线运行，不能调用云端接口。",
    "开个 PR 我看看。",
]
EN_SENTENCES = [
    "Hello, world.",
    "The meeting is moved to three o'clock this afternoon.",
    "This feature must run completely offline and never call a cloud API.",
]

log = logging.getLogger("probe")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    started = time.monotonic()
    files = ensure_translate_model(TARGET, ROOT, lambda name, done, total: None)
    log.info("model ready in %.0fs at %s", time.monotonic() - started, files.directory)

    started = time.monotonic()
    translator = load_translator(
        NmtSettings(model_dir=files.directory, target=TARGET, num_threads=4)
    )
    log.info("translator loaded in %.1fs", time.monotonic() - started)

    sentences = ZH_SENTENCES if TARGET == "en" else EN_SENTENCES
    for text in sentences:
        began = time.monotonic()
        out = translator.translate(text)
        print(f"[{time.monotonic() - began:.2f}s] {text} -> {out}", flush=True)
    first = translator.translate(sentences[0])
    print("bilingual:", repr(compose_text(sentences[0], first, "bilingual")))


if __name__ == "__main__":
    main()

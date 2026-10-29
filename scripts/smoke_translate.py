"""Smoke test: download the local NMT model, then translate real sentences."""

import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

from app.downloader import ensure_translate_model
from app.translator import NmtSettings, OnnxNmtTranslator, compose_text

ROOT = Path("/tmp/mtmodels")
state = {"t": 0.0, "pct": -1}


def progress(name: str, done: int, total: int) -> None:
    pct = int(done * 100 / total) if total else 0
    now = time.monotonic()
    if pct >= state["pct"] + 10 or now - state["t"] > 20:
        state["pct"] = pct
        state["t"] = now
        print(f"  {Path(name).name}: {done / 1048576:.0f}/{total / 1048576:.0f} MB ({pct}%)", flush=True)


def main() -> None:
    t0 = time.monotonic()
    files = ensure_translate_model("en", ROOT, progress)
    print(f"MODEL READY in {time.monotonic() - t0:.0f}s -> {files.directory}", flush=True)

    t1 = time.monotonic()
    translator = OnnxNmtTranslator(NmtSettings(model_dir=files.directory, target="en", num_threads=4))
    print(f"LOADED in {time.monotonic() - t1:.1f}s", flush=True)

    sentences = [
        "你好，世界。",
        "今天天气不错，我们下午三点开会。",
        "这个功能必须完全离线运行，不能调用云端接口。",
        "开个 PR 我看看。",
    ]
    for text in sentences:
        start = time.monotonic()
        out = translator.translate(text)
        print(f"[{time.monotonic() - start:.2f}s] {text} -> {out}", flush=True)
    print("bilingual:", repr(compose_text(sentences[0], translator.translate(sentences[0]), "bilingual")))
    print("SMOKE DONE")


if __name__ == "__main__":
    main()

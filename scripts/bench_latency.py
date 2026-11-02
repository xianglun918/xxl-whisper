#!/usr/bin/env python3
"""端到端延迟基准：把"端点检测→上屏"这条路变成可对比的数字。

豆包评审（2026-11-02）的第 4 条意见就是这件事：代码里有
``continuous: endpoint→shown %.3f s`` 这样的日志，但没有人把数字沉淀成基准，
所以"这版是不是变慢了"永远只能靠感觉。

两种用法：

1. 压纯逻辑层（任何机器都能跑，不需要模型/麦克风）：

       python scripts/bench_latency.py

   测的是 TypeBatcher 粘句、EmitQueue 入队/出队、两者在背压下的一次完整
   "句子进→句子出"。这些是每条字幕都要走的必经之路，涨上去就是每条字幕都慢。

2. 读真机日志，给出真实延迟分布（在有 app.log 的机器上跑）：

       python scripts/bench_latency.py --log %LOCALAPPDATA%/xxl-whisper/logs/app.log

   解析 ``continuous: endpoint→shown`` / ``endpoint→emit`` 两类行，输出
   count / p50 / p95 / max，并可 ``--json out.json`` 存成基线。

输出的 JSON 字段固定（op → {count, p50_ms, p95_ms, max_ms}），方便两次运行做 diff。
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path

# 直接 `python scripts/bench_latency.py` 时 sys.path[0] 是 scripts/，要把仓库根加回来。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.emit_queue import EmitQueue
from app.type_batcher import TypeBatcher

# ``continuous: endpoint→shown 0.412 s (queued 0.11 s, forced=False)``
_SHOWN_RE = re.compile(r"endpoint→shown ([\d.]+) s .*?queued ([\d.]+) s", re.IGNORECASE)
# ``continuous: endpoint→emit 0.523 s`` (older builds)
_EMIT_RE = re.compile(r"endpoint→emit ([\d.]+) s", re.IGNORECASE)


def _percentiles(samples_ms: list[float]) -> dict[str, float | int]:
    if not samples_ms:
        return {"count": 0}
    ordered = sorted(samples_ms)
    p95_index = min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))
    return {
        "count": len(ordered),
        "p50_ms": round(statistics.median(ordered), 2),
        "p95_ms": round(ordered[p95_index], 2),
        "max_ms": round(ordered[-1], 2),
    }


def _time(label: str, fn: Callable[[], None], iterations: int) -> tuple[str, list[float]]:
    samples: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - started) * 1000.0)
    return label, samples


def bench_pure(iterations: int = 200) -> dict[str, dict[str, float | int]]:
    """Micro-benchmark the two modules every subtitle sentence passes through."""
    results: dict[str, list[float]] = {}

    def batcher_flush() -> None:
        batcher = TypeBatcher(window_s=12.0, max_chars=80, clock=lambda: 0.0)
        batcher.add("这句话被上限强行切开", forced=True)
        batcher.add("所以要粘回整句", forced=True)
        batcher.flush()

    def queue_roundtrip() -> None:
        q = EmitQueue(maxsize=64)
        q.submit("一句话", endpoint_ts=time.monotonic(), forced=False)
        item = q.get(timeout=0.05)
        if item is not None:
            q.task_done()

    def queue_under_pressure() -> None:
        q = EmitQueue(maxsize=4)
        for index in range(8):  # 丢最旧：背压路径也要便宜
            q.submit(f"第{index}句", forced=False)
        q.drain()

    for label, fn in (
        ("type_batcher.flush", batcher_flush),
        ("emit_queue.roundtrip", queue_roundtrip),
        ("emit_queue.pressure", queue_under_pressure),
    ):
        name, samples = _time(label, fn, iterations)
        results[name] = samples

    return {name: _percentiles(samples) for name, samples in results.items()}


def parse_log(path: Path) -> dict[str, dict[str, float | int]]:
    """Real-device latencies from an ``app.log`` (seconds → ms)."""
    shown: list[float] = []
    emit: list[float] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if match := _SHOWN_RE.search(line):
                shown.append(float(match.group(1)) * 1000.0)
            elif match := _EMIT_RE.search(line):
                emit.append(float(match.group(1)) * 1000.0)
    return {
        "log.endpoint→shown": _percentiles(shown),
        "log.endpoint→emit": _percentiles(emit),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="xxl-whisper latency baseline")
    parser.add_argument("--log", type=Path, help="真机 app.log，解析 endpoint→shown/emit 分布")
    parser.add_argument("--iterations", type=int, default=200, help="每个纯函数基准跑几次")
    parser.add_argument("--json", type=Path, help="把结果写到这个 JSON（做基线 diff）")
    args = parser.parse_args(argv)

    report: dict[str, dict[str, float | int]] = {}
    report.update(bench_pure(args.iterations))
    if args.log:
        if not args.log.exists():
            print(f"日志不存在：{args.log}", file=sys.stderr)
            return 2
        report.update(parse_log(args.log))

    width = max(len(name) for name in report)
    print(f"{'op'.ljust(width)}  count    p50 ms   p95 ms   max ms")
    for name, stats in report.items():
        if not stats.get("count"):
            print(f"{name.ljust(width)}  {'0':>5}        -        -        -")
            continue
        print(
            f"{name.ljust(width)}  {stats['count']:>5}  {stats['p50_ms']:>8}"
            f"  {stats['p95_ms']:>8}  {stats['max_ms']:>8}"
        )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        args.json.write_text(payload, encoding="utf-8")
        print(f"\n基线已写入 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

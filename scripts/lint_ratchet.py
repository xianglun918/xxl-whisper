"""Lint ratchet: fail only when the findings *grow*.

Why this exists
---------------
``ruff check`` (select = ALL) and ``basedpyright`` (typeCheckingMode = all) both
report pre-existing findings on this branch: 117 ruff and 12 basedpyright at the
time of writing. Wiring the raw commands into CI as hard gates therefore does not
protect anything — it turns every release red until someone clears the whole
backlog, which is how a gate gets disabled "just for now" and never comes back.

So the gate compares against a committed baseline and fails only when a change
*adds* findings. The numbers may only go down: after fixing things, lower the
baseline in the same commit (``python scripts/lint_ratchet.py --update``).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

BASELINE_PATH = Path(__file__).resolve().parent.parent / ".lint-baseline.json"
RUFF_TARGETS = ("app", "tests", "run.py", "scripts")
PYRIGHT_TARGETS = ("app", "run.py")
_RUFF_LINE = re.compile(r"^[^:\n]+:\d+:\d+: [A-Z]+\d+", re.MULTILINE)
_PYRIGHT_TOTAL = re.compile(r"(\d+) errors?")


@dataclass(frozen=True, slots=True)
class Counts:
    """Current finding counts for the two tools."""

    ruff: int
    basedpyright: int

    def as_dict(self) -> dict[str, int]:
        """Return the JSON-serialisable form."""
        return {"ruff": self.ruff, "basedpyright": self.basedpyright}


def _binary(name: str) -> str:
    """Locate a tool on PATH (the venv provides it under CI)."""
    found = shutil.which(name)
    if found is None:
        msg = f"{name} not found on PATH; run this through `uv run` or the venv"
        raise SystemExit(msg)
    return found


def _run(argv: list[str]) -> str:
    """Run a command and return its combined output, ignoring the exit code."""
    proc = subprocess.run(argv, capture_output=True, text=True, check=False)  # noqa: S603
    return proc.stdout + proc.stderr


def count_ruff() -> int:
    """Count ruff findings."""
    output = _run([_binary("ruff"), "check", *RUFF_TARGETS, "--output-format=concise"])
    return len(_RUFF_LINE.findall(output))


def count_basedpyright() -> int:
    """Count basedpyright errors."""
    output = _run([_binary("basedpyright"), *PYRIGHT_TARGETS])
    totals = _PYRIGHT_TOTAL.findall(output)
    return int(totals[-1]) if totals else 0


def load_baseline() -> Counts:
    """Read the committed baseline."""
    raw = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    return Counts(ruff=int(raw["ruff"]), basedpyright=int(raw["basedpyright"]))


def main() -> int:
    """Compare current findings against the baseline; 0 = nothing new."""
    if "--update" in sys.argv[1:]:
        current = Counts(ruff=count_ruff(), basedpyright=count_basedpyright())
        BASELINE_PATH.write_text(json.dumps(current.as_dict(), indent=2) + "\n", encoding="utf-8")
        print(f"baseline updated: {current.as_dict()}")
        return 0

    baseline = load_baseline()
    current = Counts(ruff=count_ruff(), basedpyright=count_basedpyright())
    print(f"{'tool':<14}{'baseline':>10}{'current':>10}{'delta':>8}")
    failed = False
    for name, was, now in (
        ("ruff", baseline.ruff, current.ruff),
        ("basedpyright", baseline.basedpyright, current.basedpyright),
    ):
        delta = now - was
        print(f"{name:<14}{was:>10}{now:>10}{delta:>+8}")
        if now > was:
            failed = True
    if failed:
        print(
            "\nFAIL: this change adds lint/type findings. Fix them, or update "
            ".lint-baseline.json in the same commit if the increase is intended."
        )
        return 1
    if current.ruff < baseline.ruff or current.basedpyright < baseline.basedpyright:
        print("\nFindings went down — lower .lint-baseline.json with --update.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

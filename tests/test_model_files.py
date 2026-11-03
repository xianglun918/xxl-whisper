"""The model layout is one table, and the downloader must agree with it.

Before this, the downloader's `_MODEL_FILES` decided the on-disk filenames and
`asr.Recognizer` guessed them again, with a tarball-alias table in between to
paper over the difference (k2-fsa ships `model.int8.onnx`; we store
`model.onnx`). A mismatch surfaced only at load time, inside sherpa-onnx.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from app.model_files import ASR_KINDS, MODEL_LAYOUT, ModelFiles

from app import asr
from app import downloader as dl


def test_every_downloaded_kind_has_a_layout() -> None:
    assert set(MODEL_LAYOUT) == set(dl._MODEL_FILES)


def test_every_layout_entry_matches_a_downloaded_destination() -> None:
    """Each logical artifact must be something the downloader actually writes."""
    for kind, layout in MODEL_LAYOUT.items():
        dests = {spec[1] for spec in dl._MODEL_FILES[kind]}
        for logical, relative in layout.items():
            written = relative in dests or any(
                dest.startswith(relative + "/") for dest in dests
            )
            assert written, (kind, logical, relative, sorted(dests))


def test_resolve_points_inside_the_model_directory(tmp_path: Path) -> None:
    files = ModelFiles.resolve("whisper_turbo", tmp_path / "whisper_turbo")

    assert files.kind == "whisper_turbo"
    assert files.path("encoder") == tmp_path / "whisper_turbo" / "turbo-encoder.int8.onnx"
    assert files.path("tokens").name == "turbo-tokens.txt"
    assert set(files.paths) == set(MODEL_LAYOUT["whisper_turbo"])


def test_an_unknown_artifact_is_a_layout_bug(tmp_path: Path) -> None:
    files = ModelFiles.resolve("sensevoice", tmp_path / "sensevoice")

    with pytest.raises(KeyError):
        files.path("decoder")  # sensevoice has no decoder


def test_a_translate_route_is_not_an_asr_kind(tmp_path: Path) -> None:
    """Translation has its own file list; the ASR layout must refuse it loudly."""
    assert set(dl._TRANSLATE_MODELS) & ASR_KINDS == set()

    with pytest.raises(KeyError):
        ModelFiles.resolve("ja-zh", tmp_path / "ja-zh")


def _artifacts_the_recogniser_asks_for() -> dict[str, set[str]]:
    """Read ``Recognizer``'s branches and collect ``files.path("<name>")`` calls.

    Parsed from the source on purpose: a hand-copied list of names would pass
    even after the recogniser started asking for something else (that mutation
    survived the first version of this file).
    """
    tree = ast.parse(Path(asr.__file__).read_text(encoding="utf-8"))
    requested: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Match):
            continue
        for case in node.cases:
            pattern = case.pattern
            if not isinstance(pattern, ast.MatchValue):
                continue
            if not isinstance(pattern.value, ast.Constant):
                continue
            names = {
                str(call.args[0].value)
                for call in ast.walk(case)
                if isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "path"
                and call.args
                and isinstance(call.args[0], ast.Constant)
            }
            if names:
                requested[str(pattern.value.value)] = names
    return requested


def test_the_layout_covers_what_the_recogniser_actually_asks_for() -> None:
    requested = _artifacts_the_recogniser_asks_for()

    assert set(requested) == set(MODEL_LAYOUT)
    for kind, names in requested.items():
        assert names <= set(MODEL_LAYOUT[kind]), kind

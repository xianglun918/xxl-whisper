"""The model layout is one table, and the downloader must agree with it.

Before this, the downloader's `_MODEL_FILES` decided the on-disk filenames and
`asr.Recognizer` guessed them again, with a tarball-alias table in between to
paper over the difference (k2-fsa ships `model.int8.onnx`; we store
`model.onnx`). A mismatch surfaced only at load time, inside sherpa-onnx.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from app.model_files import MODEL_LAYOUT, ModelFiles

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


#: What each recogniser branch asks for by name. A layout that forgets one of
#: these fails at load time inside sherpa-onnx, so it is pinned here; adding a
#: model means adding its row to both this map and the layout.
REQUIRED_ARTIFACTS: dict[str, set[str]] = {
    "sensevoice": {"model", "tokens"},
    "whisper_turbo": {"encoder", "decoder", "tokens"},
    "funasr_nano": {"encoder_adaptor", "embedding", "llm", "tokenizer"},
}


def test_the_layout_declares_every_artifact_the_recogniser_asks_for() -> None:
    for kind, required in REQUIRED_ARTIFACTS.items():
        assert required <= set(MODEL_LAYOUT[kind]), kind

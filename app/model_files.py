"""Where one model kind's artifacts live, and which file the recogniser opens.

The downloader decided the on-disk filenames (the dest column of
``_MODEL_FILES``) and ``asr.Recognizer`` guessed them a second time from the
directory name, with a tarball-alias table in between to paper over the
difference — k2-fsa ships ``model.int8.onnx``, we store ``model.onnx``. Adding a
model therefore meant editing both sides, and a missed edit only showed up at
load time, inside sherpa-onnx, as "file not found".

Here the layout is data, once: logical name → relative path. The downloader
writes files at those paths and hands back :class:`ModelFiles` with them
resolved; the recogniser asks for ``"model"`` / ``"tokens"`` / ``"encoder"``
instead of knowing any filename.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

__all__ = ["MODEL_LAYOUT", "ModelFiles"]

#: Logical artifact name → path relative to the kind's model directory. A value
#: that names a directory (the Fun-ASR tokenizer) is a directory on disk; every
#: other entry must appear as a destination in ``downloader._MODEL_FILES``, which
#: ``tests/test_model_files.py`` pins.
MODEL_LAYOUT: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        "sensevoice": MappingProxyType(
            {
                "model": "model.onnx",
                "tokens": "tokens.txt",
            }
        ),
        "whisper_turbo": MappingProxyType(
            {
                "encoder": "turbo-encoder.int8.onnx",
                "decoder": "turbo-decoder.int8.onnx",
                "tokens": "turbo-tokens.txt",
            }
        ),
        "funasr_nano": MappingProxyType(
            {
                "encoder_adaptor": "encoder_adaptor.int8.onnx",
                "embedding": "embedding.int8.onnx",
                "llm": "llm.int8.onnx",
                # A directory: the tokenizer is a small HF-style bundle.
                "tokenizer": "Qwen3-0.6B",
            }
        ),
    }
)


@dataclass(frozen=True, slots=True)
class ModelFiles:
    """Resolved local artifacts for one model kind."""

    kind: str
    directory: Path
    #: Logical name → absolute path, as declared by :data:`MODEL_LAYOUT`.
    paths: Mapping[str, Path]

    def path(self, name: str) -> Path:
        """Return the resolved path for one logical artifact.

        A ``KeyError`` here is a layout bug, not a user problem: the recogniser
        only ever asks for names its own kind declares.
        """
        return self.paths[name]

    @classmethod
    def resolve(cls, kind: str, directory: Path) -> ModelFiles:
        """Build the resolved view of *kind* living under *directory*."""
        layout = MODEL_LAYOUT[kind]
        return cls(
            kind=kind,
            directory=directory,
            paths=MappingProxyType({name: directory / rel for name, rel in layout.items()}),
        )

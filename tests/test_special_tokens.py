"""Per-model Marian special ids, and the download that must supply them.

Real failure this guards: `opus-mt-ja-en` has a 60 716-entry vocabulary and a
decoder start id of 60 715, while `opus-mt-en-zh` uses 65 001 / 65 000. The code
hardcoded 65 000, so **every** Japanese translation died with
``indices element out of data bounds, idx=65000 must be within [-60716, 60715]``
(242 failures out of 245 attempts in the field log).
"""

from __future__ import annotations

import json
from pathlib import Path

from app.translator import _DEFAULT_START_TOKEN_ID, resolve_special_ids


def _write_config(directory: Path, **values: object) -> None:
    (directory / "config.json").write_text(json.dumps(values), encoding="utf-8")


def test_japanese_model_uses_its_own_start_token(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        vocab_size=60_716,
        decoder_start_token_id=60_715,
        eos_token_id=0,
        pad_token_id=60_715,
    )
    start, eos, unk = resolve_special_ids(tmp_path)
    assert start == 60_715
    assert eos == 0
    assert unk == 1


def test_chinese_model_keeps_the_65000_start(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        vocab_size=65_001,
        decoder_start_token_id=65_000,
        eos_token_id=0,
        pad_token_id=65_000,
    )
    assert resolve_special_ids(tmp_path)[0] == 65_000


def test_missing_config_falls_back_to_defaults(tmp_path: Path) -> None:
    assert resolve_special_ids(tmp_path)[0] == _DEFAULT_START_TOKEN_ID


def test_out_of_range_start_is_clamped_not_fatal(tmp_path: Path) -> None:
    """A mismatched export must degrade to a wrong translation, not raise."""
    _write_config(tmp_path, vocab_size=30_000, decoder_start_token_id=65_000)
    start, _eos, _unk = resolve_special_ids(tmp_path)
    assert start == 29_999


def test_unreadable_config_falls_back(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{not json", encoding="utf-8")
    assert resolve_special_ids(tmp_path)[0] == _DEFAULT_START_TOKEN_ID


def test_every_translate_model_downloads_its_config() -> None:
    """Without config.json the ids silently fall back to 65000 and ja→en breaks."""
    from app.downloader import _TRANSLATE_MODELS

    for pair, (_name, files) in _TRANSLATE_MODELS.items():
        names = {relative for relative, _size in files}
        assert "config.json" in names, f"{pair} does not download config.json"

"""Upgrading a config file written by an older build.

The user's real upgrade path is: install the new build over an old one, whose
``config.toml`` still contains retired keys (``vad_*``) and no version stamp.
Every start re-inferred those values, but the file itself kept lying about them
and there was no copy of what the user had before the rewrite. These tests pin
the migration contract:

* the old file is backed up **before** anything is written;
* legacy values that the user actually tuned survive the rename;
* legacy values left at their old default do not pin the new behaviour;
* an already-current, absent, or *invalid* file is left alone.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from app.config import (
    CONFIG_VERSION,
    ConfigError,
    load_config,
    migrate_config_file,
    save_config,
    take_pending_migration,
)

#: What a 0.7-era build wrote: retired ``vad_*`` keys, no ``config_version``.
#: ``hotkey`` 用原始 VK 整数而不是预设名：预设表是分平台的（macOS 默认右 Command，
#: 根本没有 CapsLock 这个键），写预设名会让这组用例在 macOS 上以 ConfigError 失败。
LEGACY_V1 = """\
hotkey = 113
hold_threshold_ms = 400
mic = ""
num_threads = 2
language = "zh"
restore_clipboard = true
paste_delay_ms = 200
check_updates = true
model = "sensevoice"
proxy = ""
disfluency = "verbatim"
continuous = false
vad_threshold = 0.5
vad_min_speech_ms = 250
vad_min_silence_ms = 500
vad_max_speech_ms = 20000
translate = true
translate_target = "zh"
"""


def _legacy(tmp_path: Path, body: str = LEGACY_V1) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(body, encoding="utf-8")
    return path


def test_legacy_file_is_backed_up_then_rewritten(tmp_path: Path) -> None:
    path = _legacy(tmp_path)

    report = migrate_config_file(path)

    assert report is not None
    assert (report.from_version, report.to_version) == (1, CONFIG_VERSION)
    # 备份必须是原文，且必须在新文件写出来之前就存在
    assert report.backup_path is not None
    assert report.backup_path.read_text(encoding="utf-8") == LEGACY_V1
    rewritten = path.read_text(encoding="utf-8")
    assert rewritten.startswith(f"config_version = {CONFIG_VERSION}")
    # 退役键不再出现在文件里，但值已经被搬到新键上
    assert "vad_max_speech_ms" not in rewritten
    assert "vad_min_silence_ms" not in rewritten
    assert {old for old, _new in report.renamed} == {"vad_min_silence_ms", "vad_max_speech_ms"}


def test_legacy_defaults_do_not_pin_the_new_behaviour(tmp_path: Path) -> None:
    """老文件里的 20000 ms 是当时的默认值，不是用户的选择。"""
    path = _legacy(tmp_path)

    migrate_config_file(path)

    config = load_config(path)
    assert config.live_max_speech_ms == 4000  # 新默认，不是 20000
    assert config.live_min_silence_ms == 350  # 500 是旧默认，不是用户选择


def test_a_tuned_legacy_value_survives_the_rename(tmp_path: Path) -> None:
    tuned = LEGACY_V1.replace("vad_max_speech_ms = 20000", "vad_max_speech_ms = 9000")
    path = _legacy(tmp_path, tuned)

    migrate_config_file(path)

    assert load_config(path).live_max_speech_ms == 9000


def test_current_file_is_left_untouched(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    original = f"config_version = {CONFIG_VERSION}\nhotkey = 113\n"
    path.write_text(original, encoding="utf-8")

    assert migrate_config_file(path) is None
    assert path.read_text(encoding="utf-8") == original
    assert not list(tmp_path.glob("*.bak"))


def test_missing_file_is_not_created(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"

    assert migrate_config_file(path) is None
    assert not path.exists()


def test_invalid_file_is_left_alone(tmp_path: Path) -> None:
    """坏文件不能被我改写——让正常的 ConfigError 路径去告诉用户哪里错了。"""
    broken = 'hotkey = 113\nnum_threads = "many"\n'
    path = _legacy(tmp_path, broken)

    assert migrate_config_file(path) is None
    assert path.read_text(encoding="utf-8") == broken
    assert not list(tmp_path.glob("*.bak"))
    with pytest.raises(ConfigError):
        load_config(path)


def test_migration_is_idempotent(tmp_path: Path) -> None:
    path = _legacy(tmp_path)

    assert migrate_config_file(path) is not None
    second = migrate_config_file(path)

    assert second is None
    assert len(list(tmp_path.glob("*.bak"))) == 1


def test_pending_migration_is_reported_once(tmp_path: Path) -> None:
    """启动提示只能弹一次：领完就没了，否则每次模型就绪都会再提示。"""
    take_pending_migration()  # 先把别的测试留下的清掉
    path = _legacy(tmp_path)

    migrate_config_file(path)

    assert take_pending_migration() is not None
    assert take_pending_migration() is None


def test_report_describes_what_happened(tmp_path: Path) -> None:
    path = _legacy(tmp_path)

    report = migrate_config_file(path)

    assert report is not None
    text = report.describe()
    assert "v1" in text
    assert f"v{CONFIG_VERSION}" in text
    assert "vad_max_speech_ms→live_max_speech_ms" in text
    assert report.backup_path is not None
    assert report.backup_path.name in text


def test_save_config_leaves_no_partial_file(tmp_path: Path) -> None:
    """写配置必须"先写临时文件再改名"：迁移在启动时改用户的文件，写一半就退出会留坏 TOML。"""
    path = tmp_path / "config.toml"

    save_config(path, load_config(_legacy(tmp_path, LEGACY_V1)))

    assert path.exists()
    assert not list(tmp_path.glob("*.tmp")), "临时文件必须在改名后消失"
    reparsed = load_config(path)
    assert reparsed.live_max_speech_ms == 4000

"""Regression: 64-bit clipboard pointers must not be truncated by ctypes.

Access-violation crash (2026-08-31 user report): GetClipboardData/GlobalLock
returned 64-bit pointers; without restype declarations ctypes truncated them
to 32-bit ints, and wstring_at dereferenced the sign-extended garbage.

Skipped when the clipboard is held by another process (locked environments).
"""

import ctypes

import pytest

from app import winio


def _clipboard_available() -> bool:
    user32 = ctypes.WinDLL("user32")
    if user32.OpenClipboard(None):
        user32.CloseClipboard()
        return True
    return False


@pytest.mark.integration
@pytest.mark.skipif(not _clipboard_available(), reason="clipboard locked by another process")
def test_clipboard_roundtrip_survives_64bit_pointers() -> None:
    sentinel = "剪贴板回归测试SENTINEL1234567890" * 8  # force GlobalAlloc heap path
    winio._set_clipboard_text(sentinel)
    assert winio._clipboard_text() == sentinel


@pytest.mark.integration
@pytest.mark.skipif(not _clipboard_available(), reason="clipboard locked by another process")
def test_type_text_does_not_raise() -> None:
    winio.type_text("test")  # deliverability needs a real window; API errors still raise

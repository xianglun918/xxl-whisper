"""Regression: 64-bit clipboard pointers must not be truncated by ctypes.

Access-violation crash (2026-08-31 user report): GetClipboardData/GlobalLock
returned 64-bit pointers; without restype declarations ctypes truncated them
to 32-bit ints, and wstring_at dereferenced the sign-extended garbage.

Skipped when the clipboard is held by another process (locked environments).
"""

import sys

import pytest

if sys.platform != "win32":
    pytest.skip("win32-only: raw Win32 clipboard regression", allow_module_level=True)

import ctypes

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


def _register_format(name: str) -> int:
    return winio.user32.RegisterClipboardFormatW(name)


def _set_clipboard_format(fmt: int, payload: bytes, *, empty: bool = True) -> None:
    """Put ``payload`` on the clipboard under ``fmt`` (reusing declared ctypes)."""
    user32, kernel32 = winio.user32, winio.kernel32
    assert user32.OpenClipboard(None)
    try:
        if empty:
            assert user32.EmptyClipboard()
        handle = kernel32.GlobalAlloc(winio._GMEM_MOVEABLE, len(payload))
        assert handle
        ptr = kernel32.GlobalLock(handle)
        assert ptr
        ctypes.memmove(ptr, payload, len(payload))
        kernel32.GlobalUnlock(handle)
        assert user32.SetClipboardData(fmt, handle)
    finally:
        user32.CloseClipboard()


@pytest.mark.integration
@pytest.mark.skipif(not _clipboard_available(), reason="clipboard locked by another process")
def test_probe_plain_text_is_not_non_text() -> None:
    winio._set_clipboard_text("剪贴板文本 probe")
    assert winio.clipboard_has_non_text() is False


@pytest.mark.integration
@pytest.mark.skipif(not _clipboard_available(), reason="clipboard locked by another process")
def test_probe_png_only_is_non_text() -> None:
    _set_clipboard_format(_register_format("PNG"), b"\x89PNG\r\n\x1a\n")
    assert winio.clipboard_has_non_text() is True


@pytest.mark.integration
@pytest.mark.skipif(not _clipboard_available(), reason="clipboard locked by another process")
def test_probe_dib_only_is_non_text() -> None:
    _set_clipboard_format(winio._CF_DIB, b"\x00" * 40)
    assert winio.clipboard_has_non_text() is True


@pytest.mark.integration
@pytest.mark.skipif(not _clipboard_available(), reason="clipboard locked by another process")
def test_probe_html_only_is_non_text() -> None:
    _set_clipboard_format(_register_format("HTML Format"), b"<html><body>x</body></html>")
    assert winio.clipboard_has_non_text() is True


@pytest.mark.integration
@pytest.mark.skipif(not _clipboard_available(), reason="clipboard locked by another process")
def test_probe_html_with_plain_text_is_text() -> None:
    winio._set_clipboard_text("hello")
    _set_clipboard_format(_register_format("HTML Format"), b"<b>hello</b>", empty=False)
    assert winio.clipboard_has_non_text() is False

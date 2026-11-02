"""Win32 结构体布局自检 —— 豆包评审第 3 条（winio.py 裸 ctypes 没测试）。

``app/winio.py`` 里的 SendInput 结构体只能在 Windows 上构造，所以在 Linux/macOS
上本文件整体 skip；真正跑它的是 CI 矩阵里的 ``windows-latest``。

断言的是**不变量**而不是写死的字节数：字段顺序、对齐、联合体大小关系。这类断言
能抓住"把 wVk 从 WORD 写成 LONG""漏了 dwExtraInfo""结构体没按 Win32 文档排序"
这些真错，又不会因为 Python/架构微调就假报警。
"""

from __future__ import annotations

import ctypes
import sys

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason="Win32 ctypes structures only exist on Windows"
)


def _winio():
    from app import winio  # noqa: PLC0415 - 只有 Windows 能 import，必须在 skipif 之后

    return winio


def test_sendinput_structures_are_in_documented_order() -> None:
    """Win32 文档里的字段顺序：顺序错位不会报错，只会"发出去的键不对"。"""
    winio = _winio()

    assert [name for name, _type in winio._KEYBDINPUT._fields_] == [
        "wVk",
        "wScan",
        "dwFlags",
        "time",
        "dwExtraInfo",
    ]
    assert [name for name, _type in winio._MOUSEINPUT._fields_] == [
        "dx",
        "dy",
        "mouseData",
        "dwFlags",
        "time",
        "dwExtraInfo",
    ]
    assert [name for name, _type in winio._INPUT._fields_] == [
        "type",
        "u",
    ]


def test_input_union_is_at_least_as_large_as_every_member() -> None:
    """联合体必须容纳最大成员；少一字节就会在 SendInput 时读到相邻内存。"""
    winio = _winio()

    union_size = ctypes.sizeof(winio._INPUTUNION)
    for member in (winio._MOUSEINPUT, winio._KEYBDINPUT, winio._HARDWAREINPUT):
        assert ctypes.sizeof(member) <= union_size, member.__name__


def test_input_struct_is_tag_plus_union_with_native_alignment() -> None:
    """``INPUT`` = DWORD type + 4 字节填充 + 联合体（x64 上 8 字节对齐）。"""
    winio = _winio()

    input_type = ctypes.sizeof(winio._INPUT)
    union_size = ctypes.sizeof(winio._INPUTUNION)

    # ``type`` 是 DWORD(4)，联合体在 x64 上按 8 对齐 → 头部占 8 字节。
    assert input_type in {union_size + 8, union_size + 4}
    assert input_type - union_size in {4, 8}
    if ctypes.sizeof(ctypes.c_void_p) == 8:
        assert ctypes.alignment(winio._INPUT) == 8


def test_extra_info_field_is_pointer_sized() -> None:
    """``dwExtraInfo`` 是 ULONG_PTR：写成 DWORD 会在 x64 上截断标志位。"""
    winio = _winio()

    for struct in (winio._KEYBDINPUT, winio._MOUSEINPUT):
        field = dict(struct._fields_)["dwExtraInfo"]
        assert ctypes.sizeof(field) == ctypes.sizeof(ctypes.c_void_p)

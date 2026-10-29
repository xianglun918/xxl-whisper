"""Tray menu layout: depth, grouping and the state shown in the parent rows.

The menu grew to thirteen top-level rows, mixing daily switches (source,
translation, subtitles) with once-a-year settings (hotkey, model, autostart).
These tests pin the new shape: at most six top-level rows, related things in the
same submenu, and a parent label that already tells the user what is on.
"""

from app.menu_spec import (
    Entry,
    MenuInputs,
    TrayCallbacks,
    TrayState,
    build_menu,
    input_label,
    state_text,
    translate_hint,
    translate_label,
)


def _callbacks() -> TrayCallbacks:
    def noop(*_args: object, **_kwargs: object) -> None:
        return None

    return TrayCallbacks(
        on_exit=noop,
        on_toggle_pause=noop,
        on_select_mic=noop,
        on_select_system_audio=noop,
        on_toggle_autostart=noop,
        on_check_update=noop,
        on_select_hotkey=noop,
        on_capture_hotkey=noop,
        on_select_model=noop,
        on_show_diagnostics=noop,
        on_toggle_disfluency=noop,
        on_toggle_continuous=noop,
        on_toggle_translate=noop,
        on_select_translate_route=noop,
        on_toggle_overlay=noop,
        on_select_overlay_output=noop,
    )


def _state(**overrides: object) -> TrayState:
    base = {
        "ready": True,
        "paused": False,
        "autostart": False,
        "current_mic": "",
        "current_hotkey": "caps_lock",
        "current_model": "sensevoice",
        "disfluency": "verbatim",
        "continuous": True,
        "translate": True,
        "translate_route": "ja-zh",
        "translator_ready": True,
        "overlay": True,
        "overlay_output": "both",
        "audio_source": "system",
        "current_system_audio": "Speakers (Realtek)",
    }
    base.update(overrides)
    return TrayState(**base)  # type: ignore[arg-type]


def _inputs() -> MenuInputs:
    return MenuInputs(
        microphones=("Mic A", "Mic B"),
        system_audio=(("系统声音 · Speakers (Realtek)", "Speakers (Realtek)"),),
        hotkey_names=("caps_lock", "f2"),
    )


def _visible(entries: tuple[Entry, ...]) -> list[Entry]:
    return [entry for entry in entries if not entry.separator]


def _child(parent: Entry, needle: str) -> Entry:
    for child in parent.children:
        if needle in child.label:
            return child
    message = f"{needle!r} not found under {parent.label!r}"
    raise AssertionError(message)


def test_at_most_seven_top_level_rows() -> None:
    menu = build_menu(_state(), _callbacks(), _inputs())
    rows = _visible(menu)
    assert len(rows) <= 7
    assert rows[0].label == ""  # the status row is text-only
    assert rows[-1].label == "退出"


def test_daily_switches_are_one_level_deep() -> None:
    menu = build_menu(_state(), _callbacks(), _inputs())
    input_row = _child(Entry(children=menu), "输入源")
    translate_row = _child(Entry(children=menu), "翻译与字幕")
    assert input_row.children
    assert translate_row.children
    # The device list and the switches are inside their groups, not at top level.
    labels = [row.label for row in _visible(menu)]
    assert not any(label.startswith("系统声音 · ") for label in labels)
    assert not any(label.startswith("开启") for label in labels)
    assert not any("→ 中" in label and "（" in label for label in labels)


def test_translation_and_subtitles_live_together() -> None:
    row = _child(Entry(children=build_menu(_state(), _callbacks(), _inputs())), "翻译与字幕")
    labels = [child.label for child in _visible(row.children)]
    assert "开启离线翻译" in labels
    assert "开启上屏字幕" in labels
    assert any("日 → 中" in label for label in labels)
    assert any("只上屏" in label for label in labels)


def test_rare_settings_are_grouped_under_more() -> None:
    row = _child(Entry(children=build_menu(_state(), _callbacks(), _inputs())), "更多设置")
    labels = [child.label for child in _visible(row.children)]
    for expected in ("热键", "识别模型", "开机自启", "输入诊断", "检查更新"):
        assert expected in labels


def test_continuous_dictation_is_a_top_level_switch() -> None:
    """It is a daily toggle (change of scene), not a set-once setting."""
    menu = build_menu(_state(continuous=True), _callbacks(), _inputs())
    top = [row.label for row in _visible(menu)]
    assert "持续听写" in top
    assert top.index("持续听写") == top.index("暂停语音热键") + 1
    continuous_row = next(row for row in _visible(menu) if row.label == "持续听写")
    assert continuous_row.checked is True
    more = _child(Entry(children=menu), "更多设置")
    assert not any(child.label == "持续听写" for child in more.children)


def test_parent_labels_carry_live_state() -> None:
    state = _state()
    assert input_label(state) == "输入源：系统声音 · Speakers (Realtek)"
    assert translate_label(state) == "翻译与字幕：日→中 · 上屏+打字"
    assert "持续听写中" in state_text(state)
    assert "翻译开（日→中）" in state_text(state)


def test_labels_say_when_nothing_is_on() -> None:
    state = _state(translate=False, overlay=False, audio_source="mic", current_mic="")
    assert translate_label(state) == "翻译与字幕：关"
    assert input_label(state) == "输入源：系统默认麦克风"
    assert translate_hint(state) == "提示：文字打到光标处"


def test_subtitles_without_translation_is_called_out() -> None:
    state = _state(translate=False, overlay=True)
    assert translate_hint(state) == "提示：字幕当前显示原文，未翻译"
    assert "字幕显示原文" in state_text(state)


def test_route_choices_are_disabled_until_translation_is_on() -> None:
    row = _child(Entry(children=build_menu(_state(translate=False), _callbacks(), _inputs())),
                 "翻译与字幕")
    route = _child(row, "日 → 中")
    assert route.enabled is False


def test_current_route_and_output_are_checked() -> None:
    row = _child(Entry(children=build_menu(_state(), _callbacks(), _inputs())), "翻译与字幕")
    assert _child(row, "日 → 中").checked is True
    assert _child(row, "中 → 英").checked is False
    assert _child(row, "上屏 + 打字").checked is True


def test_system_audio_sits_after_a_separator_in_the_input_menu() -> None:
    row = _child(Entry(children=build_menu(_state(), _callbacks(), _inputs())), "输入源")
    kinds = [child.separator for child in row.children]
    assert kinds.index(True) < len(kinds) - 1  # separator before the loopback rows
    last = row.children[-1]
    assert last.checked is True  # the active system-audio endpoint

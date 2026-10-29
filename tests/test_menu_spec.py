"""Tray menu layout, and the ticks/labels that must stay live.

Two things are pinned here:

* the shape — a short top-level list, related things grouped, parent labels that
  already tell the user what is on;
* **liveness** — every tick and dynamic label is a callable over the state
  provider. An earlier version captured ``state.paused`` at build time, so the
  menu kept showing the old state after a switch was flipped.
"""

from dataclasses import replace

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


BASE_STATE = TrayState(
    ready=True,
    paused=False,
    autostart=False,
    current_mic="",
    current_hotkey="caps_lock",
    current_model="sensevoice",
    disfluency="verbatim",
    continuous=True,
    translate=True,
    translate_route="ja-zh",
    translator_ready=True,
    overlay=True,
    overlay_output="both",
    audio_source="system",
    current_system_audio="Speakers (Realtek)",
)


def _state(**overrides: object) -> TrayState:
    return replace(BASE_STATE, **overrides)  # type: ignore[arg-type]


class Live:
    """Mutable state holder standing in for the app's live provider."""

    def __init__(self, state: TrayState) -> None:
        self.state = state

    def provider(self) -> TrayState:
        return self.state


def _inputs() -> MenuInputs:
    return MenuInputs(
        microphones=("Mic A", "Mic B"),
        system_audio=(("系统声音 · Speakers (Realtek)", "Speakers (Realtek)"),),
        hotkey_names=("caps_lock", "f2"),
    )


def _menu(state: TrayState) -> tuple[tuple[Entry, ...], Live]:
    live = Live(state)
    return build_menu(live.provider, _callbacks(), _inputs()), live


def _label(entry: Entry) -> str:
    """Resolve a row's text: dynamic rows carry a callable, static rows a string."""
    return entry.text() if entry.text is not None else entry.label


def _visible(entries: tuple[Entry, ...]) -> list[Entry]:
    return [entry for entry in entries if not entry.separator]


def _child(parent: Entry, needle: str) -> Entry:
    for child in parent.children:
        if needle in _label(child):
            return child
    message = f"{needle!r} not found under {parent.label!r}"
    raise AssertionError(message)


def _row(menu: tuple[Entry, ...], needle: str) -> Entry:
    for entry in _visible(menu):
        if needle in _label(entry):
            return entry
    message = f"{needle!r} not found at top level"
    raise AssertionError(message)


# -- shape ---------------------------------------------------------------------


def test_at_most_seven_top_level_rows() -> None:
    menu, _live = _menu(_state())
    rows = _visible(menu)
    assert len(rows) <= 7
    assert rows[0].label == ""  # the status row is text-only
    assert rows[-1].label == "退出"


def test_continuous_dictation_is_a_top_level_switch() -> None:
    """It is a daily toggle (change of scene), not a set-once setting."""
    menu, _live = _menu(_state())
    top = [_label(row) for row in _visible(menu)]
    assert "持续听写" in top
    assert top.index("持续听写") == top.index("暂停语音热键") + 1
    more = _row(menu, "更多设置")
    assert not any(child.label == "持续听写" for child in more.children)


def test_daily_switches_are_one_level_deep() -> None:
    menu, _live = _menu(_state())
    input_row = _row(menu, "输入源")
    assert input_row.children
    labels = [_label(row) for row in _visible(menu)]
    assert not any(label.startswith("系统声音 · ") for label in labels)
    assert not any(label.startswith("开启") for label in labels)


def test_translation_and_subtitles_live_together() -> None:
    menu, _live = _menu(_state())
    row = _row(menu, "翻译与字幕")
    labels = [_label(child) for child in _visible(row.children)]
    assert "开启离线翻译" in labels
    assert "开启上屏字幕" in labels
    assert any("日 → 中" in label for label in labels)
    assert any("只上屏" in label for label in labels)


def test_rare_settings_are_grouped_under_more() -> None:
    menu, _live = _menu(_state())
    more = _row(menu, "更多设置")
    labels = [_label(child) for child in _visible(more.children)]
    for expected in ("热键", "识别模型", "开机自启", "输入诊断", "检查更新"):
        assert expected in labels


# -- liveness ------------------------------------------------------------------


def test_ticks_follow_the_state_after_the_menu_was_built() -> None:
    """The bug this pins: a frozen tick made switches look like they did nothing."""
    menu, live = _menu(_state(continuous=False, translate=False, overlay=False))
    continuous = _row(menu, "持续听写")
    translate_row = _child(_row(menu, "翻译与字幕"), "开启离线翻译")
    assert continuous.checked is not None
    assert continuous.checked() is False
    assert translate_row.checked is not None
    assert translate_row.checked() is False

    live.state = _state(continuous=True, translate=True, overlay=False)  # flipped in the app
    assert continuous.checked() is True
    assert translate_row.checked() is True


def test_route_and_output_ticks_follow_the_state() -> None:
    menu, live = _menu(_state(translate_route="ja-zh", overlay_output="both"))
    row = _row(menu, "翻译与字幕")
    ja = _child(row, "日 → 中")
    both = _child(row, "上屏 + 打字")
    assert ja.checked is not None
    assert both.checked is not None
    assert ja.checked() is True
    assert both.checked() is True

    live.state = _state(translate_route="en-zh", overlay_output="screen")
    assert ja.checked() is False
    assert _child(row, "英 → 中").checked() is True  # type: ignore[union-attr]
    assert both.checked() is False
    assert _child(row, "只上屏").checked() is True  # type: ignore[union-attr]


def test_parent_labels_are_rendered_lazily() -> None:
    menu, live = _menu(_state())
    translate_row = _row(menu, "翻译与字幕")
    assert translate_row.text is not None
    assert translate_row.text() == "翻译与字幕：日→中 · 上屏+打字"

    live.state = _state(translate=False, overlay=False)
    assert translate_row.text() == "翻译与字幕：关"


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


def test_current_route_and_output_are_checked() -> None:
    menu, _live = _menu(_state())
    row = _row(menu, "翻译与字幕")
    assert _child(row, "日 → 中").checked is not None
    assert _child(row, "日 → 中").checked() is True  # type: ignore[misc]
    assert _child(row, "中 → 英").checked() is False  # type: ignore[misc]
    assert _child(row, "上屏 + 打字").checked() is True  # type: ignore[misc]


def test_system_audio_sits_after_a_separator_in_the_input_menu() -> None:
    menu, _live = _menu(_state())
    row = _row(menu, "输入源")
    kinds = [child.separator for child in row.children]
    assert kinds.index(True) < len(kinds) - 1
    last = row.children[-1]
    assert last.checked is not None
    assert last.checked() is True

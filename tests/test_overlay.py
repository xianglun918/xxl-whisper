"""Subtitle overlay: stacking, de-duplication, expiry and mode rendering.

Only :class:`OverlayModel` is exercised here — it is the pure half of the
feature, driven by a fake clock so the hold/expiry rules are tested without a
display, a Tk loop or real sleeping. The window itself is covered by
``scripts/overlay_smoke.py`` (run under Xvfb).
"""

from dataclasses import replace

from app.overlay import Line, OverlayModel, OverlayStyle, build_style


class FakeClock:
    """Monotonic clock the test moves by hand."""

    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _model(**style: object) -> tuple[OverlayModel, FakeClock]:
    clock = FakeClock()
    return OverlayModel(OverlayStyle(**style), clock=clock), clock  # type: ignore[arg-type]


def test_push_adds_one_segment() -> None:
    model, _clock = _model()
    assert model.push("你好", "hello") is True
    assert model.lines() == [Line(kind="target", text="hello")]
    assert model.is_empty() is False


def test_blank_push_is_ignored() -> None:
    model, _clock = _model()
    assert model.push("   ", "") is False
    assert model.is_empty() is True
    assert model.lines() == []


def test_repeat_refreshes_hold_without_duplicating() -> None:
    model, clock = _model(hold_seconds=2.0)
    model.push("你好", "hello")
    clock.advance(1.5)
    assert model.push("你好", "hello") is True  # re-render, no new line
    clock.advance(1.5)
    # 3.0 s after the first push, but only 1.5 s after the repeat: still alive.
    assert model.expire() is False
    assert model.lines() == [Line(kind="target", text="hello")]


def test_old_segments_expire() -> None:
    model, clock = _model(hold_seconds=2.0)
    model.push("一", "one")
    clock.advance(2.5)
    assert model.expire() is True
    assert model.is_empty() is True


def test_expiry_only_drops_finished_segments() -> None:
    model, clock = _model(hold_seconds=2.0, max_lines=3)
    model.push("一", "one")
    clock.advance(1.5)
    model.push("二", "two")
    clock.advance(1.0)  # first is 2.5 s old, second 1.0 s
    assert model.expire() is True
    assert model.lines() == [Line(kind="target", text="two")]


def test_stack_is_trimmed_to_max_lines() -> None:
    model, _clock = _model(max_lines=2)
    for index in range(4):
        model.push(f"源{index}", f"t{index}")
    assert model.lines() == [
        Line(kind="target", text="t2"),
        Line(kind="target", text="t3"),
    ]


def test_bilingual_renders_source_above_target() -> None:
    model, _clock = _model(mode="bilingual")
    model.push("你好", "hello")
    assert model.lines() == [
        Line(kind="source", text="你好"),
        Line(kind="target", text="hello"),
    ]


def test_source_mode_shows_only_what_was_said() -> None:
    model, _clock = _model(mode="source")
    model.push("你好", "hello")
    assert model.lines() == [Line(kind="source", text="你好")]


def test_failed_translation_falls_back_to_source() -> None:
    """A blank translation must never leave an empty bar (MT is best effort)."""
    model, _clock = _model(mode="target")
    model.push("你好", "  ")
    assert model.lines() == [Line(kind="target", text="你好")]


def test_clear_drops_everything() -> None:
    model, _clock = _model()
    model.push("一", "one")
    model.clear()
    assert model.is_empty() is True
    assert model.lines() == []


def test_style_comes_from_flat_config() -> None:
    class FakeConfig:
        overlay_mode = "bilingual"
        overlay_font_size = 40
        overlay_opacity = 0.7
        overlay_width_ratio = 0.5
        overlay_bottom_margin = 120
        overlay_hold_seconds = 6.0
        overlay_max_lines = 4
        overlay_click_through = False

    style = build_style(FakeConfig())
    assert style == OverlayStyle(
        mode="bilingual",
        font_size=40,
        opacity=0.7,
        width_ratio=0.5,
        bottom_margin=120,
        hold_seconds=6.0,
        max_lines=4,
        click_through=False,
    )


def test_style_is_immutable() -> None:
    """Style is shared with the Tk thread; nobody may mutate it in place."""
    style = OverlayStyle()
    assert replace(style, mode="bilingual").mode == "bilingual"
    assert style.mode == "target"

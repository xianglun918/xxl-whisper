"""Our tray adapter against a faithful copy of pystray 0.19.5's protocol.

pystray cannot be imported headlessly, but its behaviour is small and now known
exactly (read from the installed 0.19.5 wheel):

* ``MenuItem(text, action, checked=None, ...)`` wraps constants into callables;
* ``text``/``checked`` are **called with the item** — a zero-argument lambda
  raises ``TypeError`` while the native menu is being created, which is why a
  mis-shaped callable makes the whole menu appear dead rather than wrong;
* ``MenuItem.__call__(icon)`` invokes the action as ``action(icon, item)``, and
  ``_assert_action`` wraps actions that take fewer arguments;
* ``Menu`` strips head/tail separators and collapses consecutive ones — and only
  recognises ``Menu.SEPARATOR`` by identity;
* ``Icon._update_menu`` destroys and recreates the native handle, so a menu
  rebuild must never happen while a popup is on screen.

The stub below mirrors those rules so the adapter is tested against them instead
of against our assumptions.
"""

from __future__ import annotations

import functools
import inspect
import sys
import types

import pytest


class FakeIconImage:
    """Stand-in for a PIL image (only constructed, never inspected)."""


def _wrap(value):
    return value if callable(value) else (lambda _: value)


class FakeMenuItem:
    """pystray.MenuItem, reduced to the behaviour we depend on."""

    def __init__(self, text, action, checked=None, radio=False, default=False,
                 visible=True, enabled=True):
        self.__name__ = str(text)
        self._text = _wrap(text or "")
        self._action = self._assert_action(action)
        self._checked = checked if checked is not None else (lambda _: None)
        self._radio = _wrap(radio)
        self._default = _wrap(default)
        self._visible = _wrap(visible)
        self._enabled = _wrap(enabled)

    def __call__(self, icon):
        return self._action(icon, self)

    @property
    def text(self):
        return self._text(self)

    @property
    def checked(self):
        return self._checked(self)

    @property
    def radio(self):
        return self._radio(self)

    @property
    def default(self):
        return self._default(self)

    @property
    def visible(self):
        return self._visible(self)

    @property
    def enabled(self):
        return self._enabled(self)

    @property
    def submenu(self):
        return self._action if isinstance(self._action, FakeMenu) else None

    @staticmethod
    def _assert_action(action):
        if action is None:
            return lambda *_: None
        if not hasattr(action, "__code__"):
            return action
        argcount = action.__code__.co_argcount - (1 if inspect.ismethod(action) else 0)
        if argcount == 0:
            @functools.wraps(action)
            def wrapper0(*args):
                return action()
            return wrapper0
        if argcount == 1:
            @functools.wraps(action)
            def wrapper1(icon, *args):
                return action(icon)
            return wrapper1
        if argcount == 2:
            return action
        raise ValueError(action)


class FakeMenu:
    """pystray.Menu: identity-based separators, head/tail stripping."""

    SEPARATOR = FakeMenuItem("- - - -", None)

    def __init__(self, *items):
        self._items = tuple(items)

    @property
    def items(self):
        return self._items

    @property
    def visible(self):
        return bool(self)

    def __iter__(self):
        return iter(self._visible_items())

    def __bool__(self):
        return len(self._visible_items()) > 0

    def _visible_items(self):
        kept = []
        for item in self._items:
            if not item.visible:
                continue
            if item is FakeMenu.SEPARATOR and kept and kept[-1] is FakeMenu.SEPARATOR:
                continue
            kept.append(item)
        while kept and kept[0] is FakeMenu.SEPARATOR:
            kept.pop(0)
        while kept and kept[-1] is FakeMenu.SEPARATOR:
            kept.pop()
        return tuple(kept)


@pytest.fixture()
def tray_module(monkeypatch):
    """Import app.tray with pystray/PIL/sounddevice stubbed out."""
    pystray = types.ModuleType("pystray")
    pystray.Menu = FakeMenu
    pystray.MenuItem = FakeMenuItem
    pystray.Menu.SEPARATOR = FakeMenu.SEPARATOR

    icons: list[object] = []

    class FakePystrayIcon:
        def __init__(self, name, icon=None, title=None, menu=None):
            self.name = name
            self.title = title
            self._menu = menu
            self.updates = 0
            self.notified: list[str] = []
            self.stopped = False
            icons.append(self)

        @property
        def menu(self):
            return self._menu

        @menu.setter
        def menu(self, value):
            self._menu = value
            self.update_menu()

        def update_menu(self):
            self.updates += 1

        def notify(self, message, title=None):
            self.notified.append(message)

        def run(self):
            return None

        def stop(self):
            self.stopped = True

    pystray.Icon = FakePystrayIcon

    sounddevice = types.ModuleType("sounddevice")
    sounddevice.query_devices = lambda *a, **k: [
        {"name": "Mic A", "max_input_channels": 2},
        {"name": "Mic B", "max_input_channels": 1},
        {"name": "Speakers", "max_input_channels": 0},
    ]
    sounddevice.default = types.SimpleNamespace(device=(0, 0))

    # app.recorder does ``from app import native``, and the real platform layer
    # imports AppKit/ctypes — stub it so the tray adapter can be imported here.
    native = types.ModuleType("app.native")
    native.hotkey = types.SimpleNamespace(
        PRESET_KEYCODES={"caps_lock": 0x14, "f2": 0x71, "mouse_x1": 0x05},
        MOUSE_KEYCODES=frozenset({0x05}),
        DEFAULT_HOTKEY="caps_lock",
    )
    native.data_root = lambda: "/tmp/xxl-whisper-test"
    native.io = types.SimpleNamespace(
        active_monitor_work_area=lambda: (0, 0, 1920, 1080),
        exclusive_fullscreen_owner_active=lambda: False,
    )
    native.overlay = types.SimpleNamespace()
    native.util = types.SimpleNamespace(autostart_enabled=lambda: False)

    for name, module in (
        ("pystray", pystray),
        ("sounddevice", sounddevice),
        ("app.native", native),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    for name in ("app.tray", "app.recorder", "app.loopback", "app.menu_spec", "app.config"):
        monkeypatch.delitem(sys.modules, name, raising=False)

    import app.tray as tray

    tray._TEST_ICONS = icons
    return tray


def _callbacks(tray, calls: list[str]):
    def record(name):
        def action(*_args: object) -> None:
            calls.append(name)

        return action

    return tray.TrayCallbacks(
        on_exit=record("exit"),
        on_toggle_pause=record("pause"),
        on_select_mic=record("mic"),
        on_select_system_audio=record("system-audio"),
        on_toggle_autostart=record("autostart"),
        on_check_update=record("update"),
        on_select_hotkey=record("hotkey"),
        on_capture_hotkey=record("capture"),
        on_select_model=record("model"),
        on_show_diagnostics=record("diagnostics"),
        on_select_disfluency=record("disfluency"),
        on_toggle_continuous=record("continuous"),
        on_toggle_translate=record("translate"),
        on_select_translate_route=record("route"),
        on_select_overlay_output=record("output"),
    )


def _state(tray, **overrides: object):
    base = {
        "ready": True,
        "paused": False,
        "autostart": False,
        "current_mic": "",
        "current_hotkey": "caps_lock",
        "current_model": "sensevoice",
        "disfluency": "verbatim",
        "continuous": False,
        "translate": False,
        "translate_route": "en-zh",
        "translator_ready": False,
        "overlay": False,
        "overlay_output": "both",
        "audio_source": "mic",
        "current_system_audio": "",
    }
    base.update(overrides)
    return tray.TrayState(**base)  # type: ignore[arg-type]


def _build(tray, calls: list[str], **overrides: object):
    state = {"value": _state(tray, **overrides)}
    instance = tray.Tray(_callbacks(tray, calls), lambda: state["value"])
    return instance, state


def _flat(menu) -> list[tuple[str, object]]:
    """Flatten a menu into (leaf label, item) pairs, submenus included.

    Keyed by the leaf text, because that is how a user refers to an entry; the
    same leaf may appear under different parents, which is fine for these tests.
    """
    out: list[tuple[str, object]] = []
    separator = type(menu).SEPARATOR
    for item in menu.items:
        if item is separator:
            continue
        out.append((item.text, item))
        if item.submenu is not None:
            out.extend(_flat(item.submenu))
    return out


def test_every_clickable_item_fires_its_callback(tray_module) -> None:
    """This is the regression: pystray calls actions as action(icon, item)."""
    calls: list[str] = []
    instance, _state_holder = _build(tray_module, calls)
    menu = instance._icon.menu
    icon = instance._icon

    expected = {
        "暂停语音热键": "pause",
        "持续听写": "continuous",
        "开启离线翻译": "translate",
        "开机自启": "autostart",
        "输入诊断": "diagnostics",
        "检查更新": "update",
        "退出": "exit",
    }
    items = dict(_flat(menu))
    for label, _name in expected.items():
        items[label](icon)  # must not raise

    assert len(calls) == len(expected), calls


def test_every_radio_option_is_clickable(tray_module) -> None:
    calls: list[str] = []
    instance, _state_holder = _build(tray_module, calls)
    items = dict(_flat(instance._icon.menu))
    for label in ("日 → 中（经英文中转，两段）", "只上屏（不打字）", "语义顺滑（去掉语气词与重复）",
                  "CapsLock", "Fun-ASR-Nano（中文更准 · 约 1GB · 日语偏弱）"):
        items[label](instance._icon)
    assert calls == ["route", "output", "disfluency", "hotkey", "model"]


def test_ticks_and_labels_are_evaluated_with_the_item(tray_module) -> None:
    """A zero-argument callable here raises inside pystray and kills the menu."""
    calls: list[str] = []
    instance, state = _build(tray_module, calls, continuous=True, translate=True,
                             overlay=True, translate_route="ja-zh")
    menu = instance._icon.menu

    for _path, item in _flat(menu):
        assert isinstance(item.text, str)  # raises TypeError if the callable is wrong
        assert item.checked in (True, False, None)

    state["value"] = _state(tray_module, continuous=False)
    items = dict(_flat(menu))
    assert items["持续听写"].checked is False


def test_separators_are_the_identity_pystray_expects(tray_module) -> None:
    calls: list[str] = []
    instance, _state_holder = _build(tray_module, calls)
    menu = instance._icon.menu
    separator = type(menu).SEPARATOR
    separators = [item for item in menu.items if item is separator]
    assert separators, "menu has no separator rows"
    assert instance._icon.menu.visible is True


def test_refresh_is_deferred_and_never_replaces_the_menu(tray_module, monkeypatch) -> None:
    """Replacing the menu destroys the handle pystray is using (dead clicks)."""
    calls: list[str] = []
    instance, _state_holder = _build(tray_module, calls)
    original_menu = instance._icon.menu
    scheduled: list[object] = []

    def fake_schedule(refresh, on_error, *, delay, timer_factory=None):
        scheduled.append(refresh)
        return None

    monkeypatch.setattr(tray_module, "schedule_refresh", fake_schedule)
    instance.refresh_menu()
    assert len(scheduled) == 1, "refresh should be deferred, not run inline"
    assert instance._icon.menu is original_menu, "menu object must not be replaced"


def test_menu_actions_survive_an_exception(tray_module) -> None:
    """A failing action must not escape into the platform menu loop."""

    def boom() -> None:
        reason = "action boom"
        raise RuntimeError(reason)

    instance = tray_module.Tray(
        tray_module.TrayCallbacks(
            on_exit=boom,
            on_toggle_pause=boom,
            on_select_mic=boom,
            on_select_system_audio=boom,
            on_toggle_autostart=boom,
            on_check_update=boom,
            on_select_hotkey=boom,
            on_capture_hotkey=boom,
            on_select_model=boom,
            on_show_diagnostics=boom,
            on_select_disfluency=boom,
            on_toggle_continuous=boom,
            on_toggle_translate=boom,
            on_select_translate_route=boom,
            on_select_overlay_output=boom,
        ),
        lambda: _state(tray_module),
    )
    items = dict(_flat(instance._icon.menu))
    items["退出"](instance._icon)  # must not raise

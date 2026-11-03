"""The translation routes: which directions exist, and everything about them.

One route used to be five partial facts in five modules — the ``Literal`` and the
validation set in :mod:`app.config`, the radio labels in :mod:`app.options`, the
long labels plus ``route_text`` in :mod:`app.translator`, the model chain in
:mod:`app.downloader`, and the "direct model failed, use the two-hop chain"
branch hard-coded inside :mod:`app.controls` — held together by one test that
asserted four sets were equal.

Two symptoms proved that arrangement wrong:

* **the labels drifted.** The menu said ``说中文出英文``; the status flash said
  ``说中文，出英文``, because the same fact was written twice;
* **adding a route meant five edits**, and the one time one was missed the
  ``KeyError`` fired inside the worker: the menu looked broken and every test
  passed (that is the ja→en incident recorded in ``route_text``'s docstring).

This module is the single home. Consumers project from it: the tray builds its
radio group, the downloader its model plan, the controls its fallback chain, and
the recogniser its pinned language.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "ROUTES",
    "RouteSpec",
    "fallback_of",
    "models_for",
    "pin_language",
    "route_keys",
    "route_short",
    "route_text",
    "spec_of",
]


@dataclass(frozen=True, slots=True)
class RouteSpec:
    """Everything one translation direction implies.

    Keeping ``label``/``short`` here is what makes the tray and the status
    flash read the same string; ``pin_language``/``fallback`` are what used to
    be an ``if route == "ja-zh"`` inside the download path.
    """

    #: Canonical key: ``"<source>-<target>"``.
    key: str
    #: Full label, shown in the tray radio and in every status message.
    label: str
    #: Compact label for a parent menu row (``"翻译与字幕：日→中"``).
    short: str
    #: Spoken language fed to the recogniser for this route.
    source: str
    #: Language the text comes out in.
    target: str
    #: Model keys, in hop order. Two entries means a chained translation.
    models: tuple[str, ...]
    #: Recogniser language this route requires (Japanese audio decoded as
    #: Chinese produces plausible-looking gibberish, so ja routes pin it).
    pin_language: str | None = None
    #: What to use when the preferred model cannot be fetched. Empty means the
    #: route has exactly one implementation.
    fallback: tuple[str, ...] = ()


#: Every supported route, in tray order. Adding one here is the whole edit.
ROUTES: tuple[RouteSpec, ...] = (
    RouteSpec(
        key="zh-en",
        label="中 → 英（说中文出英文）",
        short="中→英",
        source="zh",
        target="en",
        models=("zh-en",),
    ),
    RouteSpec(
        key="en-zh",
        label="英 → 中（说英文出中文）",
        short="英→中",
        source="en",
        target="zh",
        models=("en-zh",),
    ),
    RouteSpec(
        key="ja-zh",
        label="日 → 中（直译单模型）",
        short="日→中",
        source="ja",
        target="zh",
        models=("ja-zh",),
        pin_language="ja",
        # This project ships one converted 145 MB ja→zh model; if it cannot be
        # fetched, the two-hop Marian chain still exists, so the user gets
        # Japanese→Chinese rather than an error.
        fallback=("ja-en", "en-zh"),
    ),
    RouteSpec(
        key="ja-en",
        label="日 → 英（一段直出，最快最省内存）",
        short="日→英",
        source="ja",
        target="en",
        models=("ja-en",),
        pin_language="ja",
    ),
)

_BY_KEY: dict[str, RouteSpec] = {spec.key: spec for spec in ROUTES}


def route_keys() -> tuple[str, ...]:
    """Every supported route key, in tray order."""
    return tuple(spec.key for spec in ROUTES)


def spec_of(route: str) -> RouteSpec | None:
    """Return the spec for *route*, or None when it is not a supported direction."""
    return _BY_KEY.get(route)


def route_text(route: str) -> str:
    """Return the human label for *route*, falling back to the raw key.

    Never index ``ROUTES`` directly from a UI path: an unknown key must not raise
    inside the worker, where the traceback reads like a broken tray.
    """
    spec = _BY_KEY.get(route)
    return spec.label if spec is not None else route


def route_short(route: str) -> str:
    """Compact label for a parent row; the raw key when unknown."""
    spec = _BY_KEY.get(route)
    return spec.short if spec is not None else route


def models_for(route: str) -> tuple[str, ...]:
    """Model keys to download, in hop order (empty for an unknown route)."""
    spec = _BY_KEY.get(route)
    return spec.models if spec is not None else ()


def fallback_of(route: str) -> tuple[str, ...]:
    """Model chain to use when the preferred one is unreachable."""
    spec = _BY_KEY.get(route)
    return spec.fallback if spec is not None else ()


def pin_language(route: str) -> str | None:
    """Recogniser language this route requires, or None to leave it alone."""
    spec = _BY_KEY.get(route)
    return spec.pin_language if spec is not None else None

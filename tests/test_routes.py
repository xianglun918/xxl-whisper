"""The route catalogue is the single source for every route fact.

One route used to be five partial facts in five modules, kept in step by a test
that asserted four sets were equal — and the labels still drifted (the tray said
「说中文出英文」, the status flash said 「说中文，出英文」). These tests pin the
catalogue itself, then pin that each consumer projects from it instead of
carrying its own copy.
"""

from __future__ import annotations

from pathlib import Path

from app.config import language_for_route
from app.downloader import TRANSLATE_ROUTE_MODELS
from app.options import ROUTE_OPTIONS, ROUTE_SHORT
from app.routes import (
    ROUTES,
    fallback_of,
    models_for,
    pin_language,
    route_keys,
    route_short,
    route_text,
    spec_of,
)
from app.translator import TRANSLATE_ROUTES


def test_every_route_is_self_consistent() -> None:
    """Key, direction and label must agree, and every field must be usable."""
    for spec in ROUTES:
        assert spec.key == f"{spec.source}-{spec.target}", spec.key
        assert spec.label.strip(), spec.key
        assert spec.short.strip(), spec.key
        assert spec.label.startswith(spec.short.replace("→", " → ")), spec.key
        assert spec.models, spec.key


def test_keys_are_unique_and_sources_are_supported() -> None:
    keys = [spec.key for spec in ROUTES]
    assert len(keys) == len(set(keys))
    assert {spec.source for spec in ROUTES} <= {"zh", "en", "ja"}
    assert {spec.target for spec in ROUTES} <= {"zh", "en"}


def test_a_fallback_chain_points_at_real_routes() -> None:
    """A fallback naming a route that does not exist would fail at download time."""
    for spec in ROUTES:
        for hop in spec.fallback:
            assert spec_of(hop) is not None, (spec.key, hop)
        assert spec.key not in spec.fallback, spec.key


def test_japanese_routes_pin_the_recogniser_language() -> None:
    """Japanese audio decoded as Chinese is plausible-looking gibberish."""
    assert pin_language("ja-zh") == "ja"
    assert pin_language("ja-en") == "ja"
    assert pin_language("zh-en") is None
    assert language_for_route("auto", "ja-zh") == "ja"
    assert language_for_route("ja", "zh-en") == "auto"  # leaving 日→… restores it


def test_the_tray_and_the_status_flash_share_one_label() -> None:
    """The drift that started this refactor: two spellings of one direction."""
    for spec in ROUTES:
        assert route_text(spec.key) == spec.label
        assert {option.key: option.label for option in ROUTE_OPTIONS}[spec.key] == spec.label
        assert TRANSLATE_ROUTES[spec.key] == spec.label
        assert ROUTE_SHORT[spec.key] == spec.short
        assert route_short(spec.key) == spec.short


def test_consumers_carry_no_private_copy_of_the_route_list() -> None:
    """Each projection must have exactly the catalogue's keys."""
    keys = set(route_keys())
    assert {option.key for option in ROUTE_OPTIONS} == keys
    assert set(TRANSLATE_ROUTES) == keys
    assert set(ROUTE_SHORT) == keys
    assert set(TRANSLATE_ROUTE_MODELS) == keys
    for spec in ROUTES:
        assert TRANSLATE_ROUTE_MODELS[spec.key] == spec.models
        assert models_for(spec.key) == spec.models
        assert fallback_of(spec.key) == spec.fallback


def test_unknown_routes_never_raise() -> None:
    """A UI path must not explode on a key the catalogue does not know."""
    assert route_text("mt-ja-zh-v1") == "mt-ja-zh-v1"
    assert route_short("nope") == "nope"
    assert models_for("nope") == ()
    assert fallback_of("nope") == ()
    assert pin_language("nope") is None


#: User-visible wording, pinned on purpose: rewording a direction is a product
#: decision, so it should require touching a test, not slip through as a diff.
EXPECTED_LABELS: dict[str, str] = {
    "zh-en": "中 → 英（说中文出英文）",
    "en-zh": "英 → 中（说英文出中文）",
    "ja-zh": "日 → 中（直译单模型）",
    "ja-en": "日 → 英（一段直出，最快最省内存）",
}

EXPECTED_SHORT: dict[str, str] = {
    "zh-en": "中→英",
    "en-zh": "英→中",
    "ja-zh": "日→中",
    "ja-en": "日→英",
}


def test_user_visible_labels_are_pinned() -> None:
    """The menu and the status flash both read these; a typo is a release note."""
    assert {spec.key: spec.label for spec in ROUTES} == EXPECTED_LABELS
    assert {spec.key: spec.short for spec in ROUTES} == EXPECTED_SHORT


def test_the_japanese_route_is_described_as_the_direct_model() -> None:
    """ja→zh ships one 145 MB direct model; the two-hop chain is only a fallback.

    The menu used to say 「经英文中转，两段」, which described the fallback as if it
    were the shipped path — and the status flash inherited that wording once the
    two labels were unified.
    """
    label = spec_of("ja-zh").label if spec_of("ja-zh") else ""
    assert "经英文中转" not in label
    assert "直译" in label
    assert models_for("ja-zh") == ("ja-zh",)
    assert fallback_of("ja-zh") == ("ja-en", "en-zh")  # pinned: the rescue chain


def test_config_accepts_every_catalogue_direction() -> None:
    """A route in the catalogue must pass config validation without a second edit.

    ``_TRANSLATE_TARGETS``/``_TRANSLATE_SOURCES`` are runtime whitelists: when they
    were hand-written, a direction offered by the menu could be rejected on the
    next load ("translate_target must be one of: en, zh").
    """
    from app import config as config_module  # noqa: PLC0415 - needs the module, not a name

    assert {spec.target for spec in ROUTES} == set(config_module._TRANSLATE_TARGETS)
    assert {spec.source for spec in ROUTES} == set(config_module._TRANSLATE_SOURCES)


def test_config_whitelists_are_projected_not_copied() -> None:
    """Values alone cannot prove it: today's targets happen to be ``en``/``zh``.

    A hand-written set would pass an equality check and still reject the next
    direction the catalogue gains, so the mechanism is pinned too.
    """
    from app import config as config_module  # noqa: PLC0415 - need the source path

    source = Path(config_module.__file__).read_text(encoding="utf-8")
    assert "spec.target for spec in ROUTES" in source
    assert "spec.source for spec in ROUTES" in source

"""Pure unit tests for the calm listening-breath helpers (no display)."""

import pytest
from app.pulse import (
    BREATH_MS,
    SHADE_STEPS,
    blend_hex,
    blend_rgb,
    breath_phase,
    quantize_shade,
)


def test_breath_starts_and_ends_at_the_trough() -> None:
    assert breath_phase(0) == pytest.approx(0.0)
    assert breath_phase(BREATH_MS) == pytest.approx(0.0)


def test_breath_peaks_halfway_through_the_cycle() -> None:
    assert breath_phase(BREATH_MS // 2) == pytest.approx(1.0)


def test_breath_is_slow_and_stays_in_range() -> None:
    samples = [breath_phase(ms) for ms in range(0, 2 * BREATH_MS, 40)]
    assert all(0.0 <= value <= 1.0 for value in samples)
    assert min(samples) == pytest.approx(0.0, abs=0.01)
    assert max(samples) == pytest.approx(1.0, abs=0.01)


def test_breath_moves_smoothly_not_as_a_two_state_toggle() -> None:
    # A toggle would jump 0 -> 1; a breath steps gently between frames.
    previous = breath_phase(0)
    for ms in range(40, BREATH_MS, 40):
        current = breath_phase(ms)
        assert abs(current - previous) < 0.2
        previous = current


def test_breath_period_of_zero_is_flat() -> None:
    assert breath_phase(123, period_ms=0) == 0.0


def test_blend_hex_returns_the_endpoints() -> None:
    assert blend_hex("#000000", "#ffffff", 0.0) == "#000000"
    assert blend_hex("#000000", "#ffffff", 1.0) == "#ffffff"


def test_blend_hex_midpoint_and_clamping() -> None:
    assert blend_hex("#000000", "#ffffff", 0.5) == "#808080"
    assert blend_hex("#000000", "#ffffff", -5.0) == "#000000"
    assert blend_hex("#000000", "#ffffff", 5.0) == "#ffffff"


def test_blend_rgb_interpolates_each_channel() -> None:
    assert blend_rgb((0.0, 0.0, 0.0), (1.0, 0.5, 0.25), 0.5) == (0.5, 0.25, 0.125)


def test_quantize_shade_keeps_the_endpoints() -> None:
    assert quantize_shade(0.0) == 0.0
    assert quantize_shade(1.0) == 1.0


def test_quantize_shade_snaps_to_even_steps() -> None:
    assert quantize_shade(0.5) == 0.5
    assert quantize_shade(0.1) == 1 / SHADE_STEPS
    assert quantize_shade(0.6) == 5 / SHADE_STEPS


def test_quantize_shade_clamps_out_of_range_input() -> None:
    assert quantize_shade(-5.0) == 0.0
    assert quantize_shade(5.0) == 1.0


def test_quantize_shade_with_zero_steps_is_identity() -> None:
    assert quantize_shade(0.37, steps=0) == pytest.approx(0.37)


def test_quantize_shade_collapses_a_breath_to_a_few_shades() -> None:
    shades = {quantize_shade(breath_phase(ms)) for ms in range(0, BREATH_MS, 10)}
    assert len(shades) <= SHADE_STEPS + 1

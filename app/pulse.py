"""Pure helpers for the calm "listening" breath animation.

Both platform indicators (the Windows Tk pill and the macOS AppKit panel)
share this math so they breathe identically. The module is dependency-free
(no Tk, no AppKit, no numpy) and every function is pure, so the animation
curve is unit-testable without a display.
"""

import math

#: One full breath, trough to trough. Slow enough to read as "calm", never a
#: blink: mainstream "always listening" cues (Meet/Zoom unmuted, macOS mic)
#: breathe over roughly 1.5-2.5 s.
BREATH_MS: int = 2000
#: How far a loud voice lifts the breath above its resting sine (0..1 scale).
LEVEL_GAIN: float = 0.30
#: Per-frame decay of the sampled mic level, so the breath settles when silent.
LEVEL_DECAY: float = 0.85
#: Discrete shades the breath snaps to. Adjacent 1/8 shades of the accent are
#: indistinguishable at this size, so quantising lets a tick skip the widget
#: reconfigure on most frames (~8 updates per breath, not ~50) while the eye
#: still sees a smooth cycle.
SHADE_STEPS: int = 8

_TAU: float = 2.0 * math.pi


def breath_phase(elapsed_ms: int, period_ms: int = BREATH_MS) -> float:
    """Sinusoidal 0..1 ease for a slow breathing animation.

    0.0 at the trough (exhale), 1.0 at the crest (inhale). The motion is
    continuous, so the eye sees a breath rather than a two-state toggle.
    """
    if period_ms <= 0:
        return 0.0
    fraction = (elapsed_ms % period_ms) / period_ms
    return (1.0 - math.cos(_TAU * fraction)) / 2.0


def quantize_shade(intensity: float, steps: int = SHADE_STEPS) -> float:
    """Snap a 0..1 breath intensity to one of *steps* even shades.

    Quantising the breath lets the indicator reconfigure its colour only when
    the shade actually changes, instead of on every frame. ``steps <= 0``
    disables quantisation (returns the clamped input).
    """
    if steps <= 0:
        return _clamp(intensity)
    return round(_clamp(intensity) * steps) / steps


def blend_hex(start: str, end: str, t: float) -> str:
    """Linear RGB blend between two ``#rrggbb`` colours; *t* clamps to 0..1."""
    ratio = _clamp(t)
    start_rgb = _hex_channels(start)
    end_rgb = _hex_channels(end)
    mixed = tuple(
        round(a + (b - a) * ratio) for a, b in zip(start_rgb, end_rgb, strict=True)
    )
    return f"#{mixed[0]:02x}{mixed[1]:02x}{mixed[2]:02x}"


def blend_rgb(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    t: float,
) -> tuple[float, float, float]:
    """Linear blend between two float RGB triples; *t* clamps to 0..1."""
    ratio = _clamp(t)
    return (
        start[0] + (end[0] - start[0]) * ratio,
        start[1] + (end[1] - start[1]) * ratio,
        start[2] + (end[2] - start[2]) * ratio,
    )


def _clamp(t: float) -> float:
    return min(1.0, max(0.0, t))


def _hex_channels(color: str) -> tuple[int, int, int]:
    return int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)

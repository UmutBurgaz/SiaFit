"""Rest-frame wavelength coverage preflight for feature fits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RequiredInterval:
    feature: str
    lower: float
    upper: float


def describe_requirements(intervals: list[RequiredInterval], redshift: float) -> str:
    """Describe every required interval in the rest and observed frames."""
    return " | ".join(
        f"{interval.feature}: rest {interval.lower:.1f}–{interval.upper:.1f} Å, "
        f"observed {interval.lower * (1.0 + redshift):.1f}–"
        f"{interval.upper * (1.0 + redshift):.1f} Å"
        for interval in intervals
    )


def validate_coverage(
    rest_wavelengths,
    redshift: float,
    intervals: list[RequiredInterval],
    *,
    max_gap_factor: float = 5.0,
    edge_spacing_factor: float = 1.5,
    minimum_pixels: int = 8,
) -> None:
    """Require sampled endpoints and no large internal gaps in each interval.

    Each feature supplies independent rest-frame intervals. Gaps *between*
    intervals are intentionally ignored.
    """
    wavelengths = np.unique(np.asarray(rest_wavelengths, dtype=float))
    wavelengths = wavelengths[np.isfinite(wavelengths)]
    for interval in intervals:
        lower, upper = interval.lower, interval.upper
        sampled = wavelengths[(wavelengths >= lower) & (wavelengths <= upper)]
        reason = None
        if sampled.size < minimum_pixels:
            reason = f"only {sampled.size} valid pixels (need at least {minimum_pixels})"
        else:
            gaps = np.diff(sampled)
            spacing = float(np.median(gaps))
            if sampled[0] - lower > edge_spacing_factor * spacing:
                reason = f"blue edge is missing by {sampled[0] - lower:.1f} Å"
            elif upper - sampled[-1] > edge_spacing_factor * spacing:
                reason = f"red edge is missing by {upper - sampled[-1]:.1f} Å"
            elif np.max(gaps) > max_gap_factor * spacing:
                reason = (
                    f"internal gap of {np.max(gaps):.1f} Å exceeds "
                    f"{max_gap_factor:g} times the median spacing ({spacing:.1f} Å)"
                )
        if reason is not None:
            observed_lower = lower * (1.0 + redshift)
            observed_upper = upper * (1.0 + redshift)
            raise ValueError(
                f"{interval.feature} coverage failed: required rest-frame "
                f"{lower:.1f}–{upper:.1f} Å; at redshift {redshift:.6g}, "
                f"required observed-frame {observed_lower:.1f}–{observed_upper:.1f} Å; "
                f"{reason}"
            )

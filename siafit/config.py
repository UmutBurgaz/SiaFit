"""Configuration objects for public pipeline entry points."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real


@dataclass(frozen=True)
class FitConfig:
    """Numerical and workflow settings shared by all pipeline stages."""

    mode: str = "auto"
    features: str = "si"
    input_format: str = "auto"
    vexp: float | None = None
    clip_outliers: bool = True
    vel_width: float = 3.0
    sil6355_red_bounds_kms: tuple[float, float] = (-15_000.0, 15_000.0)
    sil6355_blue_bounds_kms: tuple[float, float] = (-40_000.0, -15_000.0)
    sil5972_blue_bounds_kms: tuple[float, float] = (-30_000.0, -10_000.0)
    ca_coverage_bounds_kms: tuple[float, float] = (-45_000.0, 13_000.0)
    ca_red_bounds_kms: tuple[float, float] = (-5_000.0, 25_000.0)
    ca_blue_bounds_kms: tuple[float, float] = (-55_000.0, -20_000.0)
    ca_n_resol_neighbors: int = 3
    ca_hvf_min_sep_kms: float = 2.0  # Legacy name: value is in 10³ km/s, not km/s.
    ca_pvf_vel_frac: float = 0.25
    si_velocity: float | None = None
    n_resol_neighbors: int = 3
    k_vel12_bounds: tuple[float, float] = (0.60, 1.30)
    k_fwhm12_bounds: tuple[float, float] = (0.60, 1.25)
    n_iterations: int = 100
    mc_mode: str = "resample"
    continuum_shift: float = 1.3
    seed: int | None = None
    redchi_warning_threshold: float | None = 10.0
    # Tangent (shoulder) anchors. Velocities are in 10³ km/s; prominence and
    # near are fractions of the trough depth below the continuum line.
    tangent_prominence: float = 0.05
    tangent_weak_prominence: float = 0.015
    tangent_shoulder_reach: tuple[float, float] = (16.0, 18.0)
    tangent_ca_shoulder_reach: tuple[float, float] = (10.0, 8.0)
    tangent_near: float = 0.35
    tangent_si_trough_range: tuple[float, float] = (-32.0, -5.5)
    tangent_si_trough_limits: tuple[float, float] = (-38.0, -0.5)
    tangent_search_span: tuple[float, float] = (26.0, 26.0)
    tangent_si5972_reach: float = 20.0
    tangent_si5972_min_width: float = 5.0
    tangent_ca_core_hvf: float = 6.0
    tangent_ca_reach: tuple[float, float] = (22.0, 16.0)
    tangent_ca_min_gap: float = 1.0

    def validate(self) -> None:
        def finite_number(name: str, value: object) -> None:
            if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
                raise ValueError(f"{name} must be a finite number")

        def integer_at_least(name: str, value: object, minimum: int) -> None:
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer at least {minimum}")

        def ordered_bounds(name: str, values: object, *, positive: bool = False) -> None:
            try:
                lower, upper = values
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must contain two numeric bounds") from exc
            finite_number(f"{name} lower bound", lower)
            finite_number(f"{name} upper bound", upper)
            if lower >= upper:
                raise ValueError(f"{name} lower bound must be less than its upper bound")
            if positive and lower <= 0:
                raise ValueError(f"{name} bounds must be positive")
            if not positive and (lower <= -299_792.0 or upper >= 299_792.0):
                raise ValueError(f"{name} bounds must lie strictly between ±299792 km/s")

        if self.mode not in {"auto", "review", "manual"}:
            raise ValueError("mode must be 'auto', 'review', or 'manual'")
        if self.features not in {"si", "ca", "all"}:
            raise ValueError("features must be 'si', 'ca', or 'all'")
        if self.input_format not in {"auto", "flux", "flux-error"}:
            raise ValueError("input_format must be 'auto', 'flux', or 'flux-error'")
        positive_settings = {
            "vel_width": self.vel_width,
            "ca_hvf_min_sep_kms": self.ca_hvf_min_sep_kms,
            "ca_pvf_vel_frac": self.ca_pvf_vel_frac,
        }
        if self.vexp is not None:
            positive_settings["vexp"] = self.vexp
        for name, value in positive_settings.items():
            finite_number(name, value)
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "sil6355_red_bounds_kms",
            "sil6355_blue_bounds_kms",
            "sil5972_blue_bounds_kms",
            "ca_coverage_bounds_kms",
            "ca_red_bounds_kms",
            "ca_blue_bounds_kms",
        ):
            ordered_bounds(name, getattr(self, name))
        for name in ("k_vel12_bounds", "k_fwhm12_bounds"):
            ordered_bounds(name, getattr(self, name), positive=True)
        integer_at_least("n_resol_neighbors", self.n_resol_neighbors, 2)
        integer_at_least("ca_n_resol_neighbors", self.ca_n_resol_neighbors, 2)
        integer_at_least("n_iterations", self.n_iterations, 0)
        if self.seed is not None:
            integer_at_least("seed", self.seed, 0)
        if self.mc_mode not in {"resample", "shift", "shift_resample", "all"}:
            raise ValueError("invalid mc_mode")
        finite_number("continuum_shift", self.continuum_shift)
        if self.continuum_shift < 0:
            raise ValueError("continuum_shift cannot be negative")
        if self.si_velocity is not None:
            finite_number("si_velocity", self.si_velocity)
        for name in ("tangent_prominence", "tangent_weak_prominence", "tangent_near",
                     "tangent_si5972_reach", "tangent_si5972_min_width",
                     "tangent_ca_core_hvf", "tangent_ca_min_gap"):
            value = getattr(self, name)
            finite_number(name, value)
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.tangent_weak_prominence > self.tangent_prominence:
            raise ValueError("tangent_weak_prominence cannot exceed tangent_prominence")
        for name in ("tangent_shoulder_reach", "tangent_ca_shoulder_reach",
                     "tangent_search_span", "tangent_ca_reach"):
            try:
                blue, red = getattr(self, name)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must contain a blue and a red distance") from exc
            for value in (blue, red):
                finite_number(name, value)
                if value <= 0:
                    raise ValueError(f"{name} distances must be positive")
        for name in ("tangent_si_trough_range", "tangent_si_trough_limits"):
            try:
                lower, upper = getattr(self, name)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must contain two velocities") from exc
            finite_number(name, lower)
            finite_number(name, upper)
            if not lower < upper < 0:
                raise ValueError(f"{name} must be two increasing negative velocities")
        outer_lower, outer_upper = self.tangent_si_trough_limits
        inner_lower, inner_upper = self.tangent_si_trough_range
        if not outer_lower <= inner_lower < inner_upper <= outer_upper:
            raise ValueError("tangent_si_trough_limits must contain tangent_si_trough_range")
        if self.redchi_warning_threshold is not None:
            finite_number("redchi_warning_threshold", self.redchi_warning_threshold)
            if self.redchi_warning_threshold <= 0:
                raise ValueError("redchi_warning_threshold must be positive or None")

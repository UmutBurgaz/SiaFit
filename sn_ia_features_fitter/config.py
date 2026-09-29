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
    vel_width: float = 4.0
    sil6355_red_initial: float = 0.0
    sil6355_blue_initial: float = -20.0
    sil5972_blue_initial: float = -20.0
    background_search_window_kms: float = 20_000.0
    background_search_step_kms: float = 1_000.0
    background_search_attempts: int = 5
    sil6355_red_bounds_kms: tuple[float, float] = (-15_000.0, 15_000.0)
    sil6355_blue_bounds_kms: tuple[float, float] = (-40_000.0, -15_000.0)
    sil5972_blue_bounds_kms: tuple[float, float] = (-30_000.0, -10_000.0)
    ca_red_initial: float = 3.0
    ca_blue_initial: float = -35.0
    ca_background_search_window_kms: float = 10_000.0
    ca_background_search_step_kms: float = 1_000.0
    ca_background_search_attempts: int = 5
    ca_red_bounds_kms: tuple[float, float] = (-5_000.0, 25_000.0)
    ca_blue_bounds_kms: tuple[float, float] = (-55_000.0, -20_000.0)
    ca_n_resol_neighbors: int = 3
    ca_hvf_min_sep_kms: float = 2.0
    ca_pvf_vel_frac: float = 0.25
    si_velocity: float | None = None
    n_resol_neighbors: int = 3
    k_vel12_bounds: tuple[float, float] = (0.60, 1.30)
    k_fwhm12_bounds: tuple[float, float] = (0.60, 1.25)
    n_iterations: int = 100
    mc_mode: str = "resample"
    continuum_shift: float = 1.3
    seed: int | None = None

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
            "background_search_window_kms": self.background_search_window_kms,
            "background_search_step_kms": self.background_search_step_kms,
            "ca_background_search_window_kms": self.ca_background_search_window_kms,
            "ca_background_search_step_kms": self.ca_background_search_step_kms,
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
            "sil6355_red_initial",
            "sil6355_blue_initial",
            "sil5972_blue_initial",
            "ca_red_initial",
            "ca_blue_initial",
        ):
            value = getattr(self, name)
            finite_number(name, value)
            if abs(value) >= 299.792:
                raise ValueError(f"{name} must lie strictly between ±299.792 in 10³ km/s")
        for name in (
            "sil6355_red_bounds_kms",
            "sil6355_blue_bounds_kms",
            "sil5972_blue_bounds_kms",
            "ca_red_bounds_kms",
            "ca_blue_bounds_kms",
        ):
            ordered_bounds(name, getattr(self, name))
        for name, bounds_name in (
            ("ca_red_initial", "ca_red_bounds_kms"),
            ("ca_blue_initial", "ca_blue_bounds_kms"),
        ):
            velocity_kms = getattr(self, name) * 1000.0
            lower, upper = getattr(self, bounds_name)
            if not lower <= velocity_kms <= upper:
                raise ValueError(f"{name} must lie within {bounds_name}")
        if self.ca_blue_initial >= self.ca_red_initial:
            raise ValueError("ca_blue_initial must be lower than ca_red_initial")
        for name in ("k_vel12_bounds", "k_fwhm12_bounds"):
            ordered_bounds(name, getattr(self, name), positive=True)
        integer_at_least("background_search_attempts", self.background_search_attempts, 0)
        integer_at_least("ca_background_search_attempts", self.ca_background_search_attempts, 0)
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

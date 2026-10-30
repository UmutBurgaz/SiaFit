"""Fit diagnostics and review flags, without changing model selection or acceptance."""

from __future__ import annotations

import numpy as np


def optimizer_diagnostics(result):
    """Report free parameters within 1e-6 relative tolerance of a finite bound.

    The absolute floor is 1e-6 in each parameter's units. A finite covariance
    estimate does not make boundary-limited uncertainties reliable.
    """
    parameters = {name: par for name, par in result.params.items() if par.vary and not par.expr}
    flags = {
        f"at_bound_{name}": int(any(
            np.isfinite(bound)
            and abs(par.value - bound) <= 1e-6 * max(1.0, abs(bound))
            for bound in (par.min, par.max)
        ))
        for name, par in parameters.items()
    }
    covariance_available = bool(
        getattr(result, "errorbars", False)
        and all(par.stderr is not None and np.isfinite(par.stderr) for par in parameters.values())
    )
    return {
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_nfev": result.nfev,
        "optimizer_redchi": result.redchi,
        "optimizer_chisqr": result.chisqr,
        "covariance_available": int(covariance_available),
        "any_at_bound": int(any(flags.values())),
        **flags,
    }


def mc_boundary_summary(results, suffix):
    """Fractions use successful optimizer draws only; no successes means NaN."""
    return {
        f"{name}_fraction_{suffix}": float(np.mean(values)) if len(values) else np.nan
        for name, values in results.items()
        if name.startswith("at_bound_") or name == "any_at_bound"
    }


def review_flags(metrics, mc_summary, uncertainty_source, redchi_threshold, *, prefix=""):
    """Explain review triggers. These flags do not reject or remove measurements."""
    reasons = []
    if metrics.get('sil5972_fit_success') == 0:
        reasons.append('secondary_line_fit_unavailable')
    elif metrics.get('sil5972_profile_review_flag'):
        reasons.append('weak_secondary_profile')
    velocity_error = metrics.get('sil6355_vel_err')
    primary_snr = metrics.get('sil6355_snr')
    if primary_snr is not None and np.isfinite(primary_snr) and primary_snr < 2:
        reasons.append('weak_primary_line')
    if velocity_error is not None and np.isfinite(velocity_error) and velocity_error > 3:
        reasons.append('large_primary_velocity_uncertainty')
    if metrics.get(f"{prefix}any_at_bound", 0):
        reasons.append("parameter_at_bound")
    if metrics.get(f"{prefix}covariance_available") == 0:
        reasons.append("covariance_unavailable")
    redchi = metrics.get(f"{prefix}optimizer_redchi", np.nan)
    if redchi_threshold is not None and np.isfinite(redchi) and redchi > redchi_threshold:
        reasons.append("high_reduced_chisquare")
    if uncertainty_source in {"estimated", "mixed"}:
        reasons.append("estimated_uncertainties")
    for name, value in mc_summary.items():
        if name.startswith("any_at_bound_fraction_") and np.isfinite(value) and value > 0:
            reasons.append("mc_parameter_at_bound")
            break
    requested = mc_summary.get("mc_n_iter", 0)
    if requested:
        counts = [value for name, value in mc_summary.items() if name.startswith("n_success_")]
        if not counts or any(value / requested < 0.8 for value in counts):
            reasons.append("low_mc_success_fraction")
    return {
        f"{prefix}fit_qc_flag": int(bool(reasons)),
        f"{prefix}fit_qc_reasons": " | ".join(reasons),
        f"{prefix}redchi_warning_threshold": redchi_threshold,
    }

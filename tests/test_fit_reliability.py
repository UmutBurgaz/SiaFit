"""Regression coverage for optimizer failures and constraint-limited measurements."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from lmfit import Parameters

from siafit import FitConfig, ca, combined, common, core, run_catalog
from siafit.catalog import load_catalog
from siafit.quality import optimizer_diagnostics, review_flags

ROOT = Path(__file__).resolve().parents[1]


def ca_spectrum(pvf_velocity=-10.0):
    wavelength = np.linspace(7200.0, 9100.0, 1200)
    return pd.DataFrame({
        "rest_wl": wavelength,
        "norm_fl": 1.0 + ca.ca_model_total(
            wavelength, -0.065, pvf_velocity, 7.0, -0.085, -21.0, 8.0,
        ),
        "norm_flerr": np.full(wavelength.size, 0.015),
        "vel_ca_ref": ca.wl_to_vel(wavelength, ca.rest_wl_ca_ref),
    })


BACKGROUND = {"ca_red_side": 3.0, "ca_blue_side": -35.0}


def test_failed_ca_optimizer_is_rejected_by_best_fit_and_mc(monkeypatch):
    original = ca.minimize
    successes = []

    def stop_early(*args, **kwargs):
        result = original(*args, **kwargs, max_nfev=1)
        successes.append(result.success)
        return result

    monkeypatch.setattr(ca, "minimize", stop_early)
    spectrum = ca_spectrum()
    with pytest.raises(RuntimeError, match="Ca II optimizer failed"):
        ca.fit_features(spectrum, BACKGROUND, -10.0)
    summary, figure = ca.fit_features_mc(spectrum, BACKGROUND, -10.0, n_iter=2, seed=42)
    try:
        assert successes == [False, False, False]
        assert summary["n_success_resample"] == 0
        assert summary["n_fail_resample"] == 2
        assert "Ca II optimizer failed" in summary["failure_reasons_resample"]
        assert np.isnan(summary["at_bound_v_pvf_fraction_resample"])
    finally:
        ca.plt.close(figure)


def test_boundary_limited_injection_is_flagged_in_best_fit_and_mc():
    # True PVF is outside the deliberately retained +/-25% Si constraint.
    spectrum = ca_spectrum(pvf_velocity=-6.0)
    metrics, figure, *_ = ca.fit_features(spectrum, BACKGROUND, -10.0)
    summary, mc_figure = ca.fit_features_mc(
        spectrum, BACKGROUND, -10.0, n_iter=2, mode="shift", continuum_shift=0, seed=42,
    )
    try:
        assert metrics["ca_pvf_vel"] == pytest.approx(-7.5, abs=1e-5)
        assert metrics["ca_at_bound_v_pvf"] == 1
        assert summary["at_bound_v_pvf_fraction_shift"] == 1.0
        assert summary["ca_pvf_vel_med_shift"] == pytest.approx(metrics["ca_pvf_vel"], abs=1e-10)
        assert summary["ca_hvf_ew_med_shift"] == pytest.approx(metrics["ca_hvf_ew"], abs=1e-10)
    finally:
        ca.plt.close(figure)
        ca.plt.close(mc_figure)


@pytest.mark.parametrize("problem", ["zero_continuum", "too_few_points", "bad_uncertainty"])
def test_ca_numerical_validation_is_shared_by_best_fit_and_mc(problem):
    spectrum = ca_spectrum()
    background = dict(BACKGROUND)
    if problem == "zero_continuum":
        spectrum["norm_fl"] = 0.0
        message = "continuum fit must be finite and positive"
    elif problem == "too_few_points":
        background = {"ca_blue_side": -10.01, "ca_red_side": -9.99}
        message = "Too few Ca fit points"
    else:
        spectrum["norm_flerr"] = 0.0
        message = "uncertainties must be positive"
    with pytest.raises(ValueError, match=message):
        ca.fit_features(spectrum, background, -10.0)
    summary, figure = ca.fit_features_mc(
        spectrum, background, -10.0, n_iter=1, mode="shift", continuum_shift=0,
    )
    try:
        assert summary["n_success_shift"] == 0
        assert summary["n_fail_shift"] == 1
        assert message in summary["failure_reasons_shift"]
    finally:
        ca.plt.close(figure)


def test_bound_diagnostics_do_not_confuse_finite_stderr_with_reliability():
    from types import SimpleNamespace

    params = Parameters()
    params.add("velocity", value=-7.5, min=-12.5, max=-7.5)
    params["velocity"].stderr = 0.4
    params.add("tied", expr="velocity")
    result = SimpleNamespace(params=params, errorbars=True, success=True, message="ok",
                             nfev=10, redchi=1.0, chisqr=20.0)
    diagnostics = optimizer_diagnostics(result)
    assert diagnostics["covariance_available"] == 1
    assert diagnostics["at_bound_velocity"] == 1
    assert "at_bound_tied" not in diagnostics
    params["velocity"].value = -10
    assert optimizer_diagnostics(result)["any_at_bound"] == 0


def test_review_flags_are_explanatory_and_threshold_is_configurable():
    metrics = {"optimizer_redchi": 887, "any_at_bound": 1, "covariance_available": 1}
    mc = {"mc_n_iter": 20, "n_success_resample": 20, "any_at_bound_fraction_resample": 1.0}
    flags = review_flags(metrics, mc, "provided", 10)
    assert flags["fit_qc_flag"] == 1
    assert "high_reduced_chisquare" in flags["fit_qc_reasons"]
    assert "mc_parameter_at_bound" in flags["fit_qc_reasons"]
    assert "low_mc_success" not in flags["fit_qc_reasons"]
    assert "estimated_uncertainties" not in flags["fit_qc_reasons"]
    disabled = review_flags(metrics, mc, "mixed", None)
    assert "high_reduced_chisquare" not in disabled["fit_qc_reasons"]
    assert "estimated_uncertainties" in disabled["fit_qc_reasons"]


def test_mc_failed_draw_does_not_partially_append_metrics():
    results, _, failures, reasons = common.collect_mc_draws(
        ["first", "missing"], ["resample"], 1, lambda mode: {"first": 12.0},
    )
    assert results["resample"]["first"].size == 0
    assert failures["resample"] == 1
    assert any("KeyError" in reason for reason in reasons["resample"])


def test_joint_example_records_scientific_review_flags_without_rejecting_row(tmp_path):
    summary = run_catalog(
        ROOT / "example_catalog.csv", tmp_path,
        config=FitConfig(features="all", n_iterations=2, seed=42, anchor_strategy="legacy"), start=2, end=3,
    )
    assert summary.succeeded == 1
    row = pd.read_csv(tmp_path / "fit_results.csv").iloc[0]
    assert row["best_fit_ca_optimizer_success"]
    assert row["best_fit_ca_at_bound_v_pvf"] == 1
    assert row["best_fit_ca_fit_qc_flag"] == 1
    assert "high_reduced_chisquare" in row["best_fit_ca_fit_qc_reasons"]
    assert row["best_fit_ca_optimizer_redchi"] == pytest.approx(886.918934965, rel=1e-5)
    assert row["best_fit_ca_pvf_vel"] == pytest.approx(-7.34129935667, abs=1e-5)
    assert row["ca_at_bound_v_pvf_fraction_resample"] == 1.0
    assert row["best_fit_ca_uncertainty_conditioning"] == "fixed_si_velocity"
    assert "best_fit_at_bound_v_sil1_1" in row.index


@pytest.mark.parametrize("settings", [
    {"sil6355_red_initial": 100.0},
    {"sil6355_blue_initial": -10.0},
    {"sil5972_blue_initial": -35.0},
    {"redchi_warning_threshold": 0.0},
    {"redchi_warning_threshold": float("nan")},
])
def test_invalid_review_configuration_fails_early(settings):
    with pytest.raises(ValueError, match=next(iter(settings))):
        FitConfig(**settings).validate()


@pytest.mark.parametrize("module", [core, ca])
def test_direct_fit_defaults_agree_with_pipeline_configuration(module):
    import inspect

    config = FitConfig()
    fields = {"n_resol_neighbors": "n_resol_neighbors" if module is core else "ca_n_resol_neighbors"}
    if module is core:
        fields.update(k_vel12_bounds="k_vel12_bounds", k_fwhm12_bounds="k_fwhm12_bounds")
    else:
        fields.update(hvf_min_sep_kms="ca_hvf_min_sep_kms", pvf_vel_frac="ca_pvf_vel_frac")
    for function in (module.fit_features, module.fit_features_mc):
        parameters = inspect.signature(function).parameters
        for argument, field in fields.items():
            assert parameters[argument].default == getattr(config, field)
    assert inspect.signature(module.fit_features_mc).parameters["n_iter"].default == config.n_iterations


def test_combined_background_resume_retries_latest_failed_attempt(tmp_path, monkeypatch):
    records = load_catalog(ROOT / "example_catalog.csv")[:2]
    pd.DataFrame([{"record_id": record.record_id} for record in records]).to_csv(
        tmp_path / "backgrounds.csv", index=False,
    )
    pd.DataFrame([
        {"record_id": records[0].record_id, "success": 0},
        {"record_id": records[1].record_id, "success": 1},
    ]).to_csv(tmp_path / "background_log.csv", index=False)
    attempted = []

    def preparation_failure(record, config):
        attempted.append(record.record_id)
        raise ValueError("deliberate retry probe")

    monkeypatch.setattr(combined.si, "_prepare", preparation_failure)
    summary = run_catalog(
        ROOT / "example_catalog.csv", tmp_path, end=2, stage="background", resume=True,
        config=FitConfig(features="ca", n_iterations=0),
    )
    assert attempted == [records[0].record_id]
    assert (summary.failed, summary.skipped) == (1, 1)

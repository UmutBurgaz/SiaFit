"""Ca physics, shared infrastructure, and joint workflow regressions."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sn_ia_features_fitter import FitConfig, ca, common, core, fit_background_catalog, run_catalog
from sn_ia_features_fitter.coverage import validate_coverage

ROOT = Path(__file__).resolve().parents[1]


def test_ca_triplet_recovers_synthetic_pvf_and_hvf():
    wavelength = np.linspace(7200.0, 9100.0, 1200)
    flux = 1.0 + ca.ca_model_total(
        wavelength, -0.065, -10.0, 7.0, -0.085, -21.0, 8.0
    )
    spectrum = pd.DataFrame({
        "rest_wl": wavelength,
        "norm_fl": flux,
        "norm_flerr": np.full(wavelength.size, 0.015),
        "vel_ca_ref": ca.wl_to_vel(wavelength, ca.rest_wl_ca_ref),
    })
    background = {"ca_red_side": 3.0, "ca_blue_side": -35.0}
    metrics, figure, *_ = ca.fit_features(spectrum, background, -10.0)
    try:
        # Values from the original Ca implementation on this exact input.
        assert metrics["ca_pvf_vel"] == pytest.approx(-10.0427458888, abs=1e-4)
        assert metrics["ca_hvf_vel"] == pytest.approx(-20.9790096853, abs=1e-4)
        assert metrics["ca_pvf_fwhm"] == pytest.approx(6.7252352581, abs=1e-4)
        assert metrics["ca_hvf_fwhm"] == pytest.approx(7.9545750065, abs=1e-4)
        assert metrics["ca_pvf_amp"] < 0
        assert metrics["ca_hvf_amp"] < 0
        for component in ("pvf", "hvf"):
            for label, rest_wl in (
                ("8498", ca.rest_wl_ca1), ("8542", ca.rest_wl_ca2),
                ("8662", ca.rest_wl_ca3),
            ):
                assert metrics[f"ca_{component}_{label}_fwhm_angstrom"] == pytest.approx(
                    common.velocity_fwhm_to_angstrom(
                        metrics[f"ca_{component}_vel"], metrics[f"ca_{component}_fwhm"], rest_wl
                    )
                )
        assert metrics["ca_hvf_ew"] > metrics["ca_pvf_ew"] > 0
    finally:
        ca.plt.close(figure)


def test_ca_uses_shared_reader_smoothing_and_mc_bins(tmp_path):
    assert ca.common.prepare_spectrum is core.common.prepare_spectrum
    assert ca._mc_plot_bin_edges is common._mc_plot_bin_edges
    path = tmp_path / "incomplete-first-lines.dat"
    wavelength = np.linspace(7000.0, 9300.0, 300)
    flux = 1.0 + 0.05 * np.sin(wavelength / 50.0)
    lines = ["6900", "# wavelength flux error"]
    lines += [f"{wl} {fl} 0.02" for wl, fl in zip(wavelength, flux)]
    path.write_text("\n".join(lines), encoding="utf-8")
    spectrum = ca.prepare_spectrum(path, 0.0, 0.0, vexp=0.004)
    assert len(spectrum) == 300
    assert spectrum["wl"].iloc[0] == 7000.0
    assert np.isfinite(spectrum["vel_ca_ref"]).all()
    assert len(common._mc_plot_bin_edges(np.r_[np.linspace(0, 1, 100), 1e15])) == 41


def test_feature_coverage_allows_gap_between_independent_intervals():
    config = FitConfig(features="all")
    intervals = core.required_rest_intervals(config) + ca.required_rest_intervals(config)
    sampled = np.r_[np.arange(5200.0, 6800.0, 10.0), np.arange(7200.0, 9000.0, 10.0)]
    validate_coverage(sampled, 0.04, intervals)


def test_feature_coverage_rejects_internal_gap_with_both_frames_in_error():
    sampled = np.r_[np.arange(5200.0, 5900.0, 10.0), np.arange(6200.0, 6800.0, 10.0)]
    with pytest.raises(ValueError, match="Si II coverage failed") as exc:
        validate_coverage(sampled, 0.04, core.required_rest_intervals(FitConfig()))
    assert "required rest-frame 5250.0–6750.0 Å" in str(exc.value)
    assert "required observed-frame 5460.0–7020.0 Å" in str(exc.value)
    assert "internal gap" in str(exc.value)


def test_coverage_is_feature_specific_and_tracks_search_configuration():
    si_config = FitConfig(features="si")
    ca_config = FitConfig(features="ca")
    si_only = np.arange(5200.0, 6800.0, 10.0)
    ca_only = np.arange(7200.0, 9000.0, 10.0)
    validate_coverage(si_only, 0.0, core.required_rest_intervals(si_config))
    validate_coverage(ca_only, 0.0, ca.required_rest_intervals(ca_config))
    with pytest.raises(ValueError, match="Ca II NIR coverage failed"):
        validate_coverage(si_only, 0.0, ca.required_rest_intervals(ca_config))
    with pytest.raises(ValueError, match="Si II coverage failed"):
        validate_coverage(ca_only, 0.0, core.required_rest_intervals(si_config))
    wider = replace(ca_config, ca_background_search_window_kms=15_000.0)
    assert ca.required_rest_intervals(wider)[0].upper > ca.required_rest_intervals(ca_config)[0].upper
    wider_si = replace(si_config, sil6355_red_bounds_kms=(-15_000.0, 25_000.0))
    assert core.required_rest_intervals(wider_si)[0].upper > core.required_rest_intervals(si_config)[0].upper


def test_joint_workflow_passes_selected_si_velocity_directly_to_ca(tmp_path, monkeypatch):
    passed = []
    original = ca.fit_features

    def record_velocity(df_spec, background, si_vel_1e3, *args, **kwargs):
        passed.append(si_vel_1e3)
        return original(df_spec, background, si_vel_1e3, *args, **kwargs)

    monkeypatch.setattr(ca, "fit_features", record_velocity)
    summary = run_catalog(
        ROOT / "example_catalog.csv", tmp_path / "joint",
        config=FitConfig(features="all", n_iterations=0), end=1,
    )
    assert summary.succeeded == 1
    result = pd.read_csv(summary.output_dir / "fit_results.csv").iloc[0]
    assert passed == [pytest.approx(result["best_fit_sil6355_vel"])]
    assert result["selected_si_velocity"] == pytest.approx(result["best_fit_sil6355_vel"])
    assert np.isfinite(result["best_fit_ca_pvf_vel"])
    assert np.isfinite(result["best_fit_sil6355_amp"])
    assert np.isfinite(result["best_fit_sil6355_6347_fwhm_angstrom"])
    assert np.isfinite(result["best_fit_ca_pvf_amp"])
    assert np.isfinite(result["best_fit_ca_pvf_8498_fwhm_angstrom"])
    assert "Si II: rest" in result["coverage_status"]
    assert "Ca II NIR: rest" in result["coverage_status"]
    assert "observed" in result["coverage_status"]


def test_ca_only_workflow_uses_catalogue_prior(tmp_path):
    summary = run_catalog(
        ROOT / "example_catalog.csv", tmp_path / "ca",
        config=FitConfig(features="ca", n_iterations=0), end=1,
    )
    assert summary.succeeded == 1
    result = pd.read_csv(summary.output_dir / "fit_results.csv").iloc[0]
    assert result["selected_si_velocity"] == pytest.approx(-15.432763599116)
    assert "best_fit_sil6355_vel" not in result.index


def test_joint_background_catalog_can_be_fit_in_second_stage(tmp_path):
    config = FitConfig(features="all", n_iterations=0)
    background = run_catalog(
        ROOT / "example_catalog.csv", tmp_path / "background",
        config=config, stage="background", end=1,
    )
    assert background.succeeded == 1
    fitted = fit_background_catalog(
        background.output_dir / "backgrounds.csv", tmp_path / "fit",
        config=config, end=1,
    )
    assert fitted.succeeded == 1
    result = pd.read_csv(fitted.output_dir / "fit_results.csv").iloc[0]
    assert result["selected_si_velocity"] == pytest.approx(result["best_fit_sil6355_vel"])
    assert np.isfinite(result["best_fit_ca_hvf_vel"])

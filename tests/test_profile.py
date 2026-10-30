"""Automatic profile recovery, weak-secondary isolation, and MC regression tests."""

import runpy
from pathlib import Path

import numpy as np
import pytest

from siafit import FitConfig, core, pipeline
from siafit.profile import find_background, fit_once

spectrum = runpy.run_path(str(Path(__file__).with_name('test_adaptive.py')))['spectrum']


@pytest.mark.parametrize('velocity', [-1., -2., -5., -11., -20., -28., -35.])
@pytest.mark.parametrize('width', [4., 8.])
def test_default_profiles_recover_fast_and_slow_injections(velocity, width):
    df = spectrum(velocity, width)
    bg, figure = find_background(df, FitConfig())
    metrics, fit_figure, *_ = core.fit_features(df, bg)
    try:
        assert metrics['sil6355_vel'] == pytest.approx(velocity, abs=.4)
        assert bg['anchor_strategy'] in {'profile', 'legacy'}
    finally:
        core.plt.close(figure)
        core.plt.close(fit_figure)


def test_missing_secondary_cannot_move_primary_velocity():
    df = spectrum(-25, 8)
    w = df.rest_wl.to_numpy()
    flux = 1 + core.g_sil1_1(w, -.2, -25, 8) + core.g_sil1_2(w, -.2, -25, 8)
    df['norm_fl'] = flux
    df['norm_fl_smooth_flattened'] = flux
    df['norm_fl_flattened'] = flux
    bg, figure = find_background(df, FitConfig())
    try:
        metrics, primary, _ = fit_once(df, bg)
        assert metrics['sil6355_vel'] == pytest.approx(-25, abs=.4)
        assert metrics['sil5972_profile_review_flag'] == 1
        assert np.sum(((primary['flux']-primary['model'])/primary['error'])**2) == pytest.approx(primary['fit'].chisqr)
    finally:
        core.plt.close(figure)


def test_profile_monte_carlo_and_pipeline_default():
    assert FitConfig().anchor_strategy == 'profile'
    df = spectrum(-25, 8)
    selection = pipeline.select_background(df, mode='auto', config=FitConfig())
    metrics, figure, summary, mc_figure = pipeline._fit_spectrum(
        df, selection.background, 4., FitConfig(n_iterations=3), seed=42,
    )
    try:
        assert summary['n_success_resample'] == 3
        assert summary['sil6355_vel_med_resample'] == pytest.approx(-25, abs=.8)
        assert metrics['si_fit_method'] == 'independent_profiles'
        assert np.isfinite(summary['sil6355_vel_std_resample'])
    finally:
        core.plt.close('all')


def test_numerically_zero_legacy_line_cannot_exclude_fast_profile():
    df = spectrum(-35, 4)
    bg, figure = find_background(df, FitConfig())
    try:
        metrics, fit_figure, *_ = core.fit_features(df, bg)
        assert metrics['sil6355_vel'] == pytest.approx(-35, abs=.4)
        core.plt.close(fit_figure)
    finally:
        core.plt.close(figure)


@pytest.mark.parametrize('velocity', [-1., -35.])
def test_extreme_velocity_through_reader_clipping_and_default_catalogue(tmp_path, velocity):
    from siafit import run_catalog

    df = spectrum(velocity, 6.)
    rng = np.random.default_rng(9)
    path = tmp_path / 'injection.dat'
    np.savetxt(path, np.column_stack((df.rest_wl.to_numpy()*1.03,
                                      df.norm_fl.to_numpy()+rng.normal(0, .005, len(df)),
                                      df.norm_flerr.to_numpy())))
    catalog = tmp_path / 'catalog.csv'
    import pandas as pd

    pd.DataFrame([dict(spectrum=str(path), redshift=.03, mwebv=0)]).to_csv(catalog, index=False)
    result = run_catalog(catalog, tmp_path/'results', config=FitConfig(n_iterations=2, seed=42))
    assert result.succeeded == 1 and result.failed == 0
    row = pd.read_csv(tmp_path/'results/fit_results.csv').iloc[0]
    assert row.best_fit_sil6355_vel == pytest.approx(velocity, abs=.5)
    assert row.mc_success_fraction == 1

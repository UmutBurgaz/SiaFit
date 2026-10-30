"""Velocity-extreme and saved-background regression tests for adaptive selection."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from siafit import FitConfig, core, pipeline
from siafit.adaptive import find_background
from siafit.catalog import SpectrumRecord


def spectrum(velocity, width=6):
    w = np.arange(4700., 7100., 2.)
    flux = (1 + core.g_sil1_1(w, -.2, velocity, width)
            + core.g_sil1_2(w, -.2, velocity, width)
            + core.g_sil2_1(w, -.08, velocity, width * .85)
            + core.g_sil2_2(w, -.08, velocity, width * .85))
    return pd.DataFrame(dict(
        rest_wl=w, norm_fl=flux, norm_flerr=np.full(len(w), .01),
        norm_fl_flattened=flux, norm_fl_smooth_flattened=flux,
        vel_sil1_1=core.wl_to_vel(w, core.rest_wl_sil1_1),
        vel_sil2_1=core.wl_to_vel(w, core.rest_wl_sil2_1),
    ))


@pytest.mark.parametrize('velocity', [-5., -11., -20., -28.])
@pytest.mark.parametrize('width', [4., 8.])
def test_adaptive_recovers_injected_velocity(velocity, width):
    df = spectrum(velocity, width)
    bg, figure = find_background(df, FitConfig(anchor_strategy='adaptive'))
    metrics, fit_figure, *_ = core.fit_features(df, bg)
    try:
        assert bg['sil6355_blue_side'] < velocity < bg['sil6355_red_side']
        assert metrics['sil6355_vel'] == pytest.approx(velocity, abs=.3)
    finally:
        core.plt.close(figure)
        core.plt.close(fit_figure)


def test_featureless_spectrum_requires_review():
    df = spectrum(-11)
    df['norm_fl_smooth_flattened'] = 1.
    with pytest.raises(ValueError, match='resolved trough'):
        find_background(df, FitConfig(anchor_strategy='adaptive'))


def test_adaptive_background_survives_second_stage(tmp_path):
    df = spectrum(-20)
    cfg = FitConfig(anchor_strategy='adaptive')
    bg, figure = find_background(df, cfg)
    record = SpectrumRecord('id', 'name', 'spectrum', tmp_path, 0., 0., 0)
    selection = pipeline.BackgroundSelection(bg, cfg.vel_width, figure)
    row = pipeline._background_row(record, tmp_path, df, selection, mode='auto', clip_outliers=False)
    restored, _ = pipeline._background_from_record(replace(record, metadata=row))
    assert restored['sil6355_velocity_initial'] == bg['sil6355_velocity_initial']
    assert restored['adaptive_candidates'] == bg['adaptive_candidates']
    assert restored['anchor_strategy'] == 'adaptive'
    core.plt.close(figure)


def test_adaptive_mc_uses_selected_velocity():
    df = spectrum(-20)
    bg, figure = find_background(df, FitConfig(anchor_strategy='adaptive'))
    summary, mc_figure = core.fit_features_mc(
        df, bg, n_iter=3, mode='shift', continuum_shift=0, seed=42,
    )
    try:
        assert summary['n_success_shift'] == 3
        assert summary['sil6355_vel_med_shift'] == pytest.approx(-20, abs=.3)
    finally:
        core.plt.close(figure)
        core.plt.close(mc_figure)


def test_conflicting_candidate_does_not_override_supported_legacy_fit(monkeypatch):
    import siafit.adaptive as adaptive

    df = spectrum(-11)
    real_fit = core.fit_features

    def supported_fit(*args, **kwargs):
        metrics, figure, *rest = real_fit(*args, **kwargs)
        metrics['sil6355_snr'] = 10.
        return metrics, figure, *rest

    monkeypatch.setattr(core, 'fit_features', supported_fit)
    monkeypatch.setattr(adaptive, 'trough_candidates', lambda *args, **kwargs: [
        dict(velocity=-23., blue=-28., red=-18., width=8., prominence=.1, score=.3),
    ])
    bg, figure = adaptive.find_background(df, FitConfig(anchor_strategy='adaptive'))
    try:
        assert bg['adaptive_rescued'] == 0
        assert bg['adaptive_ambiguous'] == 1
        assert bg['sil6355_velocity_initial'] == pytest.approx(-11, abs=.3)
    finally:
        core.plt.close(figure)

"""Shoulder ("tangent") continuum anchors: identity, tilt, blends and narrow lines."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from siafit import FitConfig, ca, combined, common, core, pipeline, run_catalog, tangent
from siafit.catalog import SpectrumRecord

CONFIG = FitConfig()


def frame(wavelength, flux, error=0.005):
    return pd.DataFrame(dict(
        rest_wl=wavelength, norm_fl=flux, norm_fl_smooth=flux,
        norm_flerr=np.full(len(wavelength), error),
        norm_fl_flattened=flux, norm_fl_smooth_flattened=flux,
        vel_sil1_1=core.wl_to_vel(wavelength, core.rest_wl_sil1_1),
        vel_sil2_1=core.wl_to_vel(wavelength, core.rest_wl_sil2_1),
        vel_ca_ref=ca.wl_to_vel(wavelength, ca.rest_wl_ca_ref),
    ))


def si_spectrum(velocity, width=7., secondary=-.08, tilt=0., step=2.):
    w = np.arange(4700., 7100., step)
    absorption = (core.g_sil1_1(w, -.25, velocity, width) + core.g_sil1_2(w, -.25, velocity, width)
                  + core.g_sil2_1(w, secondary, velocity, width * .85)
                  + core.g_sil2_2(w, secondary, velocity, width * .85))
    # Broad emission bumps give the features shoulders, as in real spectra.
    bumps = sum(.06 * np.exp(-.5 * ((w - centre) / 60.) ** 2) for centre in (5650., 6420.))
    return frame(w, (1 + tilt * (w - 6000.) / 1000.) * (1 + bumps) * (1 + absorption))


def wavelength_of(background, key, rest):
    return float(core.vel_to_wl(background[key], rest))


@pytest.mark.parametrize('velocity', [-5., -9., -11., -14., -18., -22., -27., -35.])
@pytest.mark.parametrize('width', [5., 9.])
def test_velocity_is_recovered_over_the_whole_range(velocity, width):
    df = si_spectrum(velocity, width)
    background, figure = tangent.find_si_background(df, CONFIG)
    metrics, fit_figure, *_ = core.fit_features(df, background)
    core.plt.close(figure)
    core.plt.close(fit_figure)
    assert background['sil6355_blue_side'] < velocity < background['sil6355_red_side']
    assert metrics['sil6355_vel'] == pytest.approx(velocity, abs=.4)


@pytest.mark.parametrize('tilt', [-.9, -.4, .4, .9])
def test_anchors_do_not_depend_on_the_slope_of_the_spectrum(tilt):
    results = []
    for slope in (0., tilt):
        df = si_spectrum(-12., tilt=slope)
        background, _ = tangent.find_si_background(df, CONFIG)
        metrics, figure, *_ = core.fit_features(df, background)
        core.plt.close(figure)
        results.append((background, metrics))
    (flat, expected), (tilted, metrics) = results
    # A rounded shoulder is touched at a slightly different point by a tilted
    # line; the anchors stay on the same shoulders and the measurement holds.
    for key in ('sil6355_blue_side', 'sil6355_red_side', 'sil5972_blue_side'):
        assert tilted[key] == pytest.approx(flat[key], abs=3.)
    assert metrics['sil6355_vel'] == pytest.approx(expected['sil6355_vel'], abs=.1)
    assert metrics['sil6355_ew'] == pytest.approx(expected['sil6355_ew'], rel=.05)


def test_fast_line_keeps_its_own_shoulder_instead_of_bridging_5972():
    df = si_spectrum(-20., 9.)
    background, _ = tangent.find_si_background(df, CONFIG)
    blue = wavelength_of(background, 'sil6355_blue_side', core.rest_wl_sil1_1)
    secondary_minimum = float(core.vel_to_wl(-20., tangent.SI5972_BLEND))
    primary_minimum = float(core.vel_to_wl(-20., tangent.SI6355_BLEND))
    assert secondary_minimum < blue < primary_minimum
    assert background['sil5972_red_side'] == pytest.approx(
        float(core.wl_to_vel(blue, core.rest_wl_sil2_1)))
    assert wavelength_of(background, 'sil5972_blue_side', core.rest_wl_sil2_1) < secondary_minimum


def test_slow_si6355_outside_inner_search_is_not_mistaken_for_si5972():
    w = np.arange(4700., 7100., 2.)
    velocity = -5.
    absorption = (
        core.g_sil1_1(w, -.2, velocity, 6.) + core.g_sil1_2(w, -.2, velocity, 6.)
        + core.g_sil2_1(w, -.08, velocity, 5.1) + core.g_sil2_2(w, -.08, velocity, 5.1)
    )
    df = frame(w, 1 + absorption)
    background, _ = tangent.find_si_background(df, CONFIG)
    metrics, figure, *_ = core.fit_features(df, background)
    core.plt.close(figure)
    assert background['sil6355_blue_side'] < velocity < background['sil6355_red_side']
    assert metrics['sil6355_vel'] == pytest.approx(velocity, abs=.4)


def test_absent_5972_gives_one_continuum_and_a_flag():
    df = si_spectrum(-12., secondary=0.)
    background, _ = tangent.find_si_background(df, CONFIG)
    metrics, figure, *_ = core.fit_features(df, background)
    core.plt.close(figure)
    assert metrics['sil6355_vel'] == pytest.approx(-12., abs=.4)
    assert metrics['sil5972_ew'] < 3.


def test_narrow_line_on_the_shoulder_moves_neither_anchor_nor_continuum():
    clean = si_spectrum(-11., step=1.)
    dirty = clean.copy()
    w = dirty.rest_wl.to_numpy()
    notch = 1 - .5 * np.exp(-.5 * ((w - 5893.) / 3.) ** 2)   # host Na I D
    for column in ('norm_fl', 'norm_fl_smooth'):
        dirty[column] = dirty[column] * notch
    reference, _ = tangent.find_si_background(clean, CONFIG)
    background, _ = tangent.find_si_background(dirty, CONFIG)
    assert background['sil6355_blue_side'] == pytest.approx(reference['sil6355_blue_side'], abs=1.)
    expected, figure, *_ = core.fit_features(clean, reference)
    core.plt.close(figure)
    metrics, figure, *_ = core.fit_features(dirty, background)
    core.plt.close(figure)
    assert metrics['sil6355_ew'] == pytest.approx(expected['sil6355_ew'], rel=.04)
    plain = dict(background, filter_continuum=False)
    biased, figure, *_ = core.fit_features(dirty, plain)
    core.plt.close(figure)
    assert abs(metrics['sil6355_ew'] - expected['sil6355_ew']) <= abs(
        biased['sil6355_ew'] - expected['sil6355_ew'])


def test_continuum_mask_drops_the_trough_wall_but_never_a_whole_side():
    df = si_spectrum(-11.)
    background, _ = tangent.find_si_background(df, CONFIG)
    velocity = df.vel_sil1_1.to_numpy()
    blue, red = background['sil6355_blue_side'], background['sil6355_red_side']
    mask = tangent.continuum_mask(df, velocity, blue, red, 8.)
    window = (np.abs(velocity - blue) <= 4.) | (np.abs(velocity - red) <= 4.)
    assert mask.sum() < window.sum() and not np.any(mask & ~window)
    for anchor in (blue, red):
        assert np.count_nonzero(mask & (np.abs(velocity - anchor) <= 4.)) >= 2
    # Deep in the wall nothing survives.
    line = np.interp(velocity, [blue, red], np.interp([blue, red], velocity, df.norm_fl))
    assert not np.any(mask & (df.norm_fl.to_numpy() < .9 * line))


def ca_spectrum(si_velocity=-11., hvf_velocity=-22., hvf=-.25, tilt=0.):
    w = np.arange(7000., 9300., 3.)
    absorption = ca.ca_model_total(w, -.3, si_velocity, 7., hvf, hvf_velocity, 7.)
    oxygen = -.25 * np.exp(-.5 * ((w - 7500.) / 60.) ** 2)   # O I 7774 blueward
    bumps = sum(.06 * np.exp(-.5 * ((w - centre) / 70.) ** 2) for centre in (7780., 8760.))
    return frame(w, (1 + tilt * (w - 8200.) / 1000.) * (1 + bumps) * (1 + absorption + oxygen))


@pytest.mark.parametrize('hvf_velocity', [-18., -22., -27.])
@pytest.mark.parametrize('tilt', [-.5, 0., .5])
def test_ca_anchors_bracket_both_components_and_stay_out_of_oxygen(hvf_velocity, tilt):
    df = ca_spectrum(hvf_velocity=hvf_velocity, tilt=tilt)
    background, figure = tangent.find_ca_background(df, CONFIG, -11.)
    core.plt.close(figure)
    blue = wavelength_of(background, 'ca_blue_side', ca.rest_wl_ca_ref)
    red = wavelength_of(background, 'ca_red_side', ca.rest_wl_ca_ref)
    assert 7560. < blue < float(ca.vel_to_wl(hvf_velocity, ca.rest_wl_ca1))
    assert float(ca.vel_to_wl(-11., ca.rest_wl_ca3)) < red < 8900.
    if hvf_velocity < -20.:   # closer components are degenerate in the fit itself
        metrics, fit_figure, *_ = ca.fit_features(df, background, -11.)
        core.plt.close(fit_figure)
        assert metrics['ca_pvf_vel'] == pytest.approx(-11., abs=1.)
        assert metrics['ca_hvf_vel'] == pytest.approx(hvf_velocity, abs=1.)


def test_ca_bump_between_components_is_not_an_anchor():
    # Well separated components leave a bump that rises close to the continuum.
    df = ca_spectrum(si_velocity=-10., hvf_velocity=-27., hvf=-.2)
    background, _ = tangent.find_ca_background(df, CONFIG, -10.)
    blue = wavelength_of(background, 'ca_blue_side', ca.rest_wl_ca_ref)
    red = wavelength_of(background, 'ca_red_side', ca.rest_wl_ca_ref)
    assert blue < float(ca.vel_to_wl(-27., ca.rest_wl_ca1))
    assert red > float(ca.vel_to_wl(-10., ca.rest_wl_ca3))


@pytest.mark.parametrize(('velocity', 'amplitude'), [(-27., -.07), (-32., -.18)])
def test_ca_search_recovers_detached_hvf_outside_prior_core(velocity, amplitude, tmp_path):
    w = np.arange(7000., 9300., 3.)
    df = frame(w, 1 + ca.ca_model_total(w, -.3, -11., 4., amplitude, velocity, 4.))
    background, _ = tangent.find_ca_background(df, CONFIG, -11.)
    assert background['status_ca_blue_side'] == 'possible_hvf'
    blue = wavelength_of(background, 'ca_blue_side', ca.rest_wl_ca_ref)
    assert blue < ca.vel_to_wl(velocity, ca.rest_wl_ca1)

    record = SpectrumRecord(
        record_id='a', name='a', spectrum='a.dat', spectrum_path=tmp_path / 'a.dat', redshift=0.,
        mwebv=0., phase=None, mode='auto', input_format='auto', vexp=None,
        catalog_index=0, metadata={},
    )
    selection = pipeline.BackgroundSelection(background, CONFIG.vel_width, None)
    row = combined._background_row(record, tmp_path, df, CONFIG, False, None, selection, -11., 'test')
    assert row['ca_background_qc_flag'] == 1
    restored = combined._saved_ca_selection(replace(record, metadata=row), df, -11., CONFIG)
    assert restored.background == background
    metrics, figure, *_ = ca.fit_features(df, restored.background, -11.)
    core.plt.close(figure)
    core.plt.close(restored.background_figure)
    assert metrics['ca_hvf_vel'] == pytest.approx(velocity, abs=.5)


def test_flattening_keeps_positive_spectrum_positive(tmp_path, monkeypatch):
    w = np.linspace(4000., 7500., 351)
    knots = np.linspace(w[0], w[-1], 8)
    y = np.interp(w, knots, [1., .01, .01, 1., 1., 1., 1., 1.])
    path = tmp_path / 'continuum.dat'
    np.savetxt(path, np.column_stack([w, y, np.full_like(w, .001)]))
    monkeypatch.setattr(common, 'smooth_spec', lambda wavelength, flux, **kwargs: flux)
    spectrum = common.prepare_spectrum(path, 0., 0., vexp=.005, try_clip=False)
    assert (spectrum['norm_fl_flattened'] > 0).all()
    assert (spectrum['norm_flerr_flattened'] > 0).all()


def test_tangent_strategy_saved_background_round_trip():
    df = si_spectrum(-16., 8.)
    selection = pipeline.select_background(df, mode='auto', config=CONFIG)
    row = {f'background_{key}': value for key, value in selection.background.items()}
    record = SpectrumRecord(
        record_id='a', name='a', spectrum='a.dat', spectrum_path='a.dat', redshift=0.,
        mwebv=0., phase=None, mode='auto', input_format='auto', vexp=None,
        catalog_index=0, metadata=row,
    )
    restored, width = pipeline._background_from_record(record)
    assert restored['filter_continuum']
    first, figure, *_ = core.fit_features(df, selection.background)
    core.plt.close(figure)
    second, figure, *_ = core.fit_features(df, restored, vel_width=width)
    core.plt.close(figure)
    assert second['sil6355_vel'] == pytest.approx(first['sil6355_vel'])
    assert second['sil6355_ew'] == pytest.approx(first['sil6355_ew'])


def test_monte_carlo_runs_on_tangent_anchors():
    df = si_spectrum(-13.)
    selection = pipeline.select_background(df, mode='auto', config=CONFIG)
    _, _, summary, _ = pipeline._fit_spectrum(
        df, selection.background, CONFIG.vel_width, replace(CONFIG, n_iterations=4, mc_mode='all'), seed=3)
    core.plt.close('all')
    for mode in ('resample', 'shift', 'shift_resample'):
        assert summary[f'n_success_{mode}'] == 4


def test_tangent_settings_are_validated():
    with pytest.raises(ValueError):
        FitConfig(tangent_prominence=0.).validate()
    with pytest.raises(ValueError):
        FitConfig(tangent_weak_prominence=.2).validate()
    with pytest.raises(ValueError):
        FitConfig(tangent_si_trough_range=(-5., -30.)).validate()
    with pytest.raises(ValueError, match='must contain'):
        FitConfig(tangent_si_trough_limits=(-50., -40.)).validate()


@pytest.mark.parametrize("velocity", [-1., -35.])
def test_extreme_velocity_through_reader_clipping_and_default_catalogue(tmp_path, velocity):
    df = si_spectrum(velocity, 6.)
    rng = np.random.default_rng(9)
    path = tmp_path / "injection.dat"
    np.savetxt(path, np.column_stack((
        df.rest_wl.to_numpy() * 1.03,
        df.norm_fl.to_numpy() + rng.normal(0, .005, len(df)),
        df.norm_flerr.to_numpy(),
    )))
    catalog = tmp_path / "catalog.csv"
    pd.DataFrame([dict(spectrum=str(path), redshift=.03, mwebv=0)]).to_csv(catalog, index=False)
    result = run_catalog(catalog, tmp_path / "results", config=FitConfig(n_iterations=2, seed=42))
    assert result.succeeded == 1 and result.failed == 0
    row = pd.read_csv(tmp_path / "results" / "fit_results.csv").iloc[0]
    assert row.best_fit_sil6355_vel == pytest.approx(velocity, abs=.5)
    assert row.mc_success_fraction == 1


def test_shoulder_search_does_not_extrapolate_beyond_spectrum():
    wavelength = np.arange(6000., 6401., 2.)
    flux = np.ones_like(wavelength)
    grid, smoothed = tangent.resample(frame(wavelength, flux), 5900., 6500.)
    assert wavelength[0] == grid[0] <= grid[-1] <= wavelength[-1]
    assert np.all(smoothed == 1.)

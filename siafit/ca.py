"""Ca II NIR triplet model, continuum selection, diagnostics, and fitting."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lmfit import Model, Parameters, minimize
from scipy.integrate import trapezoid
from scipy.interpolate import make_splrep

from . import common
from .common import (
    _choose_unique_anchor,
    _mc_plot_bin_edges,
    autoscale_ylim_for_xlim,
    lin_back,
    vel_to_wl,
    wl_to_vel,
)
from .coverage import RequiredInterval

speed_of_light = 2.99792e5  # km/s
rest_wl_ca1 = 8498.02
rest_wl_ca2 = 8542.09
rest_wl_ca3 = 8662.14
rest_wl_ca_ref = rest_wl_ca1


def required_rest_intervals(config):
    """Ca coverage from both bounded anchor-search regions and fit span."""
    blue_search_min = max(
        config.ca_blue_bounds_kms[0],
        config.ca_blue_initial * 1000.0 - config.ca_background_search_window_kms,
    )
    red_search_max = min(
        config.ca_red_bounds_kms[1],
        config.ca_red_initial * 1000.0 + config.ca_background_search_window_kms,
    )
    half_width_kms = config.vel_width * 500.0
    lower = vel_to_wl((blue_search_min - half_width_kms) / 1000.0, rest_wl_ca_ref)
    upper = vel_to_wl((red_search_max + half_width_kms) / 1000.0, rest_wl_ca_ref)
    return [RequiredInterval("Ca II NIR", float(lower), float(upper))]


def prepare_spectrum(file_spec, redshift, mwebv, vexp=None, type_of_spec="auto", try_clip=True):
    """Prepare shared spectrum columns and the Ca II velocity coordinate."""
    df_spec = common.prepare_spectrum(
        file_spec, redshift, mwebv, vexp=vexp,
        type_of_spec=type_of_spec, try_clip=try_clip,
    )
    df_spec.loc[:, "vel_ca_ref"] = wl_to_vel(df_spec["rest_wl"], rest_wl_ca_ref)
    return df_spec


def _gauss_ca(x, amplitude, v_center, fwhm, rest_wl):
    """Generic Gaussian for one Ca II NIR component (v in 10^3 km/s)."""
    zmeas    = (x - rest_wl) / rest_wl
    vel      = (((zmeas + 1.0) ** 2 - 1.0) * speed_of_light / (1.0 + (1.0 + zmeas) ** 2)) / 1e3
    sigma    = np.abs(fwhm) / 2.35482
    return amplitude * np.exp(-0.5 * ((vel - v_center) / sigma) ** 2)



def g_ca_pvf1(x, a_pvf, v_pvf, fwhm_pvf):
    return _gauss_ca(x, a_pvf, v_pvf, fwhm_pvf, rest_wl_ca1)



def g_ca_pvf2(x, a_pvf, v_pvf, fwhm_pvf):
    return _gauss_ca(x, a_pvf, v_pvf, fwhm_pvf, rest_wl_ca2)



def g_ca_pvf3(x, a_pvf, v_pvf, fwhm_pvf):
    return _gauss_ca(x, a_pvf, v_pvf, fwhm_pvf, rest_wl_ca3)



def g_ca_hvf1(x, a_hvf, v_hvf, fwhm_hvf):
    return _gauss_ca(x, a_hvf, v_hvf, fwhm_hvf, rest_wl_ca1)



def g_ca_hvf2(x, a_hvf, v_hvf, fwhm_hvf):
    return _gauss_ca(x, a_hvf, v_hvf, fwhm_hvf, rest_wl_ca2)



def g_ca_hvf3(x, a_hvf, v_hvf, fwhm_hvf):
    return _gauss_ca(x, a_hvf, v_hvf, fwhm_hvf, rest_wl_ca3)



def ca_model_total(x, a_pvf, v_pvf, fwhm_pvf, a_hvf, v_hvf, fwhm_hvf):
    pvf = g_ca_pvf1(x, a_pvf, v_pvf, fwhm_pvf) + \
          g_ca_pvf2(x, a_pvf, v_pvf, fwhm_pvf) + \
          g_ca_pvf3(x, a_pvf, v_pvf, fwhm_pvf)
    hvf = g_ca_hvf1(x, a_hvf, v_hvf, fwhm_hvf) + \
          g_ca_hvf2(x, a_hvf, v_hvf, fwhm_hvf) + \
          g_ca_hvf3(x, a_hvf, v_hvf, fwhm_hvf)
    return pvf + hvf



def find_background_regions(
    df_spec,
    ca_red_side_init,
    ca_blue_side_init,
    v_window_kms=10000.0,
    dv_kms_red=+1000.0,
    dv_kms_blue=-1000.0,
    min_sep=5,
    max_attempts=5,
    ca_red_bounds_kms=(-5000.0, 20000.0),
    ca_blue_bounds_kms=(-50000.0, -20000.0),
):
    """
    Locate Ca II NIR continuum anchor points using spline derivatives.

    Parameters
    ----------
    df_spec : DataFrame
        From prepare_spectrum. Must contain rest_wl, norm_fl_smooth_flattened.
    ca_red_side_init, ca_blue_side_init : float
        Initial guesses in 10^3 km/s (relative to rest_wl_ca_ref).
    v_window_kms : float
        Half-width of the spline-maximum search window (km/s).
    dv_kms_red, dv_kms_blue : float
        Step for bi-directional retries (km/s).
    min_sep : int
        Minimum pixel separation between anchors.
    max_attempts : int
        Bi-directional retry attempts per anchor.
    ca_red_bounds_kms, ca_blue_bounds_kms : (float, float)
        Velocity bounds (km/s) for each anchor.

    Returns
    -------
    results : dict
        Keys: ca_red_side, ca_blue_side, status_ca_red_side, status_ca_blue_side
        All velocities in 10^3 km/s.
    fig : Figure
        Diagnostic plot.
    """
    y_spl = make_splrep(df_spec['rest_wl'], df_spec['norm_fl_smooth_flattened'], s=0, k=3)
    y_spl_1d = y_spl.derivative(nu=1)
    y_spl_2d = y_spl.derivative(nu=2)

    wl_grid = df_spec['rest_wl'].values
    exclude_idx = []
    def wl_to_idx(w): return int(np.argmin(np.abs(wl_grid - w)))

    r_wl, st_r = _choose_unique_anchor(
        y_spl_1d, y_spl_2d, ca_red_side_init, rest_wl_ca_ref,
        v_window_kms, dv_kms_red, wl_grid, exclude_idx, min_sep, max_attempts, ca_red_bounds_kms
    )
    exclude_idx.append(wl_to_idx(r_wl))

    b_wl, st_b = _choose_unique_anchor(
        y_spl_1d, y_spl_2d, ca_blue_side_init, rest_wl_ca_ref,
        v_window_kms, dv_kms_blue, wl_grid, exclude_idx, min_sep, max_attempts, ca_blue_bounds_kms
    )
    exclude_idx.append(wl_to_idx(b_wl))

    results = {
        'ca_red_side':       wl_to_vel(r_wl, rest_wl_ca_ref),
        'ca_blue_side':      wl_to_vel(b_wl, rest_wl_ca_ref),
        'status_ca_red_side':  st_r,
        'status_ca_blue_side': st_b,
    }

    # Initial-guess wavelengths for plotting
    r_init_wl = vel_to_wl(ca_red_side_init,  rest_wl_ca_ref)
    b_init_wl = vel_to_wl(ca_blue_side_init, rest_wl_ca_ref)

    wl_plot = df_spec['rest_wl']

    fig, axs = plt.subplots(ncols=1, nrows=3, sharex='col', figsize=(6, 8))
    ax_ca, ax_d1, ax_d2 = axs.flatten()

    y_raw    = df_spec['norm_fl_flattened']
    y_smooth = df_spec['norm_fl_smooth_flattened']

    ax_ca.plot(wl_plot, y_raw,    color='tab:grey', label='raw')
    ax_ca.plot(wl_plot, y_smooth, color='k',        label='smoothed')
    ax_ca.plot([r_init_wl, b_init_wl],
               y_spl([r_init_wl, b_init_wl]),
               'o', color='tab:green', label='guesses')
    ax_ca.plot([r_wl, b_wl],
               y_spl([r_wl, b_wl]),
               'o', mfc='none', mew=2.5, color='tab:red', label='solutions')
    autoscale_ylim_for_xlim((7800, 9200), margin=0.1, ax=ax_ca)
    ax_ca.legend()

    ax_d1.plot(wl_plot, y_spl_1d(wl_plot), color='tab:blue', label='1st deriv')
    ax_d1.axhline(0, ls='--', color='tab:red')
    ax_d1.plot([r_init_wl, b_init_wl],
               y_spl_1d([r_init_wl, b_init_wl]),
               'o', markersize=5, color='tab:green', label='guesses')
    ax_d1.plot([r_wl, b_wl],
               y_spl_1d([r_wl, b_wl]),
               'o', mfc='none', mew=2.5, color='tab:red', label='solutions')
    autoscale_ylim_for_xlim((7800, 9200), margin=0.1, ax=ax_d1)
    ax_d1.legend()

    ax_d2.plot(wl_plot, y_spl_2d(wl_plot), color='tab:orange', label='2nd deriv')
    ax_d2.axhline(0, ls='--', color='tab:red')
    ax_d2.plot([r_init_wl, b_init_wl],
               y_spl_2d([r_init_wl, b_init_wl]),
               'o', markersize=5, color='tab:green', label='guesses')
    ax_d2.plot([r_wl, b_wl],
               y_spl_2d([r_wl, b_wl]),
               'o', mfc='none', mew=2.5, color='tab:red', label='solutions')
    autoscale_ylim_for_xlim((7800, 9200), margin=0.1, ax=ax_d2)
    ax_d2.legend()
    ax_d2.set_xlabel("Rest wavelength (Å)")

    fig.tight_layout()
    plt.close(fig)
    return results, fig



def fit_features(
    df_spec,
    background_dict,
    si_vel_1e3,
    vel_width: float = 4.0,
    n_resol_neighbors: int = 4,
    hvf_min_sep_kms: float = 2.0,
    pvf_vel_frac: float = 0.25,
):
    """
    Fit the Ca II NIR triplet with 3 PVF + 3 HVF Gaussians.

    Constraints (Maguire et al.)
    ----------------------------
    PVF: same velocity & FWHM for all three lines.
         |v_pvf - v_si| ≤ pvf_vel_frac × |v_si|  (default 25 %)
    HVF: same velocity, FWHM, and amplitude for all three lines (optically thick).
         v_hvf ≤ v_si - hvf_min_sep_kms  (default 2 × 10^3 km/s bluer)

    Parameters
    ----------
    df_spec : DataFrame
        From prepare_spectrum.
    background_dict : dict
        Keys: ca_red_side, ca_blue_side  (velocities in 10^3 km/s)
    si_vel_1e3 : float
        Si II 6355 velocity in 10^3 km/s (used to set PVF/HVF priors).
    vel_width : float
        Half-width of background windows (10^3 km/s).
    n_resol_neighbors : int
        Points around 8542 Å used to estimate instrumental resolution.
    hvf_min_sep_kms : float
        Minimum velocity separation (10^3 km/s) between HVF and Si velocity.
    pvf_vel_frac : float
        Fractional constraint on PVF velocity relative to Si velocity.

    Returns
    -------
    metrics : dict
    fig : Figure
    back_ca : lmfit ModelResult (background)
    fit_ca : lmfit MinimizerResult (Gaussian fit)
    """
    # ── background points ────────────────────────────────────────────────────
    v = df_spec['vel_ca_ref'].values
    vr = background_dict['ca_red_side']
    vb = background_dict['ca_blue_side']
    hw = vel_width / 2.0

    df_back = df_spec.loc[
        ((v >= vr - hw) & (v <= vr + hw)) |
        ((v >= vb - hw) & (v <= vb + hw))
    ]
    df_fit = df_spec.loc[(v >= vb) & (v <= vr)]

    wl_back  = df_back['rest_wl'].values
    y_back   = df_back['norm_fl'].values
    ye_back  = df_back['norm_flerr'].values
    wl_fit   = df_fit['rest_wl'].values
    vel_fit  = df_fit['vel_ca_ref'].values

    # ── instrumental resolution ───────────────────────────────────────────────
    resol_idx    = np.argsort(np.abs(wl_fit - rest_wl_ca2))[:n_resol_neighbors]
    closest      = wl_fit[resol_idx]
    resol_vel_ca = np.abs(wl_to_vel(closest.max(), rest_wl_ca2) - wl_to_vel(closest.min(), rest_wl_ca2))

    # ── linear background fit ─────────────────────────────────────────────────
    model_back  = Model(lin_back)
    pars_back   = Parameters()
    pars_back.add('a', value=0.0)
    pars_back.add('b', value=0.0)
    out_back    = model_back.fit(y_back, pars_back, x=wl_back, weights=1.0 / ye_back)
    back_vals   = lin_back(wl_fit, out_back.params['a'].value, out_back.params['b'].value)

    y_fit  = (df_fit['norm_fl'].values / back_vals) - 1.0
    ye_fit = df_fit['norm_flerr'].values / back_vals

    # ── PVF / HVF velocity bounds ─────────────────────────────────────────────
    v_si   = float(si_vel_1e3)          # e.g. -11.0 (10^3 km/s, negative)
    margin = pvf_vel_frac * abs(v_si)

    pvf_v_min = v_si - margin           # looser lower bound (more negative = faster)
    pvf_v_max = v_si + margin

    hvf_v_max = v_si - float(hvf_min_sep_kms)   # must be at least 2000 km/s faster

    # ── lmfit parameters ─────────────────────────────────────────────────────
    pars = Parameters()
    # PVF
    pars.add('a_pvf',   value=-0.4,  max=0.0)
    pars.add('v_pvf',   value=v_si,  min=pvf_v_min, max=pvf_v_max)
    pars.add('fwhm_pvf', value=8.0,  min=resol_vel_ca)
    # HVF (amplitude tied to a single parameter → optically thick)
    pars.add('a_hvf',   value=-0.2,  max=0.0)
    pars.add('v_hvf',   value=v_si - 4.0, max=hvf_v_max)
    pars.add('fwhm_hvf', value=8.0,  min=resol_vel_ca)

    def model_ca(params, wl):
        pvf = (g_ca_pvf1(wl, params['a_pvf'], params['v_pvf'], params['fwhm_pvf']) +
               g_ca_pvf2(wl, params['a_pvf'], params['v_pvf'], params['fwhm_pvf']) +
               g_ca_pvf3(wl, params['a_pvf'], params['v_pvf'], params['fwhm_pvf']))
        hvf = (g_ca_hvf1(wl, params['a_hvf'], params['v_hvf'], params['fwhm_hvf']) +
               g_ca_hvf2(wl, params['a_hvf'], params['v_hvf'], params['fwhm_hvf']) +
               g_ca_hvf3(wl, params['a_hvf'], params['v_hvf'], params['fwhm_hvf']))
        return pvf + hvf

    def resid(params):
        return (y_fit - model_ca(params, wl_fit)) / ye_fit

    out_fit = minimize(resid, pars, method='least_squares')

    # ── pEW ───────────────────────────────────────────────────────────────────
    wl_dense = np.linspace(wl_fit[0], wl_fit[-1], 500)
    pvf_dense = (g_ca_pvf1(wl_dense, out_fit.params['a_pvf'].value, out_fit.params['v_pvf'].value, out_fit.params['fwhm_pvf'].value) +
                 g_ca_pvf2(wl_dense, out_fit.params['a_pvf'].value, out_fit.params['v_pvf'].value, out_fit.params['fwhm_pvf'].value) +
                 g_ca_pvf3(wl_dense, out_fit.params['a_pvf'].value, out_fit.params['v_pvf'].value, out_fit.params['fwhm_pvf'].value))
    hvf_dense = (g_ca_hvf1(wl_dense, out_fit.params['a_hvf'].value, out_fit.params['v_hvf'].value, out_fit.params['fwhm_hvf'].value) +
                 g_ca_hvf2(wl_dense, out_fit.params['a_hvf'].value, out_fit.params['v_hvf'].value, out_fit.params['fwhm_hvf'].value) +
                 g_ca_hvf3(wl_dense, out_fit.params['a_hvf'].value, out_fit.params['v_hvf'].value, out_fit.params['fwhm_hvf'].value))

    ew_pvf     = np.abs(trapezoid(pvf_dense, x=wl_dense))
    ew_hvf     = np.abs(trapezoid(hvf_dense, x=wl_dense))
    ew_total   = np.abs(trapezoid(pvf_dense + hvf_dense, x=wl_dense))
    ew_raw     = np.abs(trapezoid(y_fit,  x=wl_fit))

    # ── SNR ───────────────────────────────────────────────────────────────────
    try:
        model_vals = model_ca(out_fit.params, wl_fit)
        residuals  = y_fit - model_vals
        rms        = np.sqrt(np.nanmean(residuals ** 2))
        SNR_ca     = (np.nanmax(model_vals) - np.nanmin(model_vals)) / rms if rms > 0 else np.inf
    except Exception:
        SNR_ca = np.nan

    metrics = dict(
        ca_pvf_vel=out_fit.params['v_pvf'].value,
        ca_pvf_vel_err=out_fit.params['v_pvf'].stderr,
        ca_pvf_fwhm=out_fit.params['fwhm_pvf'].value,
        ca_pvf_fwhm_err=out_fit.params['fwhm_pvf'].stderr,
        ca_pvf_amp=out_fit.params['a_pvf'].value,
        ca_hvf_vel=out_fit.params['v_hvf'].value,
        ca_hvf_vel_err=out_fit.params['v_hvf'].stderr,
        ca_hvf_fwhm=out_fit.params['fwhm_hvf'].value,
        ca_hvf_fwhm_err=out_fit.params['fwhm_hvf'].stderr,
        ca_hvf_amp=out_fit.params['a_hvf'].value,
        ca_pvf_ew=ew_pvf,
        ca_hvf_ew=ew_hvf,
        ca_total_ew=ew_total,
        ca_ew_raw=ew_raw,
        ca_R_ca=(ew_hvf / ew_pvf) if ew_pvf > 0 else np.nan,
        ca_snr=SNR_ca,
        fit_chisqr=out_fit.chisqr,
        fit_redchi=out_fit.redchi,
    )
    for component in ('pvf', 'hvf'):
        for label, rest_wl in (
            ('8498', rest_wl_ca1), ('8542', rest_wl_ca2), ('8662', rest_wl_ca3)
        ):
            metrics[f'ca_{component}_{label}_fwhm_angstrom'] = common.velocity_fwhm_to_angstrom(
                metrics[f'ca_{component}_vel'], metrics[f'ca_{component}_fwhm'], rest_wl
            )

    # ── summary figure ────────────────────────────────────────────────────────
    fig = _make_fit_figure(
        df_spec, background_dict, back_vals, out_back, out_fit,
        wl_fit, vel_fit, y_fit, pvf_dense, hvf_dense, wl_dense, vel_width,
    )

    return metrics, fig, out_back, out_fit



def _make_fit_figure(df_spec, background_dict, back_vals, out_back, out_fit,
                     wl_fit, vel_fit, y_fit, pvf_dense, hvf_dense, wl_dense,
                     vel_width):
    """Internal: build the 2-panel fit figure (wavelength + back-normalised)."""
    vr = background_dict['ca_red_side']
    vb = background_dict['ca_blue_side']
    hw = vel_width / 2.0

    v  = df_spec['vel_ca_ref'].values
    df_back = df_spec.loc[
        ((v >= vr - hw) & (v <= vr + hw)) |
        ((v >= vb - hw) & (v <= vb + hw))
    ]

    back_full = lin_back(df_spec['rest_wl'].values, out_back.params['a'].value, out_back.params['b'].value)

    vel_dense = wl_to_vel(wl_dense, rest_wl_ca_ref)

    fig, (ax_wl, ax_norm) = plt.subplots(ncols=1, nrows=2, figsize=(8, 8))

    # Top: wavelength space
    ax_wl.plot(df_spec['rest_wl'], df_spec['norm_fl'], color='tab:grey', lw=1.5, label='raw')
    ax_wl.plot(df_spec['rest_wl'], back_full, color='tab:red', lw=1.5, ls='--', label='background')

    ax_wl.plot(wl_dense, (pvf_dense + 1) * lin_back(wl_dense, out_back.params['a'].value, out_back.params['b'].value),
               color='tab:green', lw=1.5, label='PVF')
    ax_wl.plot(wl_dense, (hvf_dense + 1) * lin_back(wl_dense, out_back.params['a'].value, out_back.params['b'].value),
               color='tab:blue',  lw=1.5, label='HVF')
    ax_wl.plot(wl_dense, (pvf_dense + hvf_dense + 1) * lin_back(wl_dense, out_back.params['a'].value, out_back.params['b'].value),
               color='tab:orange', lw=2.0, label='total')

    ax_wl.plot(df_back['rest_wl'], df_back['norm_fl'], 'o', mfc='none', mec='k', mew=1.5)

    # Shade background windows in wavelength space
    for v_center, ref in [(vr, rest_wl_ca_ref), (vb, rest_wl_ca_ref)]:
        wl_lo = vel_to_wl(v_center - hw, ref)
        wl_hi = vel_to_wl(v_center + hw, ref)
        ax_wl.axvspan(wl_lo, wl_hi, alpha=0.2, color='tab:cyan')

    # Laboratory wavelengths are not the fitted absorption centres.  Draw both
    # so the triplet geometry is unambiguous in wavelength space.
    v_pvf_fit = out_fit.params['v_pvf'].value
    v_hvf_fit = out_fit.params['v_hvf'].value
    for j, (rwl, lbl) in enumerate([(rest_wl_ca1, '8498'), (rest_wl_ca2, '8542'), (rest_wl_ca3, '8662')]):
        ax_wl.axvline(
            x=rwl, color='tab:purple', ls=':', lw=0.8, alpha=0.7,
            label='laboratory wavelengths' if j == 0 else None,
        )
        ax_wl.text(rwl + 3, 0.05, lbl, color='tab:purple', fontsize=7,
                   transform=ax_wl.get_xaxis_transform())
        ax_wl.axvline(
            x=vel_to_wl(v_pvf_fit, rwl), color='tab:green', ls='--', lw=0.9, alpha=0.75,
            label='PVF component centres' if j == 0 else None,
        )
        ax_wl.axvline(
            x=vel_to_wl(v_hvf_fit, rwl), color='tab:blue', ls='--', lw=0.9, alpha=0.75,
            label='HVF component centres' if j == 0 else None,
        )

    ax_wl.set_xlim(vel_to_wl(vb - 5, rest_wl_ca_ref), vel_to_wl(vr + 5, rest_wl_ca_ref))
    autoscale_ylim_for_xlim(ax_wl.get_xlim(), margin=0.1, ax=ax_wl)
    ax_wl.set_xlabel('Rest wavelength (Å)')
    ax_wl.set_ylabel('Norm. flux')
    ax_wl.legend(fontsize=8)
    ax_wl.set_title('Ca II NIR – wavelength space')

    # Bottom: back-normalised velocity space
    ax_norm.plot(df_spec['vel_ca_ref'], (df_spec['norm_fl'].values / back_full) - 1.0,
                 color='tab:grey', lw=1.5)
    ax_norm.plot(vel_dense, pvf_dense,              color='tab:green',  lw=1.5, label='PVF')
    ax_norm.plot(vel_dense, hvf_dense,              color='tab:blue',   lw=1.5, label='HVF')
    ax_norm.plot(vel_dense, pvf_dense + hvf_dense,  color='tab:orange', lw=2.0, label='total')
    ax_norm.axhline(y=0, color='tab:red', ls='--')

    # The velocity axis uses the 8498.02 Å line as its reference. Convert each
    # fitted line centre to that axis before marking it.
    for j, rwl in enumerate([rest_wl_ca1, rest_wl_ca2, rest_wl_ca3]):
        pvf_center_on_ref = wl_to_vel(vel_to_wl(v_pvf_fit, rwl), rest_wl_ca_ref)
        hvf_center_on_ref = wl_to_vel(vel_to_wl(v_hvf_fit, rwl), rest_wl_ca_ref)
        ax_norm.axvline(
            pvf_center_on_ref, color='tab:green', ls='--', lw=0.9, alpha=0.7,
            label='PVF component centres' if j == 0 else None,
        )
        ax_norm.axvline(
            hvf_center_on_ref, color='tab:blue', ls='--', lw=0.9, alpha=0.7,
            label='HVF component centres' if j == 0 else None,
        )

    ax_norm.axvspan(vb - hw, vb + hw, alpha=0.2, color='tab:cyan')
    ax_norm.axvspan(vr - hw, vr + hw, alpha=0.2, color='tab:cyan', label='back window')

    autoscale_ylim_for_xlim((vb - 2, vr + 2), margin=0.05, ax=ax_norm)
    ax_norm.set_xlabel('Velocity (10³ km/s)')
    ax_norm.set_ylabel('(flux/back) – 1')
    ax_norm.legend(fontsize=8)
    ax_norm.set_title('Ca II NIR – back-normalised velocity space')

    fig.tight_layout()
    plt.close(fig)
    return fig



def fit_features_mc(
    df_spec,
    background_dict,
    si_vel_1e3,
    n_iter: int = 1000,
    vel_width: float = 4.0,
    n_resol_neighbors: int = 4,
    hvf_min_sep_kms: float = 2.0,
    pvf_vel_frac: float = 0.25,
    mode: str = "resample",
    continuum_shift: float = 1.3,
    seed=None,
):
    """
    Monte-Carlo uncertainty propagation for the Ca II NIR fit.

    Modes: resample | shift | shift_resample | all
    (Same semantics as the Si module.)

    Returns
    -------
    summary : dict
    fig : Figure
    """
    rng = np.random.default_rng(seed)

    valid_modes = {"resample", "shift", "shift_resample", "all"}
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {sorted(valid_modes)}, got {mode!r}")

    means  = df_spec['norm_fl'].values
    sigs   = df_spec['norm_flerr'].values

    def _make_background(draw_mode):
        return common.mc_shift_background(
            rng, background_dict, ("ca_red_side", "ca_blue_side"),
            continuum_shift, draw_mode,
        )

    def _make_flux(draw_mode):
        return common.mc_draw_flux(rng, means, sigs, draw_mode)

    def _single_fit(flux_draw, bg_draw):
        df_draw = df_spec.copy()
        df_draw.loc[:, 'norm_fl_draw'] = flux_draw

        v   = df_draw['vel_ca_ref'].values
        vr  = bg_draw['ca_red_side']
        vb  = bg_draw['ca_blue_side']
        hw  = vel_width / 2.0

        df_back = df_draw.loc[((v >= vr - hw) & (v <= vr + hw)) | ((v >= vb - hw) & (v <= vb + hw))]
        df_fit  = df_draw.loc[(v >= vb) & (v <= vr)]

        if len(df_back) < 2:
            raise ValueError("Too few background points")
        if len(df_fit) < max(6, n_resol_neighbors):
            raise ValueError("Too few fit points")

        wl_back  = df_back['rest_wl'].values
        y_back   = df_back['norm_fl_draw'].values
        ye_back  = df_back['norm_flerr'].values
        wl_fit   = df_fit['rest_wl'].values

        resol_idx    = np.argsort(np.abs(wl_fit - rest_wl_ca2))[:n_resol_neighbors]
        closest      = wl_fit[resol_idx]
        resol_vel_ca = np.abs(wl_to_vel(closest.max(), rest_wl_ca2) - wl_to_vel(closest.min(), rest_wl_ca2))

        model_back = Model(lin_back)
        pars_back  = Parameters()
        pars_back.add('a', value=0.0)
        pars_back.add('b', value=0.0)
        out_back   = model_back.fit(y_back, pars_back, x=wl_back, weights=1.0 / ye_back)
        back_vals  = lin_back(wl_fit, out_back.params['a'].value, out_back.params['b'].value)

        y_fit  = (df_fit['norm_fl_draw'].values / back_vals) - 1.0
        ye_fit = df_fit['norm_flerr'].values / back_vals

        v_si   = float(si_vel_1e3)
        margin = pvf_vel_frac * abs(v_si)
        pvf_v_min = v_si - margin
        pvf_v_max = v_si + margin
        hvf_v_max = v_si - float(hvf_min_sep_kms)

        pars = Parameters()
        pars.add('a_pvf',    value=-0.4, max=0.0)
        pars.add('v_pvf',    value=v_si, min=pvf_v_min, max=pvf_v_max)
        pars.add('fwhm_pvf', value=8.0,  min=resol_vel_ca)
        pars.add('a_hvf',    value=-0.2, max=0.0)
        pars.add('v_hvf',    value=v_si - 4.0, max=hvf_v_max)
        pars.add('fwhm_hvf', value=8.0,  min=resol_vel_ca)

        def model_ca(params, wl):
            pvf = (g_ca_pvf1(wl, params['a_pvf'], params['v_pvf'], params['fwhm_pvf']) +
                   g_ca_pvf2(wl, params['a_pvf'], params['v_pvf'], params['fwhm_pvf']) +
                   g_ca_pvf3(wl, params['a_pvf'], params['v_pvf'], params['fwhm_pvf']))
            hvf = (g_ca_hvf1(wl, params['a_hvf'], params['v_hvf'], params['fwhm_hvf']) +
                   g_ca_hvf2(wl, params['a_hvf'], params['v_hvf'], params['fwhm_hvf']) +
                   g_ca_hvf3(wl, params['a_hvf'], params['v_hvf'], params['fwhm_hvf']))
            return pvf + hvf

        out = minimize(lambda p: (y_fit - model_ca(p, wl_fit)) / ye_fit, pars, method='least_squares')

        wl_dense  = np.linspace(wl_fit[0], wl_fit[-1], 500)
        pvf_d = (g_ca_pvf1(wl_dense, out.params['a_pvf'].value, out.params['v_pvf'].value, out.params['fwhm_pvf'].value) +
                 g_ca_pvf2(wl_dense, out.params['a_pvf'].value, out.params['v_pvf'].value, out.params['fwhm_pvf'].value) +
                 g_ca_pvf3(wl_dense, out.params['a_pvf'].value, out.params['v_pvf'].value, out.params['fwhm_pvf'].value))
        hvf_d = (g_ca_hvf1(wl_dense, out.params['a_hvf'].value, out.params['v_hvf'].value, out.params['fwhm_hvf'].value) +
                 g_ca_hvf2(wl_dense, out.params['a_hvf'].value, out.params['v_hvf'].value, out.params['fwhm_hvf'].value) +
                 g_ca_hvf3(wl_dense, out.params['a_hvf'].value, out.params['v_hvf'].value, out.params['fwhm_hvf'].value))

        ew_pvf   = np.abs(trapezoid(pvf_d, x=wl_dense))
        ew_hvf   = np.abs(trapezoid(hvf_d, x=wl_dense))
        ew_total = np.abs(trapezoid(pvf_d + hvf_d, x=wl_dense))
        ew_raw   = np.abs(trapezoid(y_fit, x=wl_fit))

        return dict(
            ca_pvf_vel=out.params['v_pvf'].value,
            ca_pvf_fwhm=out.params['fwhm_pvf'].value,
            ca_hvf_vel=out.params['v_hvf'].value,
            ca_hvf_fwhm=out.params['fwhm_hvf'].value,
            ca_pvf_ew=ew_pvf,
            ca_hvf_ew=ew_hvf,
            ca_total_ew=ew_total,
            ca_ew_raw=ew_raw,
            ca_R_ca=ew_hvf / ew_pvf if ew_pvf > 0 else np.nan,
        )

    metric_names = [
        'ca_pvf_vel', 'ca_pvf_fwhm',
        'ca_hvf_vel', 'ca_hvf_fwhm',
        'ca_pvf_ew', 'ca_hvf_ew', 'ca_total_ew', 'ca_ew_raw',
        'ca_R_ca',
    ]


    mode_suffix  = {'resample': 'resample', 'shift': 'shift', 'shift_resample': 'shift_resample'}
    requested    = ['resample', 'shift', 'shift_resample'] if mode == 'all' else [mode]
    all_results, all_summaries, failed_draws, failure_reasons = common.collect_mc_draws(
        metric_names, requested, n_iter,
        lambda draw_mode: _single_fit(_make_flux(draw_mode), _make_background(draw_mode)),
    )

    summary = {'mc_mode': mode, 'mc_n_iter': int(n_iter), 'mc_continuum_shift': float(continuum_shift)}
    for draw_mode in requested:
        sfx = mode_suffix[draw_mode]
        summary[f'n_success_{sfx}'] = int(all_summaries[draw_mode]['ca_pvf_vel']['n'])
        summary[f'n_fail_{sfx}']    = int(failed_draws[draw_mode])
        for mname, sdict in all_summaries[draw_mode].items():
            for stat in ('med', 'hi', 'lo', 'mean', 'std', 'gap', 'flag'):
                summary[f'{mname}_{stat}_{sfx}'] = sdict[stat]

    label_map = {'resample': 'resample', 'shift': 'shift', 'shift_resample': 'shift+resample'}
    plot_metrics = [
        ('ca_pvf_vel',   'Ca PVF velocity'),
        ('ca_hvf_vel',   'Ca HVF velocity'),
        ('ca_pvf_fwhm',  'Ca PVF FWHM'),
        ('ca_hvf_fwhm',  'Ca HVF FWHM'),
        ('ca_pvf_ew',    'Ca PVF EW'),
        ('ca_hvf_ew',    'Ca HVF EW'),
        ('ca_total_ew',  'Ca total EW'),
        ('ca_ew_raw',    'Ca EW raw'),
        ('ca_R_ca',      'R_Ca (HVF/PVF)'),
    ]

    fig, axs = plt.subplots(ncols=3, nrows=3, figsize=(10, 8))
    for ax, (mname, title) in zip(axs.flatten(), plot_metrics):
        arrays = [all_results[m][mname] for m in requested if all_results[m][mname].size > 0]
        if arrays:
            combined = np.concatenate(arrays)
            bins = _mc_plot_bin_edges(combined)
        else:
            bins = 20

        for draw_mode in requested:
            arr = all_results[draw_mode][mname]
            if arr.size == 0:
                continue
            med = all_summaries[draw_mode][mname]['med']
            n, b, patches = ax.hist(arr[np.isfinite(arr)], bins=bins, alpha=0.33, label=label_map[draw_mode])
            if patches:
                fc = patches[0].get_facecolor()
                ax.axvline(med, color=(fc[0], fc[1], fc[2], 1.0), ls='--', lw=2.0)

        if arrays:
            ac = np.concatenate(arrays)
            ac = ac[np.isfinite(ac)]
            if ac.size > 0:
                p1, p99 = np.nanpercentile(ac, [1, 99])
                if np.isfinite(p1) and np.isfinite(p99) and p99 > p1:
                    pad = 0.05 * (p99 - p1)
                    ax.set_xlim(p1 - pad, p99 + pad)

        ax.set_title(title)
        if mname == 'ca_pvf_vel':
            ax.legend(fontsize=7)

    fig.tight_layout()
    plt.close(fig)
    return summary, fig



def preview_full_spectrum(df_spec, spec_meta=None, si_vel_1e3=None):
    """Quick-look of the full rest-frame spectrum with Ca II NIR guides.

    ``df_spec["rest_wl"]`` is already corrected for the SN redshift in
    :func:`prepare_spectrum`, so all guide wavelengths drawn here are also in
    the rest frame.  The purple lines are the laboratory wavelengths.  When a
    Si II velocity is supplied, the orange lines show the corresponding
    blueshifted Ca II triplet positions and therefore sit near the expected
    absorption rather than at the unshifted laboratory wavelengths.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(df_spec["rest_wl"], df_spec["norm_fl"],        lw=1.2, label="raw",      color='tab:grey')
    ax.plot(df_spec["rest_wl"], df_spec["norm_fl_smooth"], lw=1.0, label="smoothed", color='k')

    # Laboratory (zero-velocity) Ca II NIR wavelengths in the same rest-frame
    # coordinates as the plotted spectrum.
    for j, (rwl, lbl) in enumerate([
        (rest_wl_ca1, '8498'),
        (rest_wl_ca2, '8542'),
        (rest_wl_ca3, '8662'),
    ]):
        ax.axvline(
            rwl, color='tab:purple', ls=':', lw=1.2, alpha=0.9,
            label='Ca II NIR rest wavelengths' if j == 0 else None,
        )
        ax.text(
            rwl, 0.015, lbl, rotation=90, ha='right', va='bottom',
            color='tab:purple', fontsize=8,
            transform=ax.get_xaxis_transform(),
        )

    # The absorption is blueshifted.  Use the per-spectrum Si II velocity as a
    # useful photospheric guide so the user can immediately see whether the
    # spectrum actually covers the expected Ca II NIR feature.
    guide_wls = [rest_wl_ca1, rest_wl_ca2, rest_wl_ca3]
    guide_name = "rest wavelengths"
    if si_vel_1e3 is not None and np.isfinite(si_vel_1e3):
        guide_wls = [
            vel_to_wl(float(si_vel_1e3), rwl)
            for rwl in [rest_wl_ca1, rest_wl_ca2, rest_wl_ca3]
        ]
        guide_name = f"at Si velocity ({float(si_vel_1e3):.1f}×10³ km/s)"
        for j, shifted_wl in enumerate(guide_wls):
            ax.axvline(
                shifted_wl, color='tab:orange', ls='--', lw=1.2, alpha=0.9,
                label=f'Expected Ca II {guide_name}' if j == 0 else None,
            )

    wl_min = float(np.nanmin(df_spec["rest_wl"]))
    wl_max = float(np.nanmax(df_spec["rest_wl"]))
    n_covered = sum(wl_min <= w <= wl_max for w in guide_wls)
    guide_text = ", ".join(f"{w:.0f}" for w in guide_wls)
    ax.text(
        0.015, 0.02,
        f"Rest-frame coverage: {wl_min:.0f}–{wl_max:.0f} Å\n"
        f"Ca II guides {guide_name}: {guide_text} Å  ({n_covered}/3 covered)",
        transform=ax.transAxes, ha='left', va='bottom', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='0.7'),
    )

    ax.set_xlabel("Rest wavelength (Å)")
    ax.set_ylabel("Norm. flux")
    ax.legend(loc="upper right", fontsize=10)

    if spec_meta:
        info_lines = []
        if "name" in spec_meta and pd.notna(spec_meta.get("name")):
            info_lines.append(f"{spec_meta['name']}")
        tel  = spec_meta.get("telescope", None)
        inst = spec_meta.get("instrument", None)
        if tel is not None and pd.notna(tel) or inst is not None and pd.notna(inst):
            info_lines.append(f"{tel} / {inst}")
        z = spec_meta.get("redshift", None)
        if z is not None and pd.notna(z):
            info_lines.append(rf"z = {z:.4f}")
        phase = spec_meta.get("phase", None)
        if phase is not None and pd.notna(phase):
            info_lines.append(rf"phase = {phase:.1f} d")
        if info_lines:
            ax.text(0.5, 0.96, "\n".join(info_lines), transform=ax.transAxes,
                    ha="center", va="top", multialignment="center", fontsize=15,
                    bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.8, edgecolor="0.7"))
    fig.tight_layout()
    return fig



def preview_background(df_spec, background_dict, vel_width: float = 4.0, si_vel_1e3=None):
    """
    Show the Ca II NIR continuum background anchors and linear fit.
    Single-panel: Ca II NIR region only.

    If ``si_vel_1e3`` is supplied, orange dashed lines are drawn at the
    Ca II NIR triplet wavelengths blueshifted to that velocity (same as in
    :func:`preview_full_spectrum`), so the expected absorption position is
    visible on the background-selection plot too.
    """
    vr = background_dict['ca_red_side']
    vb = background_dict['ca_blue_side']
    hw = vel_width / 2.0

    # wl_xlim = (vel_to_wl(vb - 8, rest_wl_ca_ref), vel_to_wl(vr + 8, rest_wl_ca_ref))
    wl_xlim = (7000.0, 9200.0)
    v    = df_spec['vel_ca_ref'].values
    wl   = df_spec['rest_wl'].values
    y_r  = df_spec['norm_fl_flattened'].values
    y_sm = df_spec['norm_fl_smooth_flattened'].values

    df_back = df_spec.loc[((v >= vr - hw) & (v <= vr + hw)) | ((v >= vb - hw) & (v <= vb + hw))]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(wl, y_r,  color='tab:grey', lw=1.0, label='raw',      zorder=1)
    ax.plot(wl, y_sm, color='k',        lw=1.0, label='smoothed', zorder=1)

    for v_center, label in [(vr, 'background window'), (vb, None)]:
        wl_lo = vel_to_wl(v_center - hw, rest_wl_ca_ref)
        wl_hi = vel_to_wl(v_center + hw, rest_wl_ca_ref)
        ax.axvspan(wl_lo, wl_hi, alpha=0.25, color='tab:cyan', label=label, zorder=0)

    ax.plot(df_back['rest_wl'], df_back['norm_fl_flattened'],
            'o', mfc='none', mec='k', mew=2.0, zorder=2)

    # Continuum fit line
    wl_b  = wl[((v >= vr - hw) & (v <= vr + hw)) | ((v >= vb - hw) & (v <= vb + hw))]
    y_b   = y_r[((v >= vr - hw) & (v <= vr + hw)) | ((v >= vb - hw) & (v <= vb + hw))]
    if len(wl_b) >= 2:
        a_poly, b_poly = np.polyfit(wl_b, y_b, 1)
    else:
        a_poly, b_poly = 0.0, 1.0
    w_r = vel_to_wl(vr, rest_wl_ca_ref)
    w_b = vel_to_wl(vb, rest_wl_ca_ref)
    w_line = np.linspace(w_b, w_r, 300)
    ax.plot(w_line, a_poly * w_line + b_poly, '--', lw=1.2, color='tab:red', label='continuum fit', zorder=2)
    ax.plot([w_r, w_b], [a_poly * w_r + b_poly, a_poly * w_b + b_poly],
            'x', color='tab:orange', markersize=10, markeredgewidth=2.0, label='anchors', zorder=2)

    # Ca II rest wavelength markers
    for rwl, lbl in [(rest_wl_ca1, '8498'), (rest_wl_ca2, '8542'), (rest_wl_ca3, '8662')]:
        ax.axvline(x=rwl, color='tab:purple', ls=':', lw=0.8)
        ax.text(rwl + 3, 0.05, lbl, color='tab:purple', fontsize=8,
                transform=ax.get_xaxis_transform())

    # Expected Ca II NIR position at the Si II velocity (orange guides),
    # same convention as preview_full_spectrum.
    if si_vel_1e3 is not None and np.isfinite(si_vel_1e3):
        for j, rwl in enumerate([rest_wl_ca1, rest_wl_ca2, rest_wl_ca3]):
            shifted_wl = vel_to_wl(float(si_vel_1e3), rwl)
            ax.axvline(
                shifted_wl, color='tab:orange', ls='--', lw=1.2, alpha=0.9,
                label=f'Expected Ca II at Si velocity ({float(si_vel_1e3):.1f}×10³ km/s)' if j == 0 else None,
            )

    ax.set_xlim(*wl_xlim)
    autoscale_ylim_for_xlim(wl_xlim, margin=0.1, ax=ax)
    ax.set_xlabel("Rest wavelength (Å)")
    ax.set_ylabel("Norm. flux (flattened)")
    ax.set_title(f"Ca II NIR background (vel_width={vel_width:.1f}×10³ km/s)")
    ax.legend(loc='best', ncol=2)
    fig.tight_layout()
    return fig



def preview_ca_fit(df_spec, background_dict, out_back, out_fit, vel_width: float = 4.0):
    """
    Interactive-quality-check preview: back-normalised velocity space.
    Mirrors preview_silicon_fit from the Si module.
    """
    vr = background_dict['ca_red_side']
    vb = background_dict['ca_blue_side']
    hw = vel_width / 2.0

    v       = df_spec['vel_ca_ref'].values
    back_full = lin_back(df_spec['rest_wl'].values, out_back.params['a'].value, out_back.params['b'].value)

    df_fit  = df_spec.loc[(v >= vb) & (v <= vr)]
    wl_fit  = df_fit['rest_wl'].values

    wl_dense  = np.linspace(wl_fit[0], wl_fit[-1], 500)
    vel_dense = wl_to_vel(wl_dense, rest_wl_ca_ref)

    pvf_d = (g_ca_pvf1(wl_dense, out_fit.params['a_pvf'].value, out_fit.params['v_pvf'].value, out_fit.params['fwhm_pvf'].value) +
             g_ca_pvf2(wl_dense, out_fit.params['a_pvf'].value, out_fit.params['v_pvf'].value, out_fit.params['fwhm_pvf'].value) +
             g_ca_pvf3(wl_dense, out_fit.params['a_pvf'].value, out_fit.params['v_pvf'].value, out_fit.params['fwhm_pvf'].value))
    hvf_d = (g_ca_hvf1(wl_dense, out_fit.params['a_hvf'].value, out_fit.params['v_hvf'].value, out_fit.params['fwhm_hvf'].value) +
             g_ca_hvf2(wl_dense, out_fit.params['a_hvf'].value, out_fit.params['v_hvf'].value, out_fit.params['fwhm_hvf'].value) +
             g_ca_hvf3(wl_dense, out_fit.params['a_hvf'].value, out_fit.params['v_hvf'].value, out_fit.params['fwhm_hvf'].value))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_spec['vel_ca_ref'], (df_spec['norm_fl'].values / back_full) - 1.0,
            color='tab:grey', lw=1.5, zorder=1)
    ax.plot(vel_dense, pvf_d,              color='tab:green',  lw=1.5, label='PVF', zorder=2)
    ax.plot(vel_dense, hvf_d,              color='tab:blue',   lw=1.5, label='HVF', zorder=2)
    ax.plot(vel_dense, pvf_d + hvf_d,      color='tab:orange', lw=2.0, label='total', zorder=2)
    ax.axhline(y=0, color='tab:red', ls='--', zorder=1)

    v_pvf_fit = out_fit.params['v_pvf'].value
    v_hvf_fit = out_fit.params['v_hvf'].value
    for j, rwl in enumerate([rest_wl_ca1, rest_wl_ca2, rest_wl_ca3]):
        pvf_center_on_ref = wl_to_vel(vel_to_wl(v_pvf_fit, rwl), rest_wl_ca_ref)
        hvf_center_on_ref = wl_to_vel(vel_to_wl(v_hvf_fit, rwl), rest_wl_ca_ref)
        ax.axvline(
            pvf_center_on_ref, color='tab:green', ls='--', lw=0.9, alpha=0.7,
            label='PVF component centres' if j == 0 else None,
        )
        ax.axvline(
            hvf_center_on_ref, color='tab:blue', ls='--', lw=0.9, alpha=0.7,
            label='HVF component centres' if j == 0 else None,
        )

    ax.axvspan(vb - hw, vb + hw, alpha=0.2, color='tab:cyan', label='back window')
    ax.axvspan(vr - hw, vr + hw, alpha=0.2, color='tab:cyan')

    autoscale_ylim_for_xlim((vb - 2, vr + 2), margin=0.05, ax=ax)
    ax.set_xlabel('Velocity (10³ km/s)')
    ax.set_ylabel('(flux/back) – 1')
    ax.legend(fontsize=9)
    ax.set_title('Ca II NIR joint fit (back-normalised)')
    fig.tight_layout()
    return fig



def interactive_refine_background_and_fit(
    df_spec,
    initial_background_dict,
    si_vel_1e3,
    spec_meta=None,
    vel_width: float = 4.0,
    wl_xlim=(7000.0, 9300.0),
    n_resol_neighbors: int = 3,
    hvf_min_sep_kms: float = 2.0,
    pvf_vel_frac: float = 0.25,
    manual_start: bool = False,
):
    """
    Interactive quality-check and background refinement for the Ca II NIR fit.

    Flow
    ----
    0. Show full spectrum; ask 'Does the spectrum look ok?'
    1. Show Ca II NIR background preview.
       Ask: 'Do you like this Ca II background? [Y/n]
             (or type a new vel_width in 10^3 km/s):'
    2. If accepted → run 6-Gaussian fit; show fit preview.
       Ask: 'Do you like the fit? [Y/n]'
    3. If fit rejected → back to step 1 to redefine anchors.
    4. If background not accepted → click 2 new continuum points.

    Parameters
    ----------
    si_vel_1e3 : float
        Si II 6355 velocity from the silicon fit (10^3 km/s),
        used to set PVF/HVF velocity priors.

    Returns
    -------
    background, bg_fig, best_fit_dict, fit_fig, vel_width, spec_qual_flag
    """
    background    = dict(initial_background_dict)
    spec_qual_flag = None

    print("\nAutomatic Ca II NIR background anchors (10^3 km/s):")
    print(f"  ca_red_side  = {background['ca_red_side']:.2f}")
    print(f"  ca_blue_side = {background['ca_blue_side']:.2f}\n")

    ca_ref_1e4 = vel_to_wl(-11, rest_wl_ca_ref)   # approximate feature centre for click guide

    was_interactive = plt.isinteractive()
    plt.ion()

    try:
        # ── STEP 0: full-spectrum QC ─────────────────────────────────────────
        fig_full = preview_full_spectrum(df_spec, spec_meta, si_vel_1e3=si_vel_1e3)
        try:
            fig_full.canvas.manager.set_window_title("Full spectrum")
        except Exception:
            pass
        fig_full.show()
        fig_full.canvas.draw_idle()
        plt.pause(0.001)

        first_background = True
        while True:
            ans = input("Does the spectrum look ok? [Y/n]: ").strip().lower()
            if ans in ("y", "yes", ""):
                spec_qual_flag = 1
                break
            elif ans in ("n", "no"):
                spec_qual_flag = 0
                plt.close(fig_full)
                return background, None, None, None, vel_width, spec_qual_flag
            else:
                print("Please answer 'y' or 'n'.")

        plt.close(fig_full)

        # ── Main loop ────────────────────────────────────────────────────────
        while True:
            fig_bg = preview_background(df_spec, background, vel_width=vel_width, si_vel_1e3=si_vel_1e3)
            try:
                fig_bg.canvas.manager.set_window_title("Ca II NIR background preview")
            except Exception:
                pass
            fig_bg.show()
            fig_bg.canvas.draw_idle()
            plt.pause(0.001)

            if manual_start and first_background:
                raw_bg = "n"
            else:
                raw_bg = input(
                    "Do you like this Ca II NIR background? [Y/n] "
                    "(or type a new vel_width in 10^3 km/s): "
                ).strip()
            first_background = False
            ans_bg = raw_bg.lower()

            if ans_bg in ("", "y", "yes"):
                plt.close(fig_bg)

                # ── STEP 2: run fit ──────────────────────────────────────────
                try:
                    best_fit_dict, fit_fig, out_back, out_fit = fit_features(
                        df_spec, background, si_vel_1e3,
                        vel_width=vel_width,
                        n_resol_neighbors=n_resol_neighbors,
                        hvf_min_sep_kms=hvf_min_sep_kms,
                        pvf_vel_frac=pvf_vel_frac,
                    )

                    fit_preview = preview_ca_fit(df_spec, background, out_back, out_fit, vel_width=vel_width)
                except Exception as e:
                    plt.close('all')
                    print(
                        f"\n[FIT ERROR] {type(e).__name__}: {e}\n"
                        "The fit failed with the current background anchors "
                        "(often means the red/blue windows are swapped or too "
                        "narrow). Let's redefine the anchors and try again.\n"
                    )
                    continue

                try:
                    fit_preview.canvas.manager.set_window_title("Ca II NIR joint fit")
                except Exception:
                    pass
                fit_preview.show()
                fit_preview.canvas.draw_idle()
                plt.pause(0.001)

                ans_fit = input("Do you like the Ca II NIR fit? [Y/n]: ").strip().lower()
                if ans_fit in ("", "y", "yes"):
                    bg_refined_fig = preview_background(df_spec, background, vel_width=vel_width, si_vel_1e3=si_vel_1e3)
                    return background, bg_refined_fig, best_fit_dict, fit_fig, vel_width, spec_qual_flag

                # Fit rejected → go back
                plt.close(fit_preview)
                print("Fit rejected. Redefine background anchors and refit...\n")
                continue

            # ── Try to parse as new vel_width ────────────────────────────────
            try:
                new_vw = float(raw_bg)
            except ValueError:
                pass
            else:
                if new_vw <= 0:
                    print(f"vel_width must be positive; keeping {vel_width:.1f}.")
                else:
                    vel_width = new_vw
                    print(f"Updated vel_width to {vel_width:.1f}×10³ km/s.\n")
                    plt.close(fig_bg)
                    continue

            # ── Manual anchor re-click ───────────────────────────────────────
            plt.close(fig_bg)

            print("\nPlease select TWO new continuum points on the plot:")
            print("  1) Ca II NIR red-side continuum")
            print("  2) Ca II NIR blue-side continuum")
            print("Click the two points.\n")

            wl     = df_spec["rest_wl"].values
            y_raw  = df_spec["norm_fl"].values
            y_sm   = df_spec["norm_fl_smooth"].values

            fig_click, ax_click = plt.subplots(figsize=(10, 8))
            ax_click.plot(wl, y_raw, color='tab:grey', lw=1.0, label='raw')
            ax_click.plot(wl, y_sm,  color='k',        lw=1.0, label='smoothed')

            # Shade existing windows
            vr_c = background['ca_red_side']
            vb_c = background['ca_blue_side']
            hw   = vel_width / 2.0
            for v_c in [vr_c, vb_c]:
                wl_lo = vel_to_wl(v_c - hw, rest_wl_ca_ref)
                wl_hi = vel_to_wl(v_c + hw, rest_wl_ca_ref)
                ax_click.axvspan(wl_lo, wl_hi, alpha=0.2, color='tab:cyan', zorder=0)

            # Ca II rest wavelength markers
            for rwl in [rest_wl_ca1, rest_wl_ca2, rest_wl_ca3]:
                ax_click.axvline(x=rwl, color='tab:purple', ls=':', lw=0.8)

            ax_click.set_xlim(*wl_xlim)
            autoscale_ylim_for_xlim(wl_xlim, margin=0.05, ax=ax_click)
            ax_click.set_xlabel("Rest wavelength (Å)")
            ax_click.set_ylabel("Norm. flux")
            ax_click.set_title(f"Click 2 continuum points  |  vel_width={vel_width:.1f}×10³ km/s")
            fig_click.tight_layout()
            fig_click.show()
            plt.pause(0.001)

            # Click 1: red side
            for artist in ax_click.texts:
                artist.remove()
            ax_click.text(0.05, 0.95, "click Ca II NIR\nred side",
                          transform=ax_click.transAxes, va="top", ha="left",
                          fontsize=14, color='tab:red', fontweight='bold')
            ax_click.axvline(x=ca_ref_1e4, ls=':', color='tab:orange')
            plt.draw()
            click1 = plt.ginput(1, timeout=-1)

            for artist in ax_click.texts:
                artist.remove()
            ax_click.text(0.05, 0.95, "click Ca II NIR\nblue side",
                          transform=ax_click.transAxes, va="top", ha="left",
                          fontsize=14, color='tab:blue', fontweight='bold')
            plt.draw()
            click2 = plt.ginput(1, timeout=-1)

            plt.close(fig_click)

            if len(click1) < 1 or len(click2) < 1:
                print("Fewer than 2 points clicked; keeping previous background.")
                continue

            w_red  = click1[0][0]
            w_blue = click2[0][0]

            background['ca_red_side']        = wl_to_vel(w_red,  rest_wl_ca_ref)
            background['ca_blue_side']       = wl_to_vel(w_blue, rest_wl_ca_ref)
            background['status_ca_red_side']  = "manual"
            background['status_ca_blue_side'] = "manual"

            # Red side must sit at a higher (less negative) velocity than the
            # blue side. If the clicks came in reversed, swap them so the fit
            # doesn't choke on an inverted/zero-size window.
            if background['ca_red_side'] < background['ca_blue_side']:
                background['ca_red_side'], background['ca_blue_side'] = (
                    background['ca_blue_side'], background['ca_red_side']
                )
                print(
                    "[NOTE] Clicked points looked reversed (red < blue); "
                    "swapped them so red-side > blue-side.\n"
                )

            print("Updated Ca II NIR background anchors (10^3 km/s):")
            print(f"  ca_red_side  = {background['ca_red_side']:.2f}")
            print(f"  ca_blue_side = {background['ca_blue_side']:.2f}\n")

    finally:
        plt.interactive(was_interactive)


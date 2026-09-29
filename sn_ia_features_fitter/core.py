"""Scientific calculations and diagnostic plots for Si II feature fitting."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from lmfit import Model, Parameters, minimize
from scipy.integrate import trapezoid
from scipy.interpolate import make_interp_spline

from . import common
from .common import (
    _choose_unique_anchor as _choose_unique_anchor,
)
from .common import (
    _dedup_wavelength_roots as _dedup_wavelength_roots,
)
from .common import (
    _fig_to_rgb as _fig_to_rgb,
)
from .common import (
    _maxima_candidates_in_window as _maxima_candidates_in_window,
)
from .common import (
    _mc_plot_bin_edges as _mc_plot_bin_edges,
)
from .common import (
    _pick_with_index_exclusion_and_bounds as _pick_with_index_exclusion_and_bounds,
)
from .common import (
    _scan_extrema_in_window as _scan_extrema_in_window,
)
from .common import (
    autoscale_ylim_for_xlim as autoscale_ylim_for_xlim,
)
from .common import (
    calc_optimal_smooth as calc_optimal_smooth,
)
from .common import (
    clip_outliers as clip_outliers,
)
from .common import (
    lin_back as lin_back,
)
from .common import (
    qc_mean_median_gap as qc_mean_median_gap,
)
from .common import (
    save_summary_figure as save_summary_figure,
)
from .common import (
    smooth_spec as smooth_spec,
)
from .common import (
    spec_create_errors as spec_create_errors,
)
from .common import (
    vel_to_wl as vel_to_wl,
)
from .common import (
    wl_to_vel as wl_to_vel,
)
from .coverage import RequiredInterval
from .style import apply_plot_style

apply_plot_style()

speed_of_light = 2.99792e5  # km/s
rest_wl_sil1_1 = 6347.10
rest_wl_sil1_2 = 6371.36
rest_wl_sil2_1 = 5957.56
rest_wl_sil2_2 = 5978.93


def required_rest_intervals(config):
    """Si coverage, retaining the established automatic-background span."""
    half_width_kms = config.vel_width * 500.0
    bounds = (
        (rest_wl_sil1_1, config.sil6355_red_bounds_kms),
        (rest_wl_sil1_1, config.sil6355_blue_bounds_kms),
        (rest_wl_sil2_1, config.sil5972_blue_bounds_kms),
    )
    lower = min(5250.0, *(vel_to_wl((pair[0] - half_width_kms) / 1000.0, line)
                            for line, pair in bounds))
    upper = max(6750.0, *(vel_to_wl((pair[1] + half_width_kms) / 1000.0, line)
                            for line, pair in bounds))
    return [RequiredInterval("Si II", float(lower), float(upper))]

def prepare_spectrum(file_spec, redshift, mwebv, vexp=None, type_of_spec="auto", try_clip=True):
    """Prepare shared spectrum columns and Si II velocity coordinates."""
    df_spec = common.prepare_spectrum(
        file_spec, redshift, mwebv, vexp=vexp,
        type_of_spec=type_of_spec, try_clip=try_clip,
    )
    df_spec.loc[:, "vel_sil1_1"] = wl_to_vel(df_spec["rest_wl"], rest_wl_sil1_1)
    df_spec.loc[:, "vel_sil2_1"] = wl_to_vel(df_spec["rest_wl"], rest_wl_sil2_1)
    return df_spec


def g_sil1_1(x, a_sil1_1, v_sil1_1, fwhm_sil1_1):
    zmeas = (x - rest_wl_sil1_1) / rest_wl_sil1_1
    vel = (((zmeas + 1.0) ** 2 - 1.0) * speed_of_light / (1.0 + (1.0 + zmeas) ** 2)) / 1e3
    sigma_gaussian = np.abs(fwhm_sil1_1) / 2.35482
    gaussian = a_sil1_1 * np.exp(
        -np.power(vel - v_sil1_1, 2.0) / (2 * np.power(sigma_gaussian, 2.0))
    )
    return gaussian


def g_sil1_2(x, a_sil1_2, v_sil1_2, fwhm_sil1_2):
    zmeas = (x - rest_wl_sil1_2) / rest_wl_sil1_2
    vel = (((zmeas + 1.0) ** 2 - 1.0) * speed_of_light / (1.0 + (1.0 + zmeas) ** 2)) / 1e3
    sigma_gaussian = np.abs(fwhm_sil1_2) / 2.35482
    gaussian = a_sil1_2 * np.exp(
        -np.power(vel - v_sil1_2, 2.0) / (2 * np.power(sigma_gaussian, 2.0))
    )
    return gaussian


def g_sil2_1(x, a_sil2_1, v_sil2_1, fwhm_sil2_1):
    zmeas = (x - rest_wl_sil2_1) / rest_wl_sil2_1
    vel = (((zmeas + 1.0) ** 2 - 1.0) * speed_of_light / (1.0 + (1.0 + zmeas) ** 2)) / 1e3
    sigma_gaussian = np.abs(fwhm_sil2_1) / 2.35482
    gaussian = a_sil2_1 * np.exp(
        -np.power(vel - v_sil2_1, 2.0) / (2 * np.power(sigma_gaussian, 2.0))
    )
    return gaussian


def g_sil2_2(x, a_sil2_2, v_sil2_2, fwhm_sil2_2):
    zmeas = (x - rest_wl_sil2_2) / rest_wl_sil2_2
    vel = (((zmeas + 1.0) ** 2 - 1.0) * speed_of_light / (1.0 + (1.0 + zmeas) ** 2)) / 1e3
    sigma_gaussian = np.abs(fwhm_sil2_2) / 2.35482
    gaussian = a_sil2_2 * np.exp(
        -np.power(vel - v_sil2_2, 2.0) / (2 * np.power(sigma_gaussian, 2.0))
    )
    return gaussian


def find_background_regions(
    df_spec,
    sil6355_red_side_init,
    sil6355_blue_side_init,
    sil5972_blue_side_init,
    v_window_kms=10000.0,
    dv_kms_red=+1000.0,
    dv_kms_blue=-1000.0,
    min_sep=5,
    max_attempts=5,
    sil6355_red_bounds_kms=(-15000.0, 15000.0),
    sil6355_blue_bounds_kms=(-30000.0, -15000.0),
    sil5972_blue_bounds_kms=(-30000.0, -10000.0),
):
    """
    Locate local continuum (background) anchor points for the Si II features
    (6355 Å and 5972 Å) using spline derivatives and maxima search, with:
      • **index-based uniqueness** (≥ `min_sep` wavelength points apart),
      • **bi-directional retries** up to `max_attempts`,
      • **per-anchor velocity constraints** that the results must satisfy.

    Parameters
    ----------
    df_spec : pandas.DataFrame
            Must contain:
              - 'rest_wl' (Å)
              - 'norm_fl_flattened'
              - 'norm_fl_smooth_flattened'
    sil6355_red_side_init, sil6355_blue_side_init, sil5972_blue_side_init : float
            Initial guesses in velocity (10^3 km/s).
    v_window_kms : float, optional
            Half-width of velocity search window. Default 4000 km/s.
    dv_kms_red, dv_kms_blue : float, optional
            Velocity step for shifting the window center (km/s).
            The algorithm tries centers at 0, ±1, ±2, ... × (dv_kms/1e3).
            Defaults: +1000 (red side), −1000 (blue sides).
    min_sep : int, optional
            Minimum **index separation** (wavelength grid points) between any two anchors.
            Default 5.
    max_attempts : int, optional
            Number of shift steps per direction after the initial attempt.
            Total centers tested = 1 + 2*max_attempts. Default 2.
    sil6355_red_bounds_kms : (float, float)
            Allowed velocity range (km/s) for the 6355 red-side anchor. Default (-15000, +15000).
    sil6355_blue_bounds_kms : (float, float)
            Allowed velocity range (km/s) for the 6355 blue-side anchor. Default (-30000, -15000).
    sil5972_blue_bounds_kms : (float, float)
            Allowed velocity range (km/s) for the 5972 blue-side anchor. Default (-30000, -10000).

    Returns
    -------
    results : dict
            Velocities in 10^3 km/s and statuses:
              - 'sil6355_red_side', 'sil6355_blue_side', 'sil5972_red_side', 'sil5972_blue_side'
              - one ``status_...`` field for each anchor
            Status values: 'automatic' | 'out_of_bounds' | 'conflict_no_unique_max'
                                       | 'minima_only' | 'fallback_initial' | 'clipped_to_bounds'
                                       | 'derived_from_sil6355_blue'
            Note: 'sil5972_red_side' is derived from the chosen wavelength of 6355 blue-side.
    fig : matplotlib.figure.Figure
            Diagnostics (spectrum + derivatives).

    Notes
    -----
    - Candidates violating bounds are ignored during selection.
    - If no acceptable maximum is found, we fallback to the initial guess; if that
      violates bounds, we **clip** to the nearest bound (status 'clipped_to_bounds').
    - Requires external helpers/constants:
            `vel_to_wl(v_1e3, rest_wl)`, `wl_to_vel(wl, rest_wl)`,
            `make_interp_spline(x, y, k=3)` (object with `.derivative(nu)`),
            `rest_wl_sil1_1` (~6355 Å), `rest_wl_sil2_1` (~5972 Å),
            and optionally `autoscale_ylim_for_xlim(xlim, margin, ax)`.
    """
    # Prepare spline derivatives
    y_spl_sil = make_interp_spline(df_spec["rest_wl"], df_spec["norm_fl_smooth_flattened"], k=3)
    y_spl_1d_sil = y_spl_sil.derivative(nu=1)
    y_spl_2d_sil = y_spl_sil.derivative(nu=2)

    wl_grid = df_spec["rest_wl"].values

    # Initial guesses (for plotting)
    sil6355_red_side_wl = vel_to_wl(sil6355_red_side_init, rest_wl_sil1_1)
    sil6355_blue_side_wl = vel_to_wl(sil6355_blue_side_init, rest_wl_sil1_1)
    sil5972_blue_side_wl = vel_to_wl(sil5972_blue_side_init, rest_wl_sil2_1)

    # keep exclusion list as grid indices
    exclude_idx = []

    def wl_to_idx(w):
        return int(np.argmin(np.abs(wl_grid - w)))

    # --- find anchors with uniqueness, retries, and bounds ---
    r1_wl, st_r1 = _choose_unique_anchor(
        y_spl_1d_sil,
        y_spl_2d_sil,
        sil6355_red_side_init,
        rest_wl_sil1_1,
        v_window_kms,
        dv_kms_red,
        wl_grid,
        exclude_idx,
        min_sep,
        max_attempts,
        sil6355_red_bounds_kms,
    )
    exclude_idx.append(wl_to_idx(r1_wl))

    b1_wl, st_b1 = _choose_unique_anchor(
        y_spl_1d_sil,
        y_spl_2d_sil,
        sil6355_blue_side_init,
        rest_wl_sil1_1,
        v_window_kms,
        dv_kms_blue,
        wl_grid,
        exclude_idx,
        min_sep,
        max_attempts,
        sil6355_blue_bounds_kms,
    )
    exclude_idx.append(wl_to_idx(b1_wl))

    b2_wl, st_b2 = _choose_unique_anchor(
        y_spl_1d_sil,
        y_spl_2d_sil,
        sil5972_blue_side_init,
        rest_wl_sil2_1,
        v_window_kms,
        dv_kms_blue,
        wl_grid,
        exclude_idx,
        min_sep,
        max_attempts,
        sil5972_blue_bounds_kms,
    )
    exclude_idx.append(wl_to_idx(b2_wl))

    results = {
        "sil6355_red_side": wl_to_vel(r1_wl, rest_wl_sil1_1),
        "sil6355_blue_side": wl_to_vel(b1_wl, rest_wl_sil1_1),
        "sil5972_red_side": wl_to_vel(b1_wl, rest_wl_sil2_1),  # tied to b1 wl
        "sil5972_blue_side": wl_to_vel(b2_wl, rest_wl_sil2_1),
        "status_sil6355_red_side": st_r1,
        "status_sil6355_blue_side": st_b1,
        "status_sil5972_red_side": "derived_from_sil6355_blue",
        "status_sil5972_blue_side": st_b2,
    }

    # --- diagnostics plotting ---
    fig, axs = plt.subplots(ncols=1, nrows=3, sharex="col", figsize=(6, 8))
    ax_sil, ax_d1, ax_d2 = axs.flatten()

    wl = df_spec["rest_wl"]
    y_raw = df_spec["norm_fl_flattened"]
    y_smooth = df_spec["norm_fl_smooth_flattened"]

    ax_sil.plot(wl, y_raw, color="tab:grey", label="raw")
    ax_sil.plot(wl, y_smooth, color="k", label="smoothed")

    ax_sil.plot(
        [sil6355_red_side_wl, sil6355_blue_side_wl, sil5972_blue_side_wl],
        y_spl_sil([sil6355_red_side_wl, sil6355_blue_side_wl, sil5972_blue_side_wl]),
        "o",
        color="tab:green",
        label="guesses",
    )

    ax_sil.plot(
        [r1_wl, b1_wl, b2_wl],
        y_spl_sil([r1_wl, b1_wl, b2_wl]),
        "o",
        mfc="none",
        mew=2.5,
        color="tab:red",
        label="solutions",
    )

    autoscale_ylim_for_xlim((5250, 6750), margin=0.1, ax=ax_sil)
    ax_sil.legend()

    ax_d1.plot(wl, y_spl_1d_sil(wl), color="tab:blue", label="1st deriv")
    ax_d1.axhline(0, ls="--", color="tab:red")

    ax_d1.plot(
        [sil6355_red_side_wl, sil6355_blue_side_wl, sil5972_blue_side_wl],
        y_spl_1d_sil([sil6355_red_side_wl, sil6355_blue_side_wl, sil5972_blue_side_wl]),
        "o",
        markersize=5,
        color="tab:green",
        label="guesses",
    )
    ax_d1.plot(
        [r1_wl, b1_wl, b2_wl],
        y_spl_1d_sil([r1_wl, b1_wl, b2_wl]),
        "o",
        mfc="none",
        mew=2.5,
        color="tab:red",
        label="solutions",
    )
    autoscale_ylim_for_xlim((5250, 6750), margin=0.1, ax=ax_d1)
    ax_d1.legend()

    ax_d2.plot(wl, y_spl_2d_sil(wl), color="tab:orange", label="2nd deriv")
    ax_d2.axhline(0, ls="--", color="tab:red")
    ax_d2.plot(
        [sil6355_red_side_wl, sil6355_blue_side_wl, sil5972_blue_side_wl],
        y_spl_2d_sil([sil6355_red_side_wl, sil6355_blue_side_wl, sil5972_blue_side_wl]),
        "o",
        markersize=5,
        color="tab:green",
        label="guesses",
    )
    ax_d2.plot(
        [r1_wl, b1_wl, b2_wl],
        y_spl_2d_sil([r1_wl, b1_wl, b2_wl]),
        "o",
        mfc="none",
        mew=2.5,
        color="tab:red",
        label="solutions",
    )
    autoscale_ylim_for_xlim((5250, 6750), margin=0.1, ax=ax_d2)
    ax_d2.legend()
    ax_d2.set_xlabel("Rest wavelength (Å)")

    fig.tight_layout()
    plt.close(fig)

    return results, fig


def fit_features(
    df_spec,
    background_dict,
    vel_width: float = 4.0,
    n_resol_neighbors: int = 4,
    k_vel12_bounds: tuple[float, float] = (0.75, 1.25),
    k_fwhm12_bounds: tuple[float, float] = (0.25, 1.75),
):
    """
    Jointly fit Si II λ6355 and λ5972 with shared k-parameters (deterministic).

    This performs a coupled least-squares fit using `lmfit.minimize` where the
    λ5972 velocity and FWHM track λ6355 via scale factors k_vel12 and k_fwhm12.

    Parameters
    ----------
    df_spec : pandas.DataFrame
            Spectrum dataframe from `prepare_spectrum`.
    background_dict : dict
            From `find_background_regions` (contains velocity bounds).
    vel_width : float, optional
            Half-width of the background windows (10^3 km/s). Default 4.0.
    n_resol_neighbors : int, optional
            Number of nearest points around each line’s rest wavelength used to
            estimate instrumental resolution in velocity space. Default 4.
    k_vel12_bounds : (float, float), optional
            (min, max) bounds for k_vel12. Default (0.75, 1.25).
    k_fwhm12_bounds : (float, float), optional
            (min, max) bounds for k_fwhm12. Default (0.25, 1.75).

    Returns
    -------
    metrics : dict
            Velocity, FWHM, EW and their errors for both lines + r_sil.
    fig : matplotlib.figure.Figure
            Summary plot (data, background, model components).
    """
    df_spec_for_back_sil1 = df_spec.loc[
        (
            (df_spec["vel_sil1_1"] >= background_dict["sil6355_blue_side"] - vel_width / 2.0)
            & (df_spec["vel_sil1_1"] <= background_dict["sil6355_blue_side"] + vel_width / 2.0)
        )
        | (
            (df_spec["vel_sil1_1"] >= background_dict["sil6355_red_side"] - vel_width / 2.0)
            & (df_spec["vel_sil1_1"] <= background_dict["sil6355_red_side"] + vel_width / 2.0)
        )
    ]
    df_spec_for_back_sil2 = df_spec.loc[
        (
            (df_spec["vel_sil2_1"] >= background_dict["sil5972_blue_side"] - vel_width / 2.0)
            & (df_spec["vel_sil2_1"] <= background_dict["sil5972_blue_side"] + vel_width / 2.0)
        )
        | (
            (df_spec["vel_sil2_1"] >= background_dict["sil5972_red_side"] - vel_width / 2.0)
            & (df_spec["vel_sil2_1"] <= background_dict["sil5972_red_side"] + vel_width / 2.0)
        )
    ]

    wl_back_sil1 = df_spec_for_back_sil1["rest_wl"].values
    y_back_sil1 = df_spec_for_back_sil1["norm_fl"].values
    yerr_back_sil1 = df_spec_for_back_sil1["norm_flerr"].values

    wl_back_sil2 = df_spec_for_back_sil2["rest_wl"].values
    y_back_sil2 = df_spec_for_back_sil2["norm_fl"].values
    yerr_back_sil2 = df_spec_for_back_sil2["norm_flerr"].values

    df_spec_for_fit_sil1 = df_spec.loc[
        (df_spec["vel_sil1_1"] >= background_dict["sil6355_blue_side"])
        & (df_spec["vel_sil1_1"] <= background_dict["sil6355_red_side"])
    ]
    wl_fit_sil1 = df_spec_for_fit_sil1["rest_wl"].values
    vel_fit_sil1 = df_spec_for_fit_sil1["vel_sil1_1"].values

    df_spec_for_fit_sil2 = df_spec.loc[
        (df_spec["vel_sil2_1"] >= background_dict["sil5972_blue_side"])
        & (df_spec["vel_sil2_1"] <= background_dict["sil5972_red_side"])
    ]
    wl_fit_sil2 = df_spec_for_fit_sil2["rest_wl"].values
    vel_fit_sil2 = df_spec_for_fit_sil2["vel_sil2_1"].values

    resol_wl_sil1 = np.argsort(np.abs(wl_fit_sil1 - rest_wl_sil1_1))[:n_resol_neighbors]
    closest1 = wl_fit_sil1[resol_wl_sil1]
    resol_vel_sil1 = np.abs(
        wl_to_vel(closest1.max(), rest_wl_sil1_1) - wl_to_vel(closest1.min(), rest_wl_sil1_1)
    )

    model_fit_back_sil1 = Model(lin_back)
    pars_back_sil1 = Parameters()
    pars_back_sil1.add("a", value=0.0)
    pars_back_sil1.add("b", value=0.0)
    out_back_sil1 = model_fit_back_sil1.fit(
        y_back_sil1, pars_back_sil1, x=wl_back_sil1, weights=1.0 / yerr_back_sil1
    )
    back_sil1 = lin_back(
        wl_fit_sil1, out_back_sil1.params["a"].value, out_back_sil1.params["b"].value
    )
    if np.any(~np.isfinite(back_sil1)) or np.any(np.abs(back_sil1) < 1e-12):
        raise ValueError("Si II 6355 continuum fit is zero or non-finite")

    y_fit_sil1 = (df_spec_for_fit_sil1["norm_fl"].values / back_sil1) - 1.0
    yerr_fit_sil1 = df_spec_for_fit_sil1["norm_flerr"] / back_sil1

    model_fit_back_sil2 = Model(lin_back)
    pars_back_sil2 = Parameters()
    pars_back_sil2.add("a", value=0.0)
    pars_back_sil2.add("b", value=0.0)
    out_back_sil2 = model_fit_back_sil2.fit(
        y_back_sil2, pars_back_sil2, x=wl_back_sil2, weights=1.0 / yerr_back_sil2
    )
    back_sil2 = lin_back(
        wl_fit_sil2, out_back_sil2.params["a"].value, out_back_sil2.params["b"].value
    )
    if np.any(~np.isfinite(back_sil2)) or np.any(np.abs(back_sil2) < 1e-12):
        raise ValueError("Si II 5972 continuum fit is zero or non-finite")

    y_fit_sil2 = (df_spec_for_fit_sil2["norm_fl"].values / back_sil2) - 1.0
    yerr_fit_sil2 = df_spec_for_fit_sil2["norm_flerr"] / back_sil2

    pars_fit_sil = Parameters()
    pars_fit_sil.add("a_sil1_1", value=-1.1, max=0.0)
    pars_fit_sil.add("v_sil1_1", value=-11.0, max=0.0)
    pars_fit_sil.add("fwhm_sil1_1", value=8.0, min=resol_vel_sil1)
    pars_fit_sil.add("a_sil1_2", expr="a_sil1_1")
    pars_fit_sil.add("v_sil1_2", expr="v_sil1_1")
    pars_fit_sil.add("fwhm_sil1_2", expr="fwhm_sil1_1")
    pars_fit_sil.add("k_vel12", value=1.0, min=k_vel12_bounds[0], max=k_vel12_bounds[1])
    pars_fit_sil.add("k_fwhm12", value=1.0, min=k_fwhm12_bounds[0], max=k_fwhm12_bounds[1])
    pars_fit_sil.add("a_sil2_1", value=-1.0, max=0.0)
    pars_fit_sil.add("v_sil2_1", expr="k_vel12 * v_sil1_1")
    pars_fit_sil.add("fwhm_sil2_1", expr="k_fwhm12 * fwhm_sil1_1")
    pars_fit_sil.add("a_sil2_2", expr="a_sil2_1")
    pars_fit_sil.add("v_sil2_2", expr="v_sil2_1")
    pars_fit_sil.add("fwhm_sil2_2", expr="fwhm_sil2_1")

    def model_sil1(params, wl):
        model = g_sil1_1(
            wl, params["a_sil1_1"], params["v_sil1_1"], params["fwhm_sil1_1"]
        ) + g_sil1_2(wl, params["a_sil1_2"], params["v_sil1_2"], params["fwhm_sil1_2"])
        return model

    def model_sil2(params, wl):
        model = g_sil2_1(
            wl, params["a_sil2_1"], params["v_sil2_1"], params["fwhm_sil2_1"]
        ) + g_sil2_2(wl, params["a_sil2_2"], params["v_sil2_2"], params["fwhm_sil2_2"])
        return model

    def resid(params):
        m1 = model_sil1(params, wl_fit_sil1)
        m2 = model_sil2(params, wl_fit_sil2)
        r1 = (y_fit_sil1 - m1) / yerr_fit_sil1
        r2 = (y_fit_sil2 - m2) / yerr_fit_sil2
        return np.r_[r1, r2]

    out_fit_sil = minimize(resid, pars_fit_sil, method="least_squares")
    if not out_fit_sil.success:
        raise RuntimeError(f"Joint Si II optimizer failed: {out_fit_sil.message}")

    sil1_1 = g_sil1_1(
        wl_fit_sil1,
        out_fit_sil.params["a_sil1_1"].value,
        out_fit_sil.params["v_sil1_1"].value,
        out_fit_sil.params["fwhm_sil1_1"].value,
    )
    sil1_2 = g_sil1_2(
        wl_fit_sil1,
        out_fit_sil.params["a_sil1_2"].value,
        out_fit_sil.params["v_sil1_2"].value,
        out_fit_sil.params["fwhm_sil1_2"].value,
    )

    sil2_1 = g_sil2_1(
        wl_fit_sil2,
        out_fit_sil.params["a_sil2_1"].value,
        out_fit_sil.params["v_sil2_1"].value,
        out_fit_sil.params["fwhm_sil2_1"].value,
    )
    sil2_2 = g_sil2_2(
        wl_fit_sil2,
        out_fit_sil.params["a_sil2_2"].value,
        out_fit_sil.params["v_sil2_2"].value,
        out_fit_sil.params["fwhm_sil2_2"].value,
    )

    wl_fit_sil1_for_ew = np.linspace(wl_fit_sil1[0], wl_fit_sil1[-1], 300)
    wl_fit_sil2_for_ew = np.linspace(wl_fit_sil2[0], wl_fit_sil2[-1], 300)

    sil1_1_for_ew = g_sil1_1(
        wl_fit_sil1_for_ew,
        out_fit_sil.params["a_sil1_1"].value,
        out_fit_sil.params["v_sil1_1"].value,
        out_fit_sil.params["fwhm_sil1_1"].value,
    )
    sil1_2_for_ew = g_sil1_2(
        wl_fit_sil1_for_ew,
        out_fit_sil.params["a_sil1_2"].value,
        out_fit_sil.params["v_sil1_2"].value,
        out_fit_sil.params["fwhm_sil1_2"].value,
    )

    sil2_1_for_ew = g_sil2_1(
        wl_fit_sil2_for_ew,
        out_fit_sil.params["a_sil2_1"].value,
        out_fit_sil.params["v_sil2_1"].value,
        out_fit_sil.params["fwhm_sil2_1"].value,
    )
    sil2_2_for_ew = g_sil2_2(
        wl_fit_sil2_for_ew,
        out_fit_sil.params["a_sil2_2"].value,
        out_fit_sil.params["v_sil2_2"].value,
        out_fit_sil.params["fwhm_sil2_2"].value,
    )

    ew_sil1 = np.abs(trapezoid(sil1_1_for_ew + sil1_2_for_ew, x=wl_fit_sil1_for_ew))
    ew_sil2 = np.abs(trapezoid(sil2_1_for_ew + sil2_2_for_ew, x=wl_fit_sil2_for_ew))

    ew_sil1_raw = np.abs(trapezoid(y_fit_sil1, x=df_spec_for_fit_sil1["rest_wl"]))
    ew_sil2_raw = np.abs(trapezoid(y_fit_sil2, x=df_spec_for_fit_sil2["rest_wl"]))

    fig, axs = plt.subplots(ncols=2, nrows=2, sharex="col", figsize=(9, 5))
    ax_sil1, ax_sil2, ax_sil1_backnorm, ax_sil2_backnorm = axs.flatten()

    ax_sil1.plot(
        df_spec["vel_sil1_1"], df_spec["norm_fl"], color="tab:grey", label="raw", linewidth=2
    )

    ax_sil1.plot(vel_fit_sil1, back_sil1, color="tab:red", linestyle="--", label="background")

    ax_sil1.plot(vel_fit_sil1, (sil1_1 + 1) * back_sil1, color="tab:green")
    ax_sil1.plot(vel_fit_sil1, (sil1_2 + 1) * back_sil1, color="tab:blue")

    ax_sil1.plot(
        vel_fit_sil1,
        (sil1_1 + sil1_2 + 1) * back_sil1,
        color="tab:orange",
        label="Si II model",
    )

    ax_sil1.plot(
        df_spec_for_back_sil1["vel_sil1_1"],
        df_spec_for_back_sil1["norm_fl"],
        "o",
        markerfacecolor="none",
        color="k",
    )

    autoscale_ylim_for_xlim((-50, 20), margin=0.05, ax=ax_sil1)

    ax_sil1.axvspan(
        background_dict["sil6355_blue_side"] - vel_width / 2.0,
        background_dict["sil6355_blue_side"] + vel_width / 2.0,
        alpha=0.25,
        color="tab:cyan",
        label="back region",
    )
    ax_sil1.axvspan(
        background_dict["sil6355_red_side"] - vel_width / 2.0,
        background_dict["sil6355_red_side"] + vel_width / 2.0,
        alpha=0.25,
        color="tab:cyan",
    )

    ax_sil1.legend()

    ax_sil1.set_title("Si II λ6355")

    ###
    back_sil1_plot = lin_back(
        df_spec["rest_wl"], out_back_sil1.params["a"].value, out_back_sil1.params["b"].value
    )
    ax_sil1_backnorm.plot(
        df_spec["vel_sil1_1"],
        (df_spec["norm_fl"].values / back_sil1_plot) - 1.0,
        color="tab:grey",
        linewidth=2,
    )

    vel_plot = np.linspace(vel_fit_sil1[0], vel_fit_sil1[-1], 300)
    wl_plot = vel_to_wl(vel_plot, rest_wl_sil1_1)
    sil1_1_plot = g_sil1_1(
        wl_plot,
        out_fit_sil.params["a_sil1_1"].value,
        out_fit_sil.params["v_sil1_1"].value,
        out_fit_sil.params["fwhm_sil1_1"].value,
    )
    sil1_2_plot = g_sil1_2(
        wl_plot,
        out_fit_sil.params["a_sil1_2"].value,
        out_fit_sil.params["v_sil1_2"].value,
        out_fit_sil.params["fwhm_sil1_2"].value,
    )

    ax_sil1_backnorm.plot(vel_plot, (sil1_1_plot), color="tab:green")
    ax_sil1_backnorm.plot(vel_plot, (sil1_2_plot), color="tab:blue")
    ax_sil1_backnorm.plot(vel_plot, (sil1_1_plot + sil1_2_plot), color="tab:orange")

    autoscale_ylim_for_xlim((-50, 20), margin=0.05, ax=ax_sil1_backnorm)

    ax_sil1_backnorm.axvspan(
        background_dict["sil6355_blue_side"] - vel_width / 2.0,
        background_dict["sil6355_blue_side"] + vel_width / 2.0,
        alpha=0.25,
        color="tab:cyan",
    )
    ax_sil1_backnorm.axvspan(
        background_dict["sil6355_red_side"] - vel_width / 2.0,
        background_dict["sil6355_red_side"] + vel_width / 2.0,
        alpha=0.25,
        color="tab:cyan",
    )

    ax_sil1_backnorm.axhline(y=0, color="tab:red", linestyle="--")

    ax_sil1_backnorm.set_xlabel("velocity (10³ km/s)")

    #####
    ax_sil2.plot(
        df_spec["vel_sil2_1"], df_spec["norm_fl"], color="tab:grey", label="raw", linewidth=2
    )

    ax_sil2.plot(vel_fit_sil2, back_sil2, color="tab:red", linestyle="--", label="background")

    ax_sil2.plot(vel_fit_sil2, (sil2_1 + 1) * back_sil2, color="tab:green")
    ax_sil2.plot(vel_fit_sil2, (sil2_2 + 1) * back_sil2, color="tab:blue")

    ax_sil2.plot(
        vel_fit_sil2,
        (sil2_1 + sil2_2 + 1) * back_sil2,
        color="tab:orange",
        label="Si II model",
    )

    ax_sil2.plot(
        df_spec_for_back_sil2["vel_sil2_1"],
        df_spec_for_back_sil2["norm_fl"],
        "o",
        markerfacecolor="none",
        color="k",
    )

    autoscale_ylim_for_xlim((-50, 20), margin=0.05, ax=ax_sil2)

    ax_sil2.axvspan(
        background_dict["sil5972_blue_side"] - vel_width / 2.0,
        background_dict["sil5972_blue_side"] + vel_width / 2.0,
        alpha=0.25,
        color="tab:cyan",
        label="back region",
    )
    ax_sil2.axvspan(
        background_dict["sil5972_red_side"] - vel_width / 2.0,
        background_dict["sil5972_red_side"] + vel_width / 2.0,
        alpha=0.25,
        color="tab:cyan",
    )
    ax_sil2.legend()

    ax_sil2.set_title("Si II λ5972")

    ###

    back_sil2_plot = lin_back(
        df_spec["rest_wl"], out_back_sil2.params["a"].value, out_back_sil2.params["b"].value
    )
    ax_sil2_backnorm.plot(
        df_spec["vel_sil2_1"],
        (df_spec["norm_fl"].values / back_sil2_plot) - 1.0,
        color="tab:grey",
        linewidth=2,
    )

    vel_plot = np.linspace(vel_fit_sil2[0], vel_fit_sil2[-1], 300)
    wl_plot = vel_to_wl(vel_plot, rest_wl_sil2_1)
    sil2_1_plot = g_sil2_1(
        wl_plot,
        out_fit_sil.params["a_sil2_1"].value,
        out_fit_sil.params["v_sil2_1"].value,
        out_fit_sil.params["fwhm_sil2_1"].value,
    )
    sil2_2_plot = g_sil2_2(
        wl_plot,
        out_fit_sil.params["a_sil2_2"].value,
        out_fit_sil.params["v_sil2_2"].value,
        out_fit_sil.params["fwhm_sil2_2"].value,
    )

    ax_sil2_backnorm.plot(vel_plot, (sil2_1_plot), color="tab:green")
    ax_sil2_backnorm.plot(vel_plot, (sil2_2_plot), color="tab:blue")
    ax_sil2_backnorm.plot(vel_plot, (sil2_1_plot + sil2_2_plot), color="tab:orange")

    autoscale_ylim_for_xlim((-50, 20), margin=0.05, ax=ax_sil2_backnorm)

    ax_sil2_backnorm.axvspan(
        background_dict["sil5972_blue_side"] - vel_width / 2.0,
        background_dict["sil5972_blue_side"] + vel_width / 2.0,
        alpha=0.25,
        color="tab:cyan",
    )
    ax_sil2_backnorm.axvspan(
        background_dict["sil5972_red_side"] - vel_width / 2.0,
        background_dict["sil5972_red_side"] + vel_width / 2.0,
        alpha=0.25,
        color="tab:cyan",
    )

    ax_sil2_backnorm.axhline(y=0, color="tab:red", linestyle="--")

    ax_sil2_backnorm.set_xlabel("velocity (10³ km/s)")

    fig.tight_layout()
    plt.close(fig)

    # --- SNR gating for Si II 6355 ---
    try:
        # Model for 6355 at fitted wavelengths
        sil1_model = model_sil1(out_fit_sil.params, wl_fit_sil1)
        # Residuals in back-normalized flux units
        residuals_6355 = y_fit_sil1 - sil1_model
        rms_6355 = np.sqrt(np.nanmean(residuals_6355**2)) if residuals_6355.size else np.inf

        # Compute peak-to-trough signal from the model
        peak_signal = np.nanmax(sil1_model)
        background_signal = np.nanmin(sil1_model)
        signal_difference = peak_signal - background_signal
        SNR_6355 = signal_difference / rms_6355 if rms_6355 > 0 else np.inf
    except Exception as _e:
        # If anything goes wrong during SNR calc, do not block normal flow
        print(f"SNR calculation error for Si II 6355: {str(_e)}")
        SNR_6355 = np.nan  # Set to NaN on error (saved in return dict)
        pass
    # --- end SNR gating ---

    # --- SNR gating for Si II 5972 ---
    try:
        # Model for 5972 at fitted wavelengths
        sil2_model = model_sil2(out_fit_sil.params, wl_fit_sil2)
        # Residuals in back-normalized flux units
        residuals_5972 = y_fit_sil2 - sil2_model
        rms_5972 = np.sqrt(np.nanmean(residuals_5972**2)) if residuals_5972.size else np.inf

        # Compute peak-to-trough signal from the model
        peak_signal_2 = np.nanmax(sil2_model)
        background_signal_2 = np.nanmin(sil2_model)
        signal_difference_2 = peak_signal_2 - background_signal_2
        SNR_5972 = signal_difference_2 / rms_5972 if rms_5972 > 0 else np.inf
    except Exception as _e:
        print(f"SNR calculation error for Si II 5972: {str(_e)}")
        SNR_5972 = np.nan
        pass
    # --- end SNR gating ---

    return (
        dict(
            optimizer_success=bool(out_fit_sil.success),
            optimizer_message=str(out_fit_sil.message),
            optimizer_chisqr=out_fit_sil.chisqr,
            optimizer_redchi=out_fit_sil.redchi,
            optimizer_aic=out_fit_sil.aic,
            optimizer_bic=out_fit_sil.bic,
            optimizer_nfev=out_fit_sil.nfev,
            sil6355_vel=out_fit_sil.params["v_sil1_1"].value,
            sil6355_vel_err=out_fit_sil.params["v_sil1_1"].stderr,
            sil6355_fwhm=out_fit_sil.params["fwhm_sil1_1"].value,
            sil6355_fwhm_err=out_fit_sil.params["fwhm_sil1_1"].stderr,
            sil6355_6347_fwhm_angstrom=common.velocity_fwhm_to_angstrom(
                out_fit_sil.params["v_sil1_1"].value,
                out_fit_sil.params["fwhm_sil1_1"].value, rest_wl_sil1_1,
            ),
            sil6355_6371_fwhm_angstrom=common.velocity_fwhm_to_angstrom(
                out_fit_sil.params["v_sil1_1"].value,
                out_fit_sil.params["fwhm_sil1_1"].value, rest_wl_sil1_2,
            ),
            sil6355_amp=out_fit_sil.params["a_sil1_1"].value,
            sil6355_ew=ew_sil1,
            sil6355_ew_raw=ew_sil1_raw,
            sil6355_snr=SNR_6355,
            sil5972_vel=out_fit_sil.params["v_sil2_1"].value,
            sil5972_vel_err=out_fit_sil.params["v_sil2_1"].stderr,
            sil5972_fwhm=out_fit_sil.params["fwhm_sil2_1"].value,
            sil5972_fwhm_err=out_fit_sil.params["fwhm_sil2_1"].stderr,
            sil5972_5958_fwhm_angstrom=common.velocity_fwhm_to_angstrom(
                out_fit_sil.params["v_sil2_1"].value,
                out_fit_sil.params["fwhm_sil2_1"].value, rest_wl_sil2_1,
            ),
            sil5972_5979_fwhm_angstrom=common.velocity_fwhm_to_angstrom(
                out_fit_sil.params["v_sil2_1"].value,
                out_fit_sil.params["fwhm_sil2_1"].value, rest_wl_sil2_2,
            ),
            sil5972_amp=out_fit_sil.params["a_sil2_1"].value,
            sil5972_ew=ew_sil2,
            sil5972_ew_raw=ew_sil2_raw,
            sil5972_snr=SNR_5972,
            r_sil=np.nan if ew_sil1 == 0 else ew_sil2 / ew_sil1,
        ),
        fig,
        out_back_sil1,
        out_back_sil2,
        out_fit_sil,
    )


def fit_features_mc(
    df_spec,
    background_dict,
    n_iter: int = 1000,
    vel_width: float = 4.0,
    n_resol_neighbors: int = 4,
    k_vel12_bounds: tuple[float, float] = (0.75, 1.25),
    k_fwhm12_bounds: tuple[float, float] = (0.25, 1.75),
    mode: str = "resample",
    continuum_shift: float = 1.3,
    seed=None,
):
    """
    Monte-Carlo propagation for the joint Si II fit with three uncertainty modes.

    Modes
    -----
    resample
            Current behaviour: resample the spectrum n_iter times.
    shift
            Shift the continuum/background anchors n_iter times by drawing each anchor
            from a Gaussian with width ``continuum_shift`` (in 10^3 km/s).
    shift_resample
            For each iteration, do both: resample the spectrum and shift the anchors.
    all
            Run all three modes above and return separate summaries for each, with
            overlaid histograms.

    Returns
    -------
    summary : dict
    fig : matplotlib.figure.Figure
    """

    rng = np.random.default_rng(seed)

    valid_modes = {"resample", "shift", "shift_resample", "all"}
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {sorted(valid_modes)}, got {mode!r}")
    if continuum_shift < 0:
        raise ValueError("continuum_shift must be non-negative")

    means = df_spec["norm_fl"].values
    sigs = df_spec["norm_flerr"].values

    def _make_background(draw_mode: str):
        return common.mc_shift_background(
            rng, background_dict,
            ("sil6355_blue_side", "sil6355_red_side", "sil5972_blue_side", "sil5972_red_side"),
            continuum_shift, draw_mode,
        )

    def _make_flux(draw_mode: str):
        return common.mc_draw_flux(rng, means, sigs, draw_mode)

    def _single_fit(flux_draw, bg_draw):
        df_draw = df_spec.copy()
        df_draw.loc[:, "norm_fl_draw"] = flux_draw

        df_spec_for_back_sil1 = df_draw.loc[
            (
                (df_draw["vel_sil1_1"] >= bg_draw["sil6355_blue_side"] - vel_width / 2.0)
                & (df_draw["vel_sil1_1"] <= bg_draw["sil6355_blue_side"] + vel_width / 2.0)
            )
            | (
                (df_draw["vel_sil1_1"] >= bg_draw["sil6355_red_side"] - vel_width / 2.0)
                & (df_draw["vel_sil1_1"] <= bg_draw["sil6355_red_side"] + vel_width / 2.0)
            )
        ]
        df_spec_for_back_sil2 = df_draw.loc[
            (
                (df_draw["vel_sil2_1"] >= bg_draw["sil5972_blue_side"] - vel_width / 2.0)
                & (df_draw["vel_sil2_1"] <= bg_draw["sil5972_blue_side"] + vel_width / 2.0)
            )
            | (
                (df_draw["vel_sil2_1"] >= bg_draw["sil5972_red_side"] - vel_width / 2.0)
                & (df_draw["vel_sil2_1"] <= bg_draw["sil5972_red_side"] + vel_width / 2.0)
            )
        ]

        df_spec_for_fit_sil1 = df_draw.loc[
            (df_draw["vel_sil1_1"] >= bg_draw["sil6355_blue_side"])
            & (df_draw["vel_sil1_1"] <= bg_draw["sil6355_red_side"])
        ]
        df_spec_for_fit_sil2 = df_draw.loc[
            (df_draw["vel_sil2_1"] >= bg_draw["sil5972_blue_side"])
            & (df_draw["vel_sil2_1"] <= bg_draw["sil5972_red_side"])
        ]

        if len(df_spec_for_back_sil1) < 2 or len(df_spec_for_back_sil2) < 2:
            raise ValueError("Too few background points after continuum shift")
        if len(df_spec_for_fit_sil1) < max(6, n_resol_neighbors) or len(df_spec_for_fit_sil2) < max(
            6, n_resol_neighbors
        ):
            raise ValueError("Too few fit points after continuum shift")

        wl_back_sil1 = df_spec_for_back_sil1["rest_wl"].values
        y_back_sil1 = df_spec_for_back_sil1["norm_fl_draw"].values
        yerr_back_sil1 = df_spec_for_back_sil1["norm_flerr"].values
        wl_fit_sil1 = df_spec_for_fit_sil1["rest_wl"].values
        wl_back_sil2 = df_spec_for_back_sil2["rest_wl"].values
        y_back_sil2 = df_spec_for_back_sil2["norm_fl_draw"].values
        yerr_back_sil2 = df_spec_for_back_sil2["norm_flerr"].values
        wl_fit_sil2 = df_spec_for_fit_sil2["rest_wl"].values
        resol_wl_sil1 = np.argsort(np.abs(wl_fit_sil1 - rest_wl_sil1_1))[:n_resol_neighbors]
        closest1 = wl_fit_sil1[resol_wl_sil1]
        resol_vel_sil1 = np.abs(
            wl_to_vel(closest1.max(), rest_wl_sil1_1) - wl_to_vel(closest1.min(), rest_wl_sil1_1)
        )

        model_fit_back_sil1 = Model(lin_back)
        pars_back_sil1 = Parameters()
        pars_back_sil1.add("a", value=0.0)
        pars_back_sil1.add("b", value=0.0)
        out_back_sil1 = model_fit_back_sil1.fit(
            y_back_sil1, pars_back_sil1, x=wl_back_sil1, weights=1.0 / yerr_back_sil1
        )
        back_sil1 = lin_back(
            wl_fit_sil1, out_back_sil1.params["a"].value, out_back_sil1.params["b"].value
        )
        if np.any(~np.isfinite(back_sil1)) or np.any(np.abs(back_sil1) < 1e-12):
            raise ValueError("Si II 6355 continuum fit is zero or non-finite")
        y_fit_sil1 = (df_spec_for_fit_sil1["norm_fl_draw"].values / back_sil1) - 1.0
        yerr_fit_sil1 = df_spec_for_fit_sil1["norm_flerr"].values / back_sil1

        model_fit_back_sil2 = Model(lin_back)
        pars_back_sil2 = Parameters()
        pars_back_sil2.add("a", value=0.0)
        pars_back_sil2.add("b", value=0.0)
        out_back_sil2 = model_fit_back_sil2.fit(
            y_back_sil2, pars_back_sil2, x=wl_back_sil2, weights=1.0 / yerr_back_sil2
        )
        back_sil2 = lin_back(
            wl_fit_sil2, out_back_sil2.params["a"].value, out_back_sil2.params["b"].value
        )
        if np.any(~np.isfinite(back_sil2)) or np.any(np.abs(back_sil2) < 1e-12):
            raise ValueError("Si II 5972 continuum fit is zero or non-finite")
        y_fit_sil2 = (df_spec_for_fit_sil2["norm_fl_draw"].values / back_sil2) - 1.0
        yerr_fit_sil2 = df_spec_for_fit_sil2["norm_flerr"].values / back_sil2

        pars_fit_sil = Parameters()
        pars_fit_sil.add("a_sil1_1", value=-1.1, max=0.0)
        pars_fit_sil.add("v_sil1_1", value=-11.0, max=0.0)
        pars_fit_sil.add("fwhm_sil1_1", value=8.0, min=resol_vel_sil1)
        pars_fit_sil.add("a_sil1_2", expr="a_sil1_1")
        pars_fit_sil.add("v_sil1_2", expr="v_sil1_1")
        pars_fit_sil.add("fwhm_sil1_2", expr="fwhm_sil1_1")
        pars_fit_sil.add("k_vel12", value=1.0, min=k_vel12_bounds[0], max=k_vel12_bounds[1])
        pars_fit_sil.add("k_fwhm12", value=1.0, min=k_fwhm12_bounds[0], max=k_fwhm12_bounds[1])
        pars_fit_sil.add("a_sil2_1", value=-1.0, max=0.0)
        pars_fit_sil.add("v_sil2_1", expr="k_vel12 * v_sil1_1")
        pars_fit_sil.add("fwhm_sil2_1", expr="k_fwhm12 * fwhm_sil1_1")
        pars_fit_sil.add("a_sil2_2", expr="a_sil2_1")
        pars_fit_sil.add("v_sil2_2", expr="v_sil2_1")
        pars_fit_sil.add("fwhm_sil2_2", expr="fwhm_sil2_1")

        def model_sil1(params, wl):
            return g_sil1_1(
                wl, params["a_sil1_1"], params["v_sil1_1"], params["fwhm_sil1_1"]
            ) + g_sil1_2(wl, params["a_sil1_2"], params["v_sil1_2"], params["fwhm_sil1_2"])

        def model_sil2(params, wl):
            return g_sil2_1(
                wl, params["a_sil2_1"], params["v_sil2_1"], params["fwhm_sil2_1"]
            ) + g_sil2_2(wl, params["a_sil2_2"], params["v_sil2_2"], params["fwhm_sil2_2"])

        def resid(params):
            m1 = model_sil1(params, wl_fit_sil1)
            m2 = model_sil2(params, wl_fit_sil2)
            r1 = (y_fit_sil1 - m1) / yerr_fit_sil1
            r2 = (y_fit_sil2 - m2) / yerr_fit_sil2
            return np.r_[r1, r2]

        out = minimize(resid, pars_fit_sil, method="least_squares")
        if not out.success:
            raise RuntimeError(f"Joint Si II optimizer failed: {out.message}")

        wl_dense_1 = np.linspace(wl_fit_sil1[0], wl_fit_sil1[-1], 300)
        wl_dense_2 = np.linspace(wl_fit_sil2[0], wl_fit_sil2[-1], 300)
        s1_1 = g_sil1_1(
            wl_dense_1,
            out.params["a_sil1_1"].value,
            out.params["v_sil1_1"].value,
            out.params["fwhm_sil1_1"].value,
        )
        s1_2 = g_sil1_2(
            wl_dense_1,
            out.params["a_sil1_2"].value,
            out.params["v_sil1_2"].value,
            out.params["fwhm_sil1_2"].value,
        )
        s2_1 = g_sil2_1(
            wl_dense_2,
            out.params["a_sil2_1"].value,
            out.params["v_sil2_1"].value,
            out.params["fwhm_sil2_1"].value,
        )
        s2_2 = g_sil2_2(
            wl_dense_2,
            out.params["a_sil2_2"].value,
            out.params["v_sil2_2"].value,
            out.params["fwhm_sil2_2"].value,
        )

        ew1 = np.abs(trapezoid(s1_1 + s1_2, x=wl_dense_1))
        ew2 = np.abs(trapezoid(s2_1 + s2_2, x=wl_dense_2))
        ew1_raw = np.abs(trapezoid(y_fit_sil1, x=wl_fit_sil1))
        ew2_raw = np.abs(trapezoid(y_fit_sil2, x=wl_fit_sil2))

        return dict(
            sil6355_vel=out.params["v_sil1_1"].value,
            sil5972_vel=out.params["v_sil2_1"].value,
            sil6355_fwhm=out.params["fwhm_sil1_1"].value,
            sil5972_fwhm=out.params["fwhm_sil2_1"].value,
            sil6355_ew=ew1,
            sil5972_ew=ew2,
            sil6355_ew_raw=ew1_raw,
            sil5972_ew_raw=ew2_raw,
            r_sil=np.nan if ew1 == 0 else ew2 / ew1,
            r_sil_raw=np.nan if ew1_raw == 0 else ew2_raw / ew1_raw,
        )

    metric_names = [
        "sil6355_vel",
        "sil5972_vel",
        "sil6355_fwhm",
        "sil5972_fwhm",
        "sil6355_ew",
        "sil5972_ew",
        "sil6355_ew_raw",
        "sil5972_ew_raw",
        "r_sil",
        "r_sil_raw",
    ]


    mode_suffix = {
        "resample": "resample",
        "shift": "shift",
        "shift_resample": "shift_resample",
    }

    requested_modes = ["resample", "shift", "shift_resample"] if mode == "all" else [mode]
    all_results, all_summaries, failed_draws, failure_reasons = common.collect_mc_draws(
        metric_names, requested_modes, n_iter,
        lambda draw_mode: _single_fit(_make_flux(draw_mode), _make_background(draw_mode)),
    )

    summary = {
        "mc_mode": mode,
        "mc_n_iter": int(n_iter),
        "mc_continuum_shift": float(continuum_shift),
    }
    for draw_mode in requested_modes:
        suffix = mode_suffix[draw_mode]
        summary[f"n_success_{suffix}"] = int(all_summaries[draw_mode]["sil6355_vel"]["n"])
        summary[f"n_fail_{suffix}"] = int(failed_draws[draw_mode])
        summary[f"failure_reasons_{suffix}"] = " | ".join(
            f"{count}x {reason}" for reason, count in failure_reasons[draw_mode].most_common()
        )
        for metric_name, stats_dict in all_summaries[draw_mode].items():
            summary[f"{metric_name}_med_{suffix}"] = stats_dict["med"]
            summary[f"{metric_name}_hi_{suffix}"] = stats_dict["hi"]
            summary[f"{metric_name}_lo_{suffix}"] = stats_dict["lo"]
            summary[f"{metric_name}_mean_{suffix}"] = stats_dict["mean"]
            summary[f"{metric_name}_std_{suffix}"] = stats_dict["std"]
            summary[f"{metric_name}_gap_{suffix}"] = stats_dict["gap"]
            summary[f"{metric_name}_flag_{suffix}"] = stats_dict["flag"]

    label_map = {
        "resample": "resample",
        "shift": "shift",
        "shift_resample": "shift+resample",
    }

    plot_metrics = [
        ("sil6355_vel", "Si II λ6355 velocity"),
        ("sil5972_vel", "Si II λ5972 velocity"),
        ("sil6355_fwhm", "Si II λ6355 FWHM"),
        ("sil5972_fwhm", "Si II λ5972 FWHM"),
        ("sil6355_ew", "Si II λ6355 EW"),
        ("sil5972_ew", "Si II λ5972 EW"),
        ("sil6355_ew_raw", "Si II λ6355 EW raw"),
        ("sil5972_ew_raw", "Si II λ5972 EW raw"),
        ("r_sil", "R_sil"),
        ("r_sil_raw", "R_sil raw"),
    ]

    fig, axs = plt.subplots(ncols=2, nrows=5, figsize=(8, 8))
    for ax, (metric_name, title) in zip(axs.flatten(), plot_metrics):
        arrays = [
            all_results[m][metric_name]
            for m in requested_modes
            if all_results[m][metric_name].size > 0
        ]
        if not arrays:
            ax.text(
                0.5,
                0.5,
                "No successful\nMC draws",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="tab:red",
            )
        if arrays:
            combined = np.concatenate(arrays)
            bins = _mc_plot_bin_edges(combined)
        else:
            bins = 20

        for draw_mode in requested_modes:
            arr = all_results[draw_mode][metric_name]
            if arr.size == 0:
                continue

            med = all_summaries[draw_mode][metric_name]["med"]

            n, b, patches = ax.hist(
                arr[np.isfinite(arr)], bins=bins, alpha=0.33, label=label_map[draw_mode]
            )

            if len(patches) > 0:
                fc = patches[0].get_facecolor()
                line_color = (fc[0], fc[1], fc[2], 1.0)
            else:
                line_color = None

            ax.axvline(med, color=line_color, linestyle="--", linewidth=2.2, zorder=3)

        if arrays:
            all_concat = np.concatenate(arrays)
            all_concat = all_concat[np.isfinite(all_concat)]
            if all_concat.size > 0:
                p1, p99 = np.nanpercentile(all_concat, [1, 99])
                if np.isfinite(p1) and np.isfinite(p99) and p99 > p1:
                    pad = 0.05 * (p99 - p1)
                    ax.set_xlim(p1 - pad, p99 + pad)

        ax.set_title(title)
        if metric_name == "sil6355_vel" and ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)

    fig.tight_layout()
    plt.close(fig)

    return summary, fig


def preview_background_all(df_spec, background_dict, vel_width: float = 4.0):
    """
    Preview the continuum background for both Si II features (6355 and 5972).

    This shows:
      - raw flattened spectrum
      - smoothed flattened spectrum
      - shaded background windows of width ``vel_width`` (in 10^3 km/s)
      - straight-line continuum fits through the background windows
      - anchor points at the centers of the background windows
    """

    wl_xlim_sil1 = [vel_to_wl(-40, rest_wl_sil1_1), vel_to_wl(20, rest_wl_sil1_1)]
    wl_xlim_sil2 = [vel_to_wl(-40, rest_wl_sil2_1), vel_to_wl(20, rest_wl_sil2_1)]

    wl = df_spec["rest_wl"].values
    y_raw = df_spec["norm_fl_flattened"].values
    y_smooth = df_spec["norm_fl_smooth_flattened"].values

    df_spec_for_back_sil1 = df_spec.loc[
        (
            (df_spec["vel_sil1_1"] >= background_dict["sil6355_blue_side"] - vel_width / 2.0)
            & (df_spec["vel_sil1_1"] <= background_dict["sil6355_blue_side"] + vel_width / 2.0)
        )
        | (
            (df_spec["vel_sil1_1"] >= background_dict["sil6355_red_side"] - vel_width / 2.0)
            & (df_spec["vel_sil1_1"] <= background_dict["sil6355_red_side"] + vel_width / 2.0)
        )
    ]
    df_spec_for_back_sil2 = df_spec.loc[
        (
            (df_spec["vel_sil2_1"] >= background_dict["sil5972_blue_side"] - vel_width / 2.0)
            & (df_spec["vel_sil2_1"] <= background_dict["sil5972_blue_side"] + vel_width / 2.0)
        )
        | (
            (df_spec["vel_sil2_1"] >= background_dict["sil5972_red_side"] - vel_width / 2.0)
            & (df_spec["vel_sil2_1"] <= background_dict["sil5972_red_side"] + vel_width / 2.0)
        )
    ]

    fig, (ax_sil1, ax_sil2) = plt.subplots(ncols=1, nrows=2, figsize=(10, 8))

    ax_sil1.plot(wl, y_raw, lw=1.0, color="tab:grey", label="raw", zorder=1)
    ax_sil1.plot(wl, y_smooth, lw=1.0, color="k", label="smoothed", zorder=1)

    v_red = background_dict["sil6355_red_side"]
    v_blue = background_dict["sil6355_blue_side"]

    w_red = vel_to_wl(v_red, rest_wl_sil1_1)
    w_blue = vel_to_wl(v_blue, rest_wl_sil1_1)

    for v_center, color, label in [
        (v_red, "tab:cyan", "background window"),
        (v_blue, "tab:cyan", None),
    ]:
        v_left = v_center - vel_width / 2.0
        v_right = v_center + vel_width / 2.0
        wl_left = vel_to_wl(v_left, rest_wl_sil1_1)
        wl_right = vel_to_wl(v_right, rest_wl_sil1_1)
        ax_sil1.axvspan(wl_left, wl_right, alpha=0.25, color=color, label=label, zorder=0)

    ax_sil1.plot(
        vel_to_wl(df_spec_for_back_sil1["vel_sil1_1"], rest_wl_sil1_1),
        df_spec_for_back_sil1["norm_fl_flattened"],
        "o",
        markerfacecolor="none",
        markeredgecolor="k",
        zorder=1,
        mew=2.0,
    )

    v = df_spec["vel_sil1_1"].values
    mask_back = ((v >= v_red - vel_width / 2.0) & (v <= v_red + vel_width / 2.0)) | (
        (v >= v_blue - vel_width / 2.0) & (v <= v_blue + vel_width / 2.0)
    )
    wl_back = wl[mask_back]
    y_back = y_raw[mask_back]
    if len(wl_back) >= 2:
        a, b = np.polyfit(wl_back, y_back, 1)
    elif len(wl_back) == 1:
        a, b = 0.0, float(y_back[0])
    else:
        a, b = 0.0, 1.0
    w_line = np.linspace(min(w_red, w_blue), max(w_red, w_blue), 200)
    y_line = a * w_line + b
    ax_sil1.plot(w_line, y_line, "--", lw=1.2, color="tab:red", label="continuum fit", zorder=2)

    ax_sil1.plot(
        [w_red, w_blue],
        [a * w_red + b, a * w_blue + b],
        "x",
        color="tab:orange",
        markersize=10.0,
        markeredgewidth=2.0,
        label="background anchors",
        zorder=2,
    )

    ax_sil1.set_ylabel("norm. flux")
    ax_sil1.set_xlim(*wl_xlim_sil1)
    autoscale_ylim_for_xlim(wl_xlim_sil1, margin=0.1, ax=ax_sil1)
    ax_sil1.set_title(f"Si II 6355 background (vel_width={vel_width:.1f}×10³ km/s)")

    ##

    ax_sil2.plot(wl, y_raw, lw=1.0, color="tab:grey", label="raw", zorder=1)
    ax_sil2.plot(wl, y_smooth, lw=1.0, color="k", label="smoothed", zorder=1)

    v_red = background_dict["sil5972_red_side"]
    v_blue = background_dict["sil5972_blue_side"]

    w_red = vel_to_wl(v_red, rest_wl_sil2_1)
    w_blue = vel_to_wl(v_blue, rest_wl_sil2_1)

    for v_center, color, label in [
        (v_red, "tab:cyan", "background window"),
        (v_blue, "tab:cyan", None),
    ]:
        v_left = v_center - vel_width / 2.0
        v_right = v_center + vel_width / 2.0
        wl_left = vel_to_wl(v_left, rest_wl_sil2_1)
        wl_right = vel_to_wl(v_right, rest_wl_sil2_1)
        ax_sil2.axvspan(wl_left, wl_right, alpha=0.25, color=color, label=label, zorder=0)

    ax_sil2.plot(
        vel_to_wl(df_spec_for_back_sil2["vel_sil2_1"], rest_wl_sil2_1),
        df_spec_for_back_sil2["norm_fl_flattened"],
        "o",
        markerfacecolor="none",
        markeredgecolor="k",
        zorder=1,
        mew=2.0,
    )

    v = df_spec["vel_sil2_1"].values
    mask_back = ((v >= v_red - vel_width / 2.0) & (v <= v_red + vel_width / 2.0)) | (
        (v >= v_blue - vel_width / 2.0) & (v <= v_blue + vel_width / 2.0)
    )
    wl_back = wl[mask_back]
    y_back = y_raw[mask_back]
    if len(wl_back) >= 2:
        a, b = np.polyfit(wl_back, y_back, 1)
    elif len(wl_back) == 1:
        a, b = 0.0, float(y_back[0])
    else:
        a, b = 0.0, 1.0
    w_line = np.linspace(min(w_red, w_blue), max(w_red, w_blue), 200)
    y_line = a * w_line + b
    ax_sil2.plot(w_line, y_line, "--", lw=1.2, color="tab:red", label="continuum fit", zorder=2)

    ax_sil2.plot(
        [w_red, w_blue],
        [a * w_red + b, a * w_blue + b],
        "x",
        color="tab:orange",
        markersize=10.0,
        markeredgewidth=2.0,
        label="background anchors",
        zorder=2,
    )

    ax_sil2.set_ylabel("norm. flux")
    ax_sil2.set_xlim(*wl_xlim_sil2)
    autoscale_ylim_for_xlim(wl_xlim_sil2, margin=0.1, ax=ax_sil2)
    ax_sil2.set_title(f"Si II 5972 background (vel_width={vel_width:.1f}×10³ km/s)")

    ax_sil2.set_xlabel("Rest wavelength (Å)")

    handles, labels = ax_sil2.get_legend_handles_labels()
    if handles:
        ax_sil2.legend(loc="best", ncol=2)

    fig.tight_layout()
    return fig


def preview_silicon_fit(
    df_spec, background_dict, back_sil1, back_sil2, fit_sil, vel_width: float = 4.0
):

    df_spec_for_back_sil1 = df_spec.loc[
        (
            (df_spec["vel_sil1_1"] >= background_dict["sil6355_blue_side"] - vel_width / 2.0)
            & (df_spec["vel_sil1_1"] <= background_dict["sil6355_blue_side"] + vel_width / 2.0)
        )
        | (
            (df_spec["vel_sil1_1"] >= background_dict["sil6355_red_side"] - vel_width / 2.0)
            & (df_spec["vel_sil1_1"] <= background_dict["sil6355_red_side"] + vel_width / 2.0)
        )
    ]
    df_spec_for_back_sil2 = df_spec.loc[
        (
            (df_spec["vel_sil2_1"] >= background_dict["sil5972_blue_side"] - vel_width / 2.0)
            & (df_spec["vel_sil2_1"] <= background_dict["sil5972_blue_side"] + vel_width / 2.0)
        )
        | (
            (df_spec["vel_sil2_1"] >= background_dict["sil5972_red_side"] - vel_width / 2.0)
            & (df_spec["vel_sil2_1"] <= background_dict["sil5972_red_side"] + vel_width / 2.0)
        )
    ]

    df_spec_for_fit_sil1 = df_spec.loc[
        (df_spec["vel_sil1_1"] >= background_dict["sil6355_blue_side"])
        & (df_spec["vel_sil1_1"] <= background_dict["sil6355_red_side"])
    ]

    vel_fit_sil1 = df_spec_for_fit_sil1["vel_sil1_1"].values

    back_sil1_plot = lin_back(
        df_spec["rest_wl"], back_sil1.params["a"].value, back_sil1.params["b"].value
    )
    back_sil1_plot2 = lin_back(
        df_spec_for_back_sil1["rest_wl"], back_sil1.params["a"].value, back_sil1.params["b"].value
    )

    vel_plot = np.linspace(vel_fit_sil1[0], vel_fit_sil1[-1], 300)
    wl_plot = vel_to_wl(vel_plot, rest_wl_sil1_1)
    sil1_1_plot = g_sil1_1(
        wl_plot,
        fit_sil.params["a_sil1_1"].value,
        fit_sil.params["v_sil1_1"].value,
        fit_sil.params["fwhm_sil1_1"].value,
    )
    sil1_2_plot = g_sil1_2(
        wl_plot,
        fit_sil.params["a_sil1_2"].value,
        fit_sil.params["v_sil1_2"].value,
        fit_sil.params["fwhm_sil1_2"].value,
    )

    fig, (ax_sil1_backnorm, ax_sil2_backnorm) = plt.subplots(
        ncols=1, nrows=2, sharex=True, figsize=(10, 8)
    )

    ax_sil1_backnorm.plot(
        df_spec["vel_sil1_1"],
        (df_spec["norm_fl"].values / back_sil1_plot) - 1.0,
        color="tab:grey",
        linewidth=2,
        zorder=1,
    )
    ax_sil1_backnorm.plot(vel_plot, (sil1_1_plot), color="tab:green", zorder=2)
    ax_sil1_backnorm.plot(vel_plot, (sil1_2_plot), color="tab:blue", zorder=2)
    ax_sil1_backnorm.plot(vel_plot, (sil1_1_plot + sil1_2_plot), color="tab:orange", zorder=2)

    autoscale_ylim_for_xlim((-50, 20), margin=0.05, ax=ax_sil1_backnorm)

    ax_sil1_backnorm.axvspan(
        background_dict["sil6355_blue_side"] - vel_width / 2.0,
        background_dict["sil6355_blue_side"] + vel_width / 2.0,
        alpha=0.25,
        color="tab:cyan",
        zorder=0,
    )
    ax_sil1_backnorm.axvspan(
        background_dict["sil6355_red_side"] - vel_width / 2.0,
        background_dict["sil6355_red_side"] + vel_width / 2.0,
        alpha=0.25,
        color="tab:cyan",
        zorder=0,
    )

    ax_sil1_backnorm.plot(
        df_spec_for_back_sil1["vel_sil1_1"],
        (df_spec_for_back_sil1["norm_fl"].values / back_sil1_plot2) - 1.0,
        "o",
        markerfacecolor="none",
        markeredgecolor="k",
        zorder=1,
        mew=2.0,
    )

    ax_sil1_backnorm.axhline(y=0, color="tab:red", linestyle="--", zorder=1)

    ax_sil1_backnorm.set_xlabel("velocity (10³ km/s)")

    ##

    df_spec_for_fit_sil2 = df_spec.loc[
        (df_spec["vel_sil2_1"] >= background_dict["sil5972_blue_side"])
        & (df_spec["vel_sil2_1"] <= background_dict["sil5972_red_side"])
    ]

    vel_fit_sil2 = df_spec_for_fit_sil2["vel_sil2_1"].values

    back_sil2_plot = lin_back(
        df_spec["rest_wl"], back_sil2.params["a"].value, back_sil2.params["b"].value
    )
    back_sil2_plot2 = lin_back(
        df_spec_for_back_sil2["rest_wl"], back_sil2.params["a"].value, back_sil2.params["b"].value
    )

    vel_plot = np.linspace(vel_fit_sil2[0], vel_fit_sil2[-1], 300)
    wl_plot = vel_to_wl(vel_plot, rest_wl_sil2_1)
    sil2_1_plot = g_sil2_1(
        wl_plot,
        fit_sil.params["a_sil2_1"].value,
        fit_sil.params["v_sil2_1"].value,
        fit_sil.params["fwhm_sil2_1"].value,
    )
    sil2_2_plot = g_sil2_2(
        wl_plot,
        fit_sil.params["a_sil2_2"].value,
        fit_sil.params["v_sil2_2"].value,
        fit_sil.params["fwhm_sil2_2"].value,
    )

    ax_sil2_backnorm.plot(
        df_spec["vel_sil2_1"],
        (df_spec["norm_fl"].values / back_sil2_plot) - 1.0,
        color="tab:grey",
        linewidth=2,
        zorder=1,
    )
    ax_sil2_backnorm.plot(vel_plot, (sil2_1_plot), color="tab:green", zorder=2)
    ax_sil2_backnorm.plot(vel_plot, (sil2_2_plot), color="tab:blue", zorder=2)
    ax_sil2_backnorm.plot(vel_plot, (sil2_1_plot + sil2_2_plot), color="tab:orange", zorder=2)

    autoscale_ylim_for_xlim((-50, 20), margin=0.05, ax=ax_sil2_backnorm)

    ax_sil2_backnorm.axvspan(
        background_dict["sil5972_blue_side"] - vel_width / 2.0,
        background_dict["sil5972_blue_side"] + vel_width / 2.0,
        alpha=0.25,
        color="tab:cyan",
        zorder=0,
    )
    ax_sil2_backnorm.axvspan(
        background_dict["sil5972_red_side"] - vel_width / 2.0,
        background_dict["sil5972_red_side"] + vel_width / 2.0,
        alpha=0.25,
        color="tab:cyan",
        zorder=0,
    )

    ax_sil2_backnorm.plot(
        df_spec_for_back_sil2["vel_sil2_1"],
        (df_spec_for_back_sil2["norm_fl"].values / back_sil2_plot2) - 1.0,
        "o",
        markerfacecolor="none",
        markeredgecolor="k",
        zorder=1,
        mew=2.0,
    )

    ax_sil2_backnorm.axhline(y=0, color="tab:red", linestyle="--", zorder=1)

    ax_sil2_backnorm.set_xlabel("velocity (10³ km/s)")

    handles, labels = ax_sil2_backnorm.get_legend_handles_labels()
    if handles:
        ax_sil2_backnorm.legend(loc="best")

    fig.tight_layout()
    return fig


def _require_interactive_backend() -> None:
    backend = str(plt.get_backend()).lower()
    non_interactive = {"agg", "pdf", "pgf", "ps", "svg", "template"}
    if backend in non_interactive or "inline" in backend:
        raise RuntimeError(
            f"Interactive mode requires a Matplotlib GUI backend; current backend is {backend!r}"
        )


def select_background_manually(
    df_spec,
    current_background=None,
    vel_width: float = 4.0,
    wl_xlim=(5250.0, 6750.0),
):
    """Select the four Si II continuum anchors with mouse clicks.

    The click order is 6355 red, 6355 blue, 5972 red, and 5972 blue. Existing
    anchors are shown as shaded regions when ``current_background`` is supplied.
    """
    _require_interactive_backend()
    labels = (
        ("Si II 6355: click red-side continuum", rest_wl_sil1_1, "tab:red"),
        ("Si II 6355: click blue-side continuum", rest_wl_sil1_1, "tab:blue"),
        ("Si II 5972: click red-side continuum", rest_wl_sil2_1, "tab:red"),
        ("Si II 5972: click blue-side continuum", rest_wl_sil2_1, "tab:blue"),
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(df_spec["rest_wl"], df_spec["norm_fl"], color="tab:grey", lw=1, label="raw")
    ax.plot(df_spec["rest_wl"], df_spec["norm_fl_smooth"], color="k", lw=1, label="smoothed")

    if current_background:
        anchors = (
            ("sil6355_red_side", rest_wl_sil1_1),
            ("sil6355_blue_side", rest_wl_sil1_1),
            ("sil5972_red_side", rest_wl_sil2_1),
            ("sil5972_blue_side", rest_wl_sil2_1),
        )
        for index, (key, rest_wavelength) in enumerate(anchors):
            center = current_background[key]
            left = vel_to_wl(center - vel_width / 2.0, rest_wavelength)
            right = vel_to_wl(center + vel_width / 2.0, rest_wavelength)
            ax.axvspan(
                left,
                right,
                alpha=0.18,
                color="tab:cyan",
                label="current anchors" if index == 0 else None,
                zorder=0,
            )

    ax.axvline(vel_to_wl(-11, rest_wl_sil1_1), ls=":", color="tab:orange")
    ax.axvline(vel_to_wl(-11, rest_wl_sil2_1), ls=":", color="tab:orange")
    ax.set_xlim(*wl_xlim)
    autoscale_ylim_for_xlim(wl_xlim, margin=0.05, ax=ax)
    ax.set_xlabel("Rest wavelength (Å)")
    ax.set_ylabel("Normalized flux")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.show()
    plt.pause(0.001)

    clicks = []
    try:
        for message, _, color in labels:
            for artist in list(ax.texts):
                artist.remove()
            ax.text(
                0.04,
                0.96,
                message,
                transform=ax.transAxes,
                va="top",
                fontsize=14,
                color=color,
                fontweight="bold",
            )
            fig.canvas.draw_idle()
            selected = fig.ginput(1, timeout=-1, show_clicks=True)
            if not selected:
                raise RuntimeError("Manual background selection was cancelled")
            clicks.append(float(selected[0][0]))
    finally:
        plt.close(fig)

    background = dict(current_background or {})
    background.update(
        {
            "sil6355_red_side": wl_to_vel(clicks[0], rest_wl_sil1_1),
            "sil6355_blue_side": wl_to_vel(clicks[1], rest_wl_sil1_1),
            "sil5972_red_side": wl_to_vel(clicks[2], rest_wl_sil2_1),
            "sil5972_blue_side": wl_to_vel(clicks[3], rest_wl_sil2_1),
            "status_sil6355_red_side": "manual",
            "status_sil6355_blue_side": "manual",
            "status_sil5972_red_side": "manual",
            "status_sil5972_blue_side": "manual",
        }
    )
    if background["sil6355_blue_side"] >= background["sil6355_red_side"]:
        raise ValueError("Si II 6355 blue anchor must be bluer than its red anchor")
    if background["sil5972_blue_side"] >= background["sil5972_red_side"]:
        raise ValueError("Si II 5972 blue anchor must be bluer than its red anchor")
    return background


def interactive_refine_background_and_fit(
    df_spec,
    initial_background_dict,
    vel_width: float = 4.0,
    wl_xlim=(5250.0, 6750.0),
    n_resol_neighbors: int = 3,
    k_vel12_bounds: tuple[float, float] = (0.75, 1.25),
    k_fwhm12_bounds: tuple[float, float] = (0.25, 1.75),
    manual_start: bool = False,
):
    """Review continuum anchors and a joint fit with only essential prompts.

    Enter accepts a displayed background or fit. Entering a positive number at
    the background prompt changes the continuum-window width. Entering ``n``
    opens the four-click manual selector. When ``manual_start`` is true, the
    selector opens immediately.

    Returns
    -------
    background, background_figure, fit_metrics, fit_figure, vel_width
    """
    _require_interactive_backend()
    background = dict(initial_background_dict)
    was_interactive = plt.isinteractive()
    plt.ion()

    try:
        while True:
            if manual_start:
                try:
                    background = select_background_manually(
                        df_spec, background, vel_width=vel_width, wl_xlim=wl_xlim
                    )
                except ValueError as exc:
                    print(f"Invalid anchor selection: {exc}. Please try again.")
                    continue
                manual_start = False

            fig_bg = preview_background_all(df_spec, background, vel_width=vel_width)
            try:
                fig_bg.canvas.manager.set_window_title("Si II background preview")
            except Exception:
                pass
            fig_bg.show()
            fig_bg.canvas.draw_idle()
            plt.pause(0.001)

            raw = input("Accept backgrounds? [Y/n] (or enter window width in 10^3 km/s): ").strip()
            answer = raw.lower()

            if answer not in ("", "y", "yes"):
                plt.close(fig_bg)
                try:
                    new_width = float(raw)
                except ValueError:
                    manual_start = True
                    continue
                if new_width <= 0:
                    print("Window width must be positive.")
                    continue
                vel_width = new_width
                continue

            plt.close(fig_bg)
            metrics, fit_fig, back_sil1, back_sil2, fit_sil = fit_features(
                df_spec,
                background,
                vel_width=vel_width,
                n_resol_neighbors=n_resol_neighbors,
                k_vel12_bounds=k_vel12_bounds,
                k_fwhm12_bounds=k_fwhm12_bounds,
            )
            preview = preview_silicon_fit(
                df_spec, background, back_sil1, back_sil2, fit_sil, vel_width=vel_width
            )
            try:
                preview.canvas.manager.set_window_title("Si II joint fit")
            except Exception:
                pass
            preview.show()
            preview.canvas.draw_idle()
            plt.pause(0.001)

            if input("Accept fit? [Y/n]: ").strip().lower() in ("", "y", "yes"):
                plt.close(preview)
                background_fig = preview_background_all(df_spec, background, vel_width=vel_width)
                return background, background_fig, metrics, fit_fig, vel_width

            plt.close(preview)
            plt.close(fit_fig)
            manual_start = True
    finally:
        plt.interactive(was_interactive)

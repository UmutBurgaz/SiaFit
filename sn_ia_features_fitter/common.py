"""Shared spectrum preparation, numerical helpers, and plotting utilities."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PyAstronomy import pyasl
from scipy.interpolate import make_interp_spline
from scipy.optimize import fsolve

from .style import apply_plot_style

apply_plot_style()

speed_of_light = 2.99792e5  # km/s


def mc_draw_flux(rng, means, uncertainties, mode):
    """Draw a flux realization using the shared Monte Carlo mode semantics."""
    if mode in {"resample", "shift_resample"}:
        return rng.normal(loc=means, scale=uncertainties)
    return means.copy()


def mc_shift_background(rng, background, keys, shift, mode):
    """Draw continuum-anchor positions in the supplied stable key order."""
    drawn = dict(background)
    if mode in {"shift", "shift_resample"}:
        for key in keys:
            drawn[key] = rng.normal(background[key], shift)
    return drawn


def mc_summarize(values):
    """Summarize every finite successful draw for one fitted measurement."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return dict(mean=np.nan, std=np.nan, med=np.nan, lo=np.nan, hi=np.nan, gap=np.nan, flag=1, n=0)
    p16, p50, p84 = np.percentile(arr, [16, 50, 84])
    gap, flag = qc_mean_median_gap(arr)
    return dict(
        mean=np.nanmean(arr), std=np.nanstd(arr), med=p50,
        lo=p50-p16, hi=p84-p50, gap=gap, flag=flag, n=arr.size,
    )


def collect_mc_draws(metric_names, modes, n_iter, draw):
    """Run model-specific draws with shared accounting and finite summaries."""
    all_results, all_summaries, failed_draws, failure_reasons = {}, {}, {}, {}
    for mode in modes:
        per_metric = {name: [] for name in metric_names}
        failures = 0
        reasons = Counter()
        for _ in range(n_iter):
            try:
                result = draw(mode)
                for name in metric_names:
                    per_metric[name].append(result[name])
            except Exception as exc:
                failures += 1
                reasons[f"{type(exc).__name__}: {exc}"[:240]] += 1
        all_results[mode] = {
            name: np.asarray(values, dtype=float) for name, values in per_metric.items()
        }
        all_summaries[mode] = {
            name: mc_summarize(values) for name, values in all_results[mode].items()
        }
        failed_draws[mode] = failures
        failure_reasons[mode] = reasons
    return all_results, all_summaries, failed_draws, failure_reasons


def calc_optimal_smooth(wave, flux, variance=None):
    """Calculate an automatic smoothing scale (vexp) from the spectrum S/N.

    This is adapted from D'Arcy's `find_vexp` and uses a linear relation between
    S/N and vexp calibrated in Siebert et al. (2019). The retained low-S/N
    branch uses vexp=0.0045 below S/N 2.5; the high-S/N branch uses
    vexp=0.001 above S/N 80.

    Parameters
    ----------
    wave : array_like
            Wavelength array (Å).
    flux : array_like
            Flux array.
    variance : array_like, optional
            Flux variance array. If provided, it is used to estimate the noise.
            If None, noise is estimated from residuals to a lightly smoothed spectrum.

    Returns
    -------
    vexp_auto : float
            Recommended smoothing width as a fractional Gaussian sigma, σ(λ)=vexp×λ.
    snr : float
            Robust (median) signal-to-noise estimate.
    """

    coeff_0, coeff_1 = -4.51612903e-05, 4.61290323e-03
    # --------- #
    smoothed_spec = smooth_spec(wave, flux, fl_var=variance, vexp=0.002)
    # --- #
    if variance is None:
        error = np.absolute(flux - smoothed_spec)
        smooth_err = smooth_spec(wave, error, variance, vexp=0.008)
    else:
        smooth_err = np.sqrt(variance)
    SNR = np.nanmedian(smoothed_spec / smooth_err)
    vexp_auto = coeff_0 * SNR + coeff_1
    if SNR < 2.5:
        vexp_auto = 0.0045
    if SNR > 80:
        vexp_auto = 0.001
    return vexp_auto, SNR



def smooth_spec(wl, fl, fl_var=None, vexp=0.01, nsig=5.0):
    """
    Smooth a spectrum (flux vs. wavelength) by a wavelength-dependent Gaussian kernel.

    Parameters
    ----------
    wl : array_like, shape (N,)
            Wavelength array (Å), must be strictly increasing.
    fl : array_like, shape (N,)
            Flux array corresponding to `wl`.
    fl_var : array_like, shape (N,), optional
            Flux variance at each wavelength. If None, equal weights are used.
    vexp : float, optional
            Fractional Gaussian width: σ(λ) = vexp × λ. Default is 0.01.
    nsig : float, optional
            Number of σ on each side of each λ₀ to include in the smoothing window. Default is 5.0.

    Returns
    -------
    flux_smooth : ndarray, shape (N,)
            Smoothed flux array.

    Raises
    ------
    ValueError
            If:
            - `wl`, `fl`, or `fl_var` are not 1D arrays of the same length.
            - Any element of `fl_var` is non-finite or non-positive.
            - `wl` is not strictly increasing.
            - `vexp` is non-finite or non-positive.
    """

    # Convert inputs to numpy arrays
    wl_arr = np.asarray(wl, dtype=float)
    fl_arr = np.asarray(fl, dtype=float)

    # Only relative variances matter for a weighted average.
    if fl_var is None:
        var_arr = np.ones_like(fl_arr)
    else:
        var_arr = np.asarray(fl_var, dtype=float)

    # Input validation
    if wl_arr.ndim != 1 or fl_arr.ndim != 1 or var_arr.ndim != 1:
        raise ValueError("smooth_spec: `wl`, `fl`, and `fl_var` must be 1D arrays.")
    if wl_arr.shape != fl_arr.shape or wl_arr.shape != var_arr.shape:
        raise ValueError(
            f"smooth_spec: input arrays must have the same shape, "
            f"got wl {wl_arr.shape}, fl {fl_arr.shape}, fl_var {var_arr.shape}."
        )
    if not np.all(np.isfinite(var_arr) & (var_arr > 0)):
        raise ValueError("smooth_spec: all `fl_var` entries must be finite and positive.")
    if not np.isfinite(vexp) or vexp <= 0:
        raise ValueError("smooth_spec: `vexp` must be finite and positive.")

    # Ensure wavelengths are sorted
    if not np.all(np.diff(wl_arr) > 0):
        raise ValueError("smooth_spec: `wl` must be strictly increasing (sorted).")

    N = wl_arr.size
    flux_smooth = np.zeros(N, dtype=float)

    # Loop over each pixel
    for i in range(N):
        wl0 = wl_arr[i]
        sigma = vexp * wl0
        low = wl0 - nsig * sigma
        high = wl0 + nsig * sigma

        j_min = np.searchsorted(wl_arr, low, side="left")
        j_max = np.searchsorted(wl_arr, high, side="right")

        slice_wl = wl_arr[j_min:j_max]
        slice_fl = fl_arr[j_min:j_max]
        slice_var = var_arr[j_min:j_max]

        delta = slice_wl - wl0
        exponent = -0.5 * (delta / sigma) ** 2
        # Normalization cancels in W1/W0. Scaling by the smallest local
        # variance avoids overflow when valid uncertainties are very small.
        W_lambda = np.exp(exponent) * (np.min(slice_var) / slice_var)
        W0 = W_lambda.sum()
        W1 = (W_lambda * slice_fl).sum()

        flux_smooth[i] = (W1 / W0) if W0 != 0 else fl_arr[i]

    return flux_smooth



def clip_outliers(wave_observed, flux, var, vexp=0.01, redshift=0):
    """Clip obvious outliers/cosmic rays by interpolating over flagged regions.

    Adapted from D'Arcy's `clip` routine.

    Notes
    -----
    - Uses a smoothed spectrum to estimate residuals and a smoothed residual
      spectrum as a noise proxy.
    - Avoids aggressively clipping in several strong absorption regions
      (Ca H&K, Na I, K I, selected DIBs) to reduce the chance of removing real
      spectral features.
    - Operates in rest-frame internally (wave_observed / (1+z)), but returns the
      modified *input* arrays `flux` and (optionally) `var` in-place.

    Parameters
    ----------
    wave_observed : array_like
            Observed-frame wavelength array (Å).
    flux : ndarray
            Flux array (modified in-place).
    var : ndarray or None
            Variance array (modified in-place if provided).
    vexp : float, optional
            Fractional smoothing width (σ(λ)=vexp×λ) used for the first smoothing pass.
    redshift : float, optional
            Redshift used to convert observed to rest-frame.

    Returns
    -------
    flux : ndarray
            The clipped/interpolated flux array (same object as input).
    var : ndarray or None
            The clipped/interpolated variance array (same object as input), or None.
    """
    wave = wave_observed / (1 + redshift)
    smooth_flux = smooth_spec(wave, flux, fl_var=var, vexp=vexp)
    diff = flux - smooth_flux
    error = np.absolute(diff)
    smooth_err = smooth_spec(wave, error, fl_var=None, vexp=0.1)

    # --------- #
    # Identify regions to clip.
    # NB Avoid clipping absoption lines (CaH&K, NaI, KI, DIB)
    regions_avoid = np.where(
        ((wave > 3919.0) & (wave < 3949.0))
        | ((wave > 5875.0) & (wave < 5911.0))
        | ((wave > 7650.0) & (wave < 7714.0))
        | ((wave > 5765.0) & (wave < 5795.0))
    )
    tol1, tol2 = 5.0, 10.0

    bad_inds_pos_avoid = np.where(
        (diff[regions_avoid] > 0) & (error[regions_avoid] / smooth_err[regions_avoid] > tol1)
    )
    bad_inds_neg_avoid = np.where(
        (diff[regions_avoid] < 0) & (error[regions_avoid] / smooth_err[regions_avoid] > tol2)
    )

    regions_normal = np.where(
        (wave <= 3919.0)
        | ((wave >= 3949.0) & (wave <= 5765.0))
        | ((wave >= 5795.0) & (wave <= 5875.0))
        | ((wave >= 5911.0) & (wave <= 7650.0))
        | (wave >= 7714.0)
    )
    bad_inds_normal = np.where((error[regions_normal] / smooth_err[regions_normal] > tol1))
    bad_wave_normal = wave[regions_normal][bad_inds_normal]
    bad_wave_pos_avoid = wave[regions_avoid][bad_inds_pos_avoid]
    bad_wave_neg_avoid = wave[regions_avoid][bad_inds_neg_avoid]
    # Find indices for general clipping
    bad_ranges, buff = [], 8
    for i in range(len(bad_wave_normal)):
        if bad_wave_normal[i] + buff < wave[-1] and bad_wave_normal[i] - buff >= wave[0]:
            _ = bad_ranges.append((bad_wave_normal[i] - buff, bad_wave_normal[i] + buff))
    for i in range(len(bad_wave_pos_avoid)):
        _ = bad_ranges.append((bad_wave_pos_avoid[i] - buff, bad_wave_pos_avoid[i] + buff))
    for i in range(len(bad_wave_neg_avoid)):
        _ = bad_ranges.append((bad_wave_neg_avoid[i] - buff, bad_wave_neg_avoid[i] + buff))
    # Actually clip
    for i, wave_val in enumerate(bad_ranges):
        to_clip = np.where((wave > wave_val[0]) & (wave < wave_val[1]))
        # Exclude the edges of the spectrum
        flux[to_clip] = np.interp(
            wave[to_clip],
            [wave_val[0], wave_val[1]],
            [flux[max(0, to_clip[0][0] - 1)], flux[min(flux.size - 1, to_clip[0][-1] + 1)]],
        )
        if var is not None:
            var[to_clip] = np.interp(
                wave[to_clip],
                [wave_val[0], wave_val[1]],
                [var[max(0, to_clip[0][0] - 1)], var[min(flux.size - 1, to_clip[0][-1] + 1)]],
            )
    return flux, var



def spec_create_errors(wl, fl, vexp, nsig=5.0):
    """
    Estimate per-pixel errors by smoothing residuals of a spectrum.

    Parameters
    ----------
    wl : array_like, shape (N,)
            Wavelength array (Å), must be strictly increasing.
    fl : array_like, shape (N,)
            Flux array corresponding to `wl`.
    vexp : float
            Fractional Gaussian width for the first smoothing: σ(λ) = vexp × λ.
    nsig : float, optional
            Number of σ on each side for both smoothing passes. Default is 5.0.

    Returns
    -------
    spec_err : ndarray, shape (N,)
            Estimated error spectrum, obtained by:
            1) smoothing `fl` with `vexp`, computing |fl – smooth| residuals;
            2) smoothing those residuals with vexp=0.015.
            If this gives zero errors on sparsely sampled data, the first
            smoothing pass is widened to resolve neighbouring pixels.

    Raises
    ------
    ValueError
            If inputs are invalid or a positive error cannot be estimated.
    """

    def estimate_at_width(width):
        smoothed_flux = smooth_spec(wl, fl, fl_var=None, vexp=width, nsig=nsig)
        fl_residuals = np.abs(fl - smoothed_flux)
        return smooth_spec(wl, fl_residuals, fl_var=None, vexp=0.015, nsig=nsig)

    spec_err = estimate_at_width(vexp)
    if np.all(np.isfinite(spec_err) & (spec_err > 0)):
        return spec_err

    # A high-S/N spectrum can select a smoothing width narrower than its pixel
    # spacing. Isolated pixels then have zero residual, which is not a valid
    # inverse-variance weight. Widen only the *noise estimate* in that case;
    # the requested width for the spectrum and feature fit stays unchanged.
    wavelength = np.asarray(wl, dtype=float)
    spacing = np.diff(wavelength)
    if not spacing.size or wavelength[0] <= 0:
        raise ValueError("Cannot estimate positive flux uncertainty from this spectrum")
    width = max(float(vexp), float(np.median(spacing) / (2.0 * wavelength[0])))
    for _ in range(5):
        spec_err = estimate_at_width(width)
        if np.all(np.isfinite(spec_err) & (spec_err > 0)):
            return spec_err
        width *= 2.0
    raise ValueError("Cannot estimate positive flux uncertainty from this spectrum")



def autoscale_ylim_for_xlim(xlim, margin=0.05, reverse_y=False, ax=None):
    """
    Set the x‐ and y‐limits of an Axes so that y‐limits
    tightly enclose only the data within the given x‐range,
    plus a little margin, with an option to reverse the y‐axis.
    This is just for plotting.

    Parameters
    ----------
    xlim : tuple of (xmin, xmax)
            The desired x‐axis limits.
    margin : float, default 0.05
            Fractional padding to add above/below the data range.
    reverse_y : bool, default False
            If True, invert the y‐axis (max at bottom, min at top).
    ax : matplotlib.axes.Axes, optional
            The axes to operate on. If None, uses plt.gca().
    """
    if ax is None:
        ax = plt.gca()

    xmin, xmax = xlim
    ax.set_xlim(xmin, xmax)

    ymins, ymaxs = [], []
    for line in ax.get_lines():
        x = np.asarray(line.get_xdata())
        y = np.asarray(line.get_ydata())
        mask = (x >= xmin) & (x <= xmax)
        if mask.any():
            y_visible = y[mask]
            ymins.append(y_visible.min())
            ymaxs.append(y_visible.max())

    if not ymins:
        return  # no data in range, leave ylim unchanged

    y0, y1 = min(ymins), max(ymaxs)
    pad = margin * (y1 - y0)
    lower, upper = y0 - pad, y1 + pad

    # Set y-limits, optionally reversed
    if reverse_y:
        ax.set_ylim(upper, lower)
    else:
        ax.set_ylim(lower, upper)



def _fig_to_rgb(fig):
    """
    Render a Matplotlib Figure to an RGB numpy array robustly across backends/HiDPI.
    """
    # Ensure Agg canvas is attached
    if not isinstance(fig.canvas, FigureCanvas):
        FigureCanvas(fig)

    # Draw so the renderer/buffer are populated
    fig.canvas.draw()

    # Prefer renderer's actual pixel dimensions (more reliable on HiDPI)
    try:
        renderer = fig.canvas.get_renderer()
        w, h = int(renderer.width), int(renderer.height)
    except Exception:
        w, h = fig.canvas.get_width_height()

    # Read RGBA buffer
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)

    # Expected size if there is no HiDPI scale
    expected = w * h * 4

    if buf.size != expected:
        # Try to detect an integer HiDPI scale (e.g. 2x → 4x pixels)
        # scale^2 = buf.size / expected
        ratio = buf.size / float(expected)
        scale = int(round(ratio**0.5))
        if scale >= 1 and (w * h * 4 * scale * scale) == buf.size:
            w *= scale
            h *= scale
        else:
            raise ValueError(
                f"Unexpected buffer size {buf.size} for canvas {w}x{h} (expected {expected})."
            )

    img = buf.reshape(h, w, 4)[..., :3]  # drop alpha → RGB
    return img



def save_summary_figure(
    figs,  # iterable of matplotlib.figure.Figure
    out_path,  # should end with .png
    title: str = "",
    subtitle: str = "",  # <-- new
    dpi: int = 200,
    layout: str = "horizontal",  # "horizontal" (default) or "vertical"
    panel_titles: list[str] | None = None,
    width_per_panel: float = 3.5,  # inches per panel (for horizontal)
    height: float = 5,  # total height in inches (for horizontal)
):
    """
    Compose multiple Matplotlib figures into a single PNG.

    - layout="horizontal": one row, N columns (landscape).
    - layout="vertical": N rows, one column.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    images = [_fig_to_rgb(f) for f in figs]
    for f in figs:
        plt.close(f)

    n = len(images)

    if layout == "horizontal":
        rows, cols = 1, n
        figsize = (width_per_panel * n, height)
    else:
        rows, cols = n, 1
        figsize = (8.5, max(3.5 * n, 6.0))

    final_fig, axes = plt.subplots(rows, cols, figsize=figsize, constrained_layout=True)

    # Normalize axes iterable
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.ravel()
    else:
        axes = axes.flatten()

    for idx, (ax, img) in enumerate(zip(axes, images)):
        ax.imshow(img)
        ax.axis("off")
        if panel_titles and idx < len(panel_titles) and panel_titles[idx]:
            ax.set_title(panel_titles[idx], fontsize=8, pad=6)

    if title:
        final_fig.suptitle(title, fontsize=10, y=0.995)
        if subtitle:
            final_fig.text(0.5, 0.96, subtitle, fontsize=8, ha="center", va="top")

    final_fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(final_fig)



def qc_mean_median_gap(x, thresh=0.5):
    med = np.nanmedian(x)
    mean = np.nanmean(x)
    mad = 1.4826 * np.nanmedian(np.abs(x - med))  # robust sigma
    gap = np.abs(mean - med) / (mad if mad > 0 else 1.0)
    return gap, int(gap > thresh)  # True => suspicious



def vel_to_wl(vel, rest_wl_line):  # velocity in 10^3 km/s
    zmeas = -1.0 + np.sqrt(
        (1.0 + (vel * 10**3 / speed_of_light)) / (1.0 - (vel * 10**3 / speed_of_light))
    )
    wl = (zmeas * rest_wl_line) + rest_wl_line
    return wl



def wl_to_vel(wl, rest_wl_line):  # velocity in 10^3 km/s
    zmeas = (wl - rest_wl_line) / rest_wl_line
    velocity = (((zmeas + 1.0) ** 2 - 1.0) * speed_of_light / (1.0 + (1.0 + zmeas) ** 2)) / 1e3
    return velocity



def velocity_fwhm_to_angstrom(velocity, fwhm, rest_wl_line):
    """Rest-frame Å span between a line's two half-height velocity positions."""
    lower_velocity = float(velocity) - float(fwhm) / 2.0
    upper_velocity = float(velocity) + float(fwhm) / 2.0
    speed_of_light_1e3 = speed_of_light / 1000.0
    if not (-speed_of_light_1e3 < lower_velocity <= upper_velocity < speed_of_light_1e3):
        return float("nan")
    with np.errstate(divide="ignore", invalid="ignore"):
        lower = vel_to_wl(lower_velocity, rest_wl_line)
        upper = vel_to_wl(upper_velocity, rest_wl_line)
    return float(upper - lower)


def lin_back(x, a, b):
    return a * x + b



def _dedup_wavelength_roots(wls, atol=0.25):
    """
    Deduplicate very close wavelength roots (Å) to avoid counting the same
    root multiple times. Keeps the first of any cluster within `atol` Å.
    """
    if not wls:
        return []
    w = np.sort(np.array(wls, float))
    kept = [w[0]]
    for x in w[1:]:
        if abs(x - kept[-1]) > float(atol):
            kept.append(x)
    return kept



def _scan_extrema_in_window(func, d2func, v_center_1e3, rest_wl, v_window_kms, dv_kms):
    """
    Try fsolve from multiple seeds spaced by `dv_kms` within the velocity window
    [v_center_1e3 ± v_window_kms]. Seeds are built in km/s, converted to Å.
    Returns converged roots (Å) inside that window and their second derivatives.
    """
    v0_kms = float(v_center_1e3) * 1e3
    halfwin = abs(float(v_window_kms))
    step = max(100.0, abs(float(dv_kms)))  # never smaller than 100 km/s
    grid_kms = np.arange(v0_kms - halfwin, v0_kms + halfwin + 0.5 * step, step)

    seeds_wl = [float(vel_to_wl(v / 1e3, rest_wl)) for v in grid_kms]

    roots = []
    for guess_wl in seeds_wl:
        try:
            root, info, flag, _ = fsolve(func, guess_wl, full_output=True, maxfev=200)
            root_wl = float(root[0])
            if flag != 1 or not np.isfinite(root_wl) or root_wl <= 0:
                continue
            # fsolve is unbounded: a seed inside the window can converge to an
            # unrelated extremum outside it. Check the root, not just the seed.
            root_velocity_kms = float(wl_to_vel(root_wl, rest_wl)) * 1e3
            if abs(root_velocity_kms - v0_kms) <= halfwin + 1e-8:
                roots.append(root_wl)
        except Exception:
            continue

    roots = _dedup_wavelength_roots(roots)
    d2vals = []
    for r in roots:
        try:
            d2vals.append(float(d2func(r)))
        except Exception:
            d2vals.append(np.nan)
    return roots, np.array(d2vals, float)



def _maxima_candidates_in_window(func, d2func, v_center_1e3, rest_wl, v_window_kms, dv_kms):
    """
    Return all maxima candidates (where d²<0) within the velocity window
    as (wl, v_1e3) pairs, plus a minima_only flag.
    """
    roots, d2vals = _scan_extrema_in_window(
        func, d2func, v_center_1e3, rest_wl, v_window_kms=v_window_kms, dv_kms=dv_kms
    )
    if len(roots) == 0:
        return [], False
    mask_max = (d2vals < 0) & np.isfinite(d2vals)
    if not np.any(mask_max):
        return [], True  # minima only
    wl_max = np.array(roots, float)[mask_max]
    v_max = np.array([wl_to_vel(w, rest_wl) for w in wl_max], float)
    return list(zip(wl_max, v_max)), False



def _pick_with_index_exclusion_and_bounds(
    candidates, v_center_1e3, wl_grid, exclude_idx, min_sep, bounds_kms
):
    """
    Pick the maximum candidate closest in velocity to v_center_1e3,
    enforcing:
      - min index separation (>= min_sep points) from all exclude_idx
      - velocity bounds (km/s): bounds_kms = (vmin, vmax)

    Returns wl (float) or None if none pass both tests.
    """
    if not candidates:
        return None
    vmin, vmax = bounds_kms
    v_center = float(v_center_1e3)

    grid = np.asarray(wl_grid)

    def wl_to_idx(w):
        return int(np.argmin(np.abs(grid - w)))

    def sort_key(c):
        _, v1e3_c = c
        return abs(v1e3_c - v_center)

    for wl_cand, v1e3_cand in sorted(candidates, key=sort_key):
        # Spline extrapolation can produce extrema beyond observed coverage.
        if not np.isfinite(wl_cand) or not (grid.min() <= wl_cand <= grid.max()):
            continue
        v_kms = float(v1e3_cand) * 1e3
        if not (vmin < v_kms < vmax):  # strict inequalities as you specified
            continue
        idx_cand = wl_to_idx(wl_cand)
        if all(abs(idx_cand - idx_ex) >= int(min_sep) for idx_ex in exclude_idx):
            return float(wl_cand)
    return None



def _choose_unique_anchor(
    func,
    d2func,
    v_init_1e3,
    rest_wl,
    v_window_kms,
    dv_kms,
    wl_grid,
    exclude_idx,
    min_sep,
    max_attempts,
    bounds_kms,
):
    """
    Pick a unique **maximum** ensuring:
      - ≥ min_sep grid points from other anchors
      - velocity within `bounds_kms` (km/s)
    with bi-directional velocity shifts.

    Search order of centers (in 10^3 km/s):
      0, +1, -1, +2, -2, ... times (dv_kms/1e3), up to `max_attempts`.

    Status on failure:
      - 'conflict_no_unique_max' if maxima existed but all failed separation/bounds
      - 'minima_only' if only minima were found across attempts
      - 'fallback_initial' if no extrema at all
      In all cases we fallback to the initial guess wavelength; if *that* is
      out of bounds, we clip it to the nearest bound and label 'clipped_to_bounds'.
    """
    any_cands = False
    any_minima_only = False

    # attempt offsets: [0, +1, -1, +2, -2, ...]
    attempt_offsets = [0]
    for k in range(1, int(max_attempts) + 1):
        attempt_offsets += [+k, -k]

    for off in attempt_offsets:
        v_center_1e3 = float(v_init_1e3) + (off * float(dv_kms) / 1e3)
        cands, minima_only = _maxima_candidates_in_window(
            func, d2func, v_center_1e3, rest_wl, v_window_kms, dv_kms
        )
        any_cands |= bool(cands)
        any_minima_only |= bool(minima_only)

        wl_pick = _pick_with_index_exclusion_and_bounds(
            cands, v_center_1e3, wl_grid, exclude_idx, min_sep, bounds_kms
        )
        if wl_pick is not None:
            return wl_pick, "automatic"

    # --- No acceptable maximum after all attempts → fallback logic ---
    vmin, vmax = bounds_kms
    # initial-guess wavelength
    wl_fallback = float(vel_to_wl(v_init_1e3, rest_wl))
    v_fallback_kms = float(v_init_1e3) * 1e3

    if not (vmin < v_fallback_kms < vmax):
        # clip to nearest bound (convert bound velocity to wavelength)
        v_clip_kms = vmin if abs(v_fallback_kms - vmin) < abs(v_fallback_kms - vmax) else vmax
        wl_clip = float(vel_to_wl(v_clip_kms / 1e3, rest_wl))
        return wl_clip, "clipped_to_bounds"

    if any_cands:
        return wl_fallback, "out_of_bounds"  # or separation conflicts led to failure
    if any_minima_only:
        return wl_fallback, "minima_only"
    return wl_fallback, "fallback_initial"



def prepare_spectrum(
    file_spec,
    redshift,
    mwebv,
    vexp=None,
    type_of_spec="auto",
    try_clip=False,
):
    """
    Read a spectrum and build derived columns (rest-frame, smoothed, flattened).

    Parameters
    ----------
    file_spec : str or pathlib.Path
            Path to an ASCII spectrum. The first two columns are wavelength (Å) and
            flux. A third column may contain the 1-sigma flux uncertainty.
    redshift : float
            Supernova/systemic redshift.
    mwebv : float
            Milky Way E(B-V).
    type_of_spec : {"auto", "flux", "flux-error"}, optional
            Input column format. ``auto`` uses the third column as the uncertainty
            when it exists. The legacy names ``original`` and ``homogenised`` remain
            accepted as aliases for ``flux`` and ``flux-error``.
    try_clip : bool, optional
            If True, clip/interpolate outliers before downstream smoothing.
    vexp : float, optional
            Fractional Gaussian width for the first smoothing: σ(λ) = vexp × λ.
            When omitted, it is estimated from the spectrum signal-to-noise ratio.

    Returns
    -------
    df_spec : pandas.DataFrame
            Dataframe with added columns:
              - rest_wl, norm_fl, fl_err, fl_smooth, norm_flerr,
                    norm_fl_smooth, and *_flattened.
    """

    format_aliases = {
        "auto": "auto",
        "flux": "flux",
        "flux-error": "flux-error",
        "original": "flux",
        "homogenised": "flux-error",
        "homogenized": "flux-error",
    }
    try:
        input_format = format_aliases[type_of_spec]
    except KeyError as exc:
        raise ValueError(
            "type_of_spec must be 'auto', 'flux', or 'flux-error' "
            f"(legacy aliases are also accepted), got {type_of_spec!r}"
        ) from exc

    spectrum_path = Path(file_spec).expanduser()
    if not spectrum_path.is_file():
        raise FileNotFoundError(f"Spectrum file not found: {spectrum_path}")

    # Some spectra start or end with wavelength-only rows. pandas infers a
    # single column from the first such row and then skips the usable rows as
    # malformed. Parse each row independently and discard incomplete pixels.
    rows = []
    has_third_column = False
    with spectrum_path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            fields = re.split(r"[,\s]+", line.split("#", 1)[0].strip())
            if len(fields) < 2:
                continue
            try:
                wavelength, flux = float(fields[0]), float(fields[1])
            except ValueError:
                continue
            has_third_column |= len(fields) >= 3
            try:
                error = float(fields[2]) if len(fields) >= 3 else np.nan
            except ValueError:
                error = np.nan
            rows.append((wavelength, flux, error))
    if not rows:
        raise ValueError(
            f"Spectrum {spectrum_path} must contain at least wavelength and flux columns"
        )
    raw = pd.DataFrame.from_records(rows, columns=[0, 1, 2])
    if input_format == "auto":
        input_format = "flux-error" if has_third_column else "flux"
    if input_format == "flux-error" and not has_third_column:
        raise ValueError(f"Spectrum {spectrum_path} has no third (flux-error) column")

    columns = {0: "wl", 1: "fl"}
    if input_format == "flux-error":
        columns[2] = "fl_err"
    df_spec = raw.iloc[:, : len(columns)].rename(columns=columns)

    # Make sure main columns are numeric
    df_spec["wl"] = pd.to_numeric(df_spec["wl"], errors="coerce")
    df_spec["fl"] = pd.to_numeric(df_spec["fl"], errors="coerce")

    if "fl_err" in df_spec.columns:
        df_spec["fl_err"] = pd.to_numeric(df_spec["fl_err"], errors="coerce")

    df_spec = (
        df_spec.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["wl", "fl"])
        .sort_values("wl")
        .drop_duplicates(subset="wl", keep="first")
        .reset_index(drop=True)
    )
    if len(df_spec) < 8:
        raise ValueError(f"Spectrum {spectrum_path} contains fewer than 8 valid pixels")

    redshift = float(redshift)
    mwebv = float(mwebv)
    if not np.isfinite(redshift) or redshift <= -1:
        raise ValueError(f"redshift must be finite and greater than -1, got {redshift}")
    if not np.isfinite(mwebv) or mwebv < 0:
        raise ValueError(f"mwebv must be finite and non-negative, got {mwebv}")

    variance = None
    if "fl_err" in df_spec.columns:
        error_values = df_spec["fl_err"].to_numpy(dtype=float)
        if np.all(np.isfinite(error_values) & (error_values > 0)):
            with np.errstate(over="ignore", under="ignore"):
                candidate_variance = error_values**2
            if np.all(np.isfinite(candidate_variance) & (candidate_variance > 0)):
                variance = candidate_variance
    vexp_auto, snr_estimate = calc_optimal_smooth(
        df_spec["wl"].values,
        df_spec["fl"].values,
        variance=variance,
    )
    vexp = float(vexp_auto if vexp is None else vexp)
    if not np.isfinite(vexp) or vexp <= 0:
        raise ValueError(f"vexp must be finite and positive, got {vexp}")

    if try_clip:
        fl_clipped, flerr_clipped = clip_outliers(
            df_spec["wl"].values,
            df_spec["fl"].to_numpy(copy=True),
            None if variance is None else variance.copy(),
            vexp=vexp,
            redshift=redshift,
        )

        df_spec.loc[:, "fl"] = fl_clipped

        if "fl_err" in df_spec.columns and flerr_clipped is not None:
            df_spec.loc[:, "fl_err"] = np.sqrt(flerr_clipped)

    extinction_factor = pyasl.unred(
        df_spec["wl"].values,
        np.ones(len(df_spec)),
        ebv=mwebv,
        R_V=3.1,
    )
    df_spec.loc[:, "fl"] = df_spec["fl"].values * extinction_factor
    if "fl_err" in df_spec.columns:
        # Preserve invalid negative values (including -99 missing-data sentinels)
        # until the fallback below; abs() would turn them into usable errors.
        df_spec.loc[:, "fl_err"] = df_spec["fl_err"].values * extinction_factor

    df_spec.loc[:, "rest_wl"] = df_spec["wl"] / (1.0 + redshift)

    if input_format == "flux":
        n_estimated_uncertainties = len(df_spec)
        df_spec.loc[:, "fl_err"] = spec_create_errors(
            df_spec["wl"].values,
            df_spec["fl"].values,
            vexp=vexp,
            nsig=5.0,
        )

    else:  # flux-error
        bad = (~np.isfinite(df_spec["fl_err"])) | (df_spec["fl_err"] <= 0)
        n_estimated_uncertainties = int(bad.sum())
        if np.any(bad):
            df_spec.loc[bad, "fl_err"] = np.nan

            fl_err_auto = spec_create_errors(
                df_spec["wl"].values,
                df_spec["fl"].values,
                vexp=vexp,
                nsig=5.0,
            )
            df_spec.loc[:, "fl_err"] = df_spec["fl_err"].fillna(
                pd.Series(fl_err_auto, index=df_spec.index)
            )

    # Smoothing depends on relative inverse variance only. Scale uncertainties
    # before squaring so small but valid flux units cannot underflow to zero.
    errors_for_weights = df_spec["fl_err"].to_numpy(dtype=float)
    relative_errors = errors_for_weights / np.max(errors_for_weights)
    relative_errors = np.maximum(relative_errors, np.sqrt(np.finfo(float).tiny))
    df_spec.loc[:, "fl_smooth"] = smooth_spec(
        df_spec["wl"].values,
        df_spec["fl"].values,
        fl_var=relative_errors**2,
        vexp=vexp,
        nsig=5.0,
    )

    mean_fl = float(df_spec["fl"].mean())
    if not np.isfinite(mean_fl) or abs(mean_fl) < np.finfo(float).tiny:
        raise ValueError(f"Spectrum {spectrum_path} has zero or invalid mean flux")
    df_spec.loc[:, "norm_fl"] = df_spec["fl"] / mean_fl
    df_spec.loc[:, "norm_flerr"] = df_spec["fl_err"] / abs(mean_fl)
    df_spec.loc[:, "norm_fl_smooth"] = df_spec["fl_smooth"] / mean_fl

    x_for_flatten = np.linspace(df_spec["rest_wl"].iloc[0], df_spec["rest_wl"].iloc[-1], 8)
    y_data_for_flatten = np.interp(
        x_for_flatten,
        df_spec["rest_wl"],
        df_spec["norm_fl_smooth"],
    )
    y_spl_for_flatten = make_interp_spline(x_for_flatten, y_data_for_flatten, k=3)
    y_spl_flatten_eval = y_spl_for_flatten(df_spec["rest_wl"])
    if np.any(~np.isfinite(y_spl_flatten_eval)) or np.any(np.isclose(y_spl_flatten_eval, 0.0)):
        raise ValueError("Cannot flatten spectrum: continuum spline is zero or invalid")

    df_spec.loc[:, "norm_fl_flattened"] = df_spec["norm_fl"] / y_spl_flatten_eval
    df_spec.loc[:, "norm_flerr_flattened"] = df_spec["norm_flerr"] / y_spl_flatten_eval
    df_spec.loc[:, "norm_fl_smooth_flattened"] = df_spec["norm_fl_smooth"] / y_spl_flatten_eval
    df_spec.attrs.update(
        {
            "spectrum_path": str(spectrum_path.resolve()),
            "input_format": input_format,
            "vexp": vexp,
            "snr_estimate": float(snr_estimate),
            "clipped": bool(try_clip),
            "uncertainty_source": (
                "estimated"
                if n_estimated_uncertainties == len(df_spec)
                else "mixed"
                if n_estimated_uncertainties
                else "provided"
            ),
            "n_estimated_uncertainties": n_estimated_uncertainties,
        }
    )

    return df_spec



def _mc_plot_bin_edges(values, n_bins=40):
    """Bound diagnostic histogram size even when MC values have extreme tails."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return 20
    p1, p99 = np.percentile(finite, [1, 99])
    if not np.isfinite(p1) or not np.isfinite(p99) or p99 <= p1:
        return 20
    edges = np.linspace(p1, p99, n_bins + 1)
    return edges if np.all(np.diff(edges) > 0) else 20


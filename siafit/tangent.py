"""Pseudo-continuum anchors from the shoulders of absorption features.

The legacy search looks for the spline maximum closest to a *fixed* velocity
guess.  That fails whenever the absorption moves (high-velocity Si II, Ca II
high-velocity features) or the spectrum is steep enough that a shoulder is a
knee and not a maximum.  This module instead

1. locates the absorption trough itself,
2. removes the local tilt with the line joining the current anchors, and
3. takes, on each side, the innermost *prominent* maximum of the tilt-corrected
   flux that rises close to the continuum,

and repeats 2-3 until the anchors stop moving.  The fixed point is the line
that touches both shoulders of the feature (a bitangent). Removing the local
tilt helps identify shoulders on sloped spectra.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d, grey_closing
from scipy.signal import find_peaks

from . import core
from .common import vel_to_wl, wl_to_vel

C_1E3 = core.speed_of_light / 1e3  # speed of light in 10^3 km/s

# Effective wavelengths of the blended doublets/triplet, used only to turn a
# trough minimum into a starting velocity for the optimizer.
SI6355_BLEND = 0.5 * (core.rest_wl_sil1_1 + core.rest_wl_sil1_2)
SI5972_BLEND = 0.5 * (core.rest_wl_sil2_1 + core.rest_wl_sil2_2)


def _dv(x, x0):
    """Velocity offset (10^3 km/s) of wavelength(s) ``x`` relative to ``x0``."""
    return C_1E3 * np.log(np.asarray(x, dtype=float) / x0)


def _shift(x0, dv):
    """Wavelength reached from ``x0`` after a velocity offset in 10^3 km/s."""
    return x0 * np.exp(dv / C_1E3)


def smooth_noise(spectrum, lower, upper):
    """Noise left in the smoothed flux between two rest wavelengths.

    The scatter of raw minus smoothed flux gives the per-pixel noise; a
    Gaussian kernel of width ``vexp`` averages ``2 sqrt(pi) sigma / pixel``
    pixels.  Anchor candidates must stand out against this level.
    """
    wavelength = spectrum["rest_wl"].to_numpy()
    mask = (wavelength >= lower) & (wavelength <= upper)
    if np.count_nonzero(mask) < 5:
        return 0.0
    residual = (spectrum["norm_fl"].to_numpy() - spectrum["norm_fl_smooth"].to_numpy())[mask]
    per_pixel = 1.4826 * np.median(np.abs(residual - np.median(residual)))
    vexp = float(spectrum.attrs.get("vexp", 0.0) or 0.0)
    pixel = float(np.median(np.diff(wavelength[mask])))
    centre = 0.5 * (lower + upper)
    n_effective = max(1.0, 2.0 * np.sqrt(np.pi) * vexp * centre / pixel) if pixel > 0 else 1.0
    return float(per_pixel / np.sqrt(n_effective))


def envelope(spectrum, fill_kms=2000.0):
    """Smoothed flux with dips narrower than ``fill_kms`` filled, per pixel."""
    if "norm_fl_envelope" in spectrum.columns:
        return spectrum["norm_fl_envelope"].to_numpy()
    wavelength = spectrum["rest_wl"].to_numpy()
    step = float(np.median(np.diff(np.log(wavelength)))) * C_1E3 * 1e3
    size = int(round(fill_kms / step)) if step > 0 else 0
    values = spectrum["norm_fl_smooth"].to_numpy()
    if size >= 2:
        values = grey_closing(values, size=size | 1, mode="nearest")
    spectrum.loc[:, "norm_fl_envelope"] = values
    return spectrum["norm_fl_envelope"].to_numpy()


def continuum_mask(spectrum, velocity, blue, red, vel_width, tolerance=0.03):
    """Pixels of the two continuum windows that really are continuum.

    A window centred on a shoulder also covers the top of the trough wall and
    any narrow line on the shoulder (host Na I D is the usual one).  Fitting a
    line through those pixels pulls the continuum below the shoulder.  Keep
    only window pixels whose filled, smoothed flux lies within ``tolerance``
    of the line joining the two shoulders.  A side left with fewer than two
    pixels gets its whole window back.
    """
    velocity = np.asarray(velocity, dtype=float)
    wavelength = spectrum["rest_wl"].to_numpy()
    filled = envelope(spectrum)
    half = vel_width / 2.0
    sides = [(velocity >= anchor - half) & (velocity <= anchor + half) for anchor in (blue, red)]
    window = sides[0] | sides[1]
    if not (np.isfinite(blue) and np.isfinite(red) and blue < red) or not window.any():
        return window
    order = np.argsort(velocity)
    level = np.interp([blue, red], velocity[order], filled[order])
    where = np.interp([blue, red], velocity[order], wavelength[order])
    line = level[0] + (level[1] - level[0]) * (wavelength - where[0]) / (where[1] - where[0])
    noise = smooth_noise(spectrum, wavelength[window].min(), wavelength[window].max())
    # The line comes from the filled flux; the pixels are judged on the
    # smoothed flux itself, so a narrow line inside a window is dropped too.
    smoothed = spectrum["norm_fl_smooth"].to_numpy()
    good = smoothed >= line - np.maximum(2.0 * noise, tolerance * np.abs(line))
    keep = np.zeros_like(window)
    for side in sides:
        chosen = side & good
        keep |= chosen if np.count_nonzero(chosen) >= 2 else side
    return keep


def resample(spectrum, lower, upper, step_kms=150.0, fill_kms=2000.0):
    """Smoothed flux on a uniform velocity grid, without extrapolation.

    Dips narrower than ``fill_kms`` (host Na I D, telluric residuals) are
    filled by a morphological closing.  Supernova features are several times
    broader and their shoulders are untouched, so a narrow line sitting on a
    shoulder can no longer split it into two false maxima.
    """
    wavelength = spectrum["rest_wl"].to_numpy()
    flux = spectrum["norm_fl_smooth"].to_numpy()
    lower = max(lower, wavelength[0])
    upper = min(upper, wavelength[-1])
    if upper <= lower:
        raise ValueError("Spectrum does not cover the continuum search region")
    n = int(np.ceil(_dv(upper, lower) * 1e3 / step_kms)) + 1
    if n < 16:
        raise ValueError("Continuum search region is too short")
    x = lower * np.exp(np.arange(n) * step_kms / 1e3 / C_1E3)
    x = x[x <= upper]
    y = np.interp(x, wavelength, flux)
    size = int(round(fill_kms / step_kms))
    if size >= 2:
        y = grey_closing(y, size=size | 1, mode="nearest")
    return x, y


def _upper_hull(x, y):
    """Indices of the upper convex hull of points sorted by ``x``."""
    hull = []
    for i in range(len(x)):
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            if (y[b] - y[a]) * (x[i] - x[a]) <= (y[i] - y[a]) * (x[b] - x[a]):
                hull.pop()
            else:
                break
        hull.append(i)
    return np.asarray(hull)


def _line(x, i, j, y):
    slope = (y[j] - y[i]) / (x[j] - x[i])
    return y[i] + slope * (x - x[i])


def find_trough(x, y, lower, upper, *, prefer_red=0.0, outer=None, secondary_bounds=None):
    """Index of the absorption minimum inside ``[lower, upper]``.

    Depth is measured below the upper convex hull. With ``prefer_red`` > 0,
    the reddest minimum at least that fraction as deep as the deepest one wins.
    This helps separate Si II 6355 from a secondary trough to its blue.

    ``outer`` is a wider ``(lower, upper)`` interval for unusual velocities.
    A stronger outer trough replaces the inner candidate when the inner
    trough is much shallower, or when the inner trough is consistent with
    the outer trough's Si II 5972 companion.
    """
    hull = _upper_hull(x, y)
    depth = np.interp(x, x[hull], y[hull]) - y
    inside = np.flatnonzero((x >= lower) & (x <= upper))
    if inside.size < 3:
        raise ValueError("Spectrum does not sample the expected absorption trough")
    best = inside[np.argmax(depth[inside])]
    if outer is not None or prefer_red > 0:
        peaks, plateaus = find_peaks(depth, plateau_size=(1, None))
        inner_peaks = peaks[(x[peaks] >= lower) & (x[peaks] <= upper)]
    if outer is not None:
        outside = np.flatnonzero((x[peaks] >= outer[0]) & (x[peaks] <= outer[1])
                                 & ((x[peaks] < lower) | (x[peaks] > upper)))
        if outside.size:
            peak_index = outside[np.argmax(depth[peaks[outside]])]
            candidate = peaks[peak_index]
            if (plateaus["left_edges"][peak_index] <= best <= plateaus["right_edges"][peak_index]
                    or depth[best] < 0.5 * depth[candidate]):
                # Closing can leave a flat trough bottom straddling a search boundary.
                return int(candidate)
            if secondary_bounds is not None and x[candidate] > upper and inner_peaks.size:
                companion = inner_peaks[np.argmax(depth[inner_peaks])]
                primary_velocity = wl_to_vel(x[candidate], SI6355_BLEND)
                secondary_velocity = wl_to_vel(x[companion], SI5972_BLEND)
                low, high = sorted(ratio * primary_velocity for ratio in secondary_bounds)
                # Allow for the doublet centroid and smoothing of a shallow trough.
                if depth[candidate] > depth[best] and low - 1.5 <= secondary_velocity <= high + 1.5:
                    return int(candidate)
    if prefer_red > 0:
        strong = inner_peaks[depth[inner_peaks] >= prefer_red * depth[best]]
        if strong.size:
            best = strong[-1]
    if depth[best] <= 0:
        raise ValueError("No absorption trough below the local continuum")
    return int(best)


def _side_anchor(x, y, g, inner, side, current, depth, noise, settings, margin):
    """Innermost qualifying maximum between the trough and the current anchor.

    Maxima are taken from the tilt-corrected flux ``g`` and from the flux ``y``
    itself: on a steep spectrum a real emission peak can vanish once the tilt
    of a long line is removed.  A candidate must rise to within ``near`` trough
    depths of the line, sit at least ``min_dv`` from ``inner`` and stand out
    from its surroundings: by ``weak`` trough depths where a shoulder is
    expected (within ``reach`` of ``inner``), by ``strong`` trough depths
    farther out, always by three times the noise, and by a quarter of its own
    distance below the line.  ``None`` means nothing qualifies.
    """
    strong, weak, near, min_dv, reach = settings
    deficit_share = 0.25
    dominance = int(round(4.5 / (float(np.median(np.diff(np.log(x)))) * C_1E3)))
    if side < 0:
        start, stop = max(0, current - margin), inner + 1
    else:
        start, stop = inner, min(len(x), current + margin + 1)
    floor = 3.0 * noise
    candidates = []
    for signal in (g, y):
        peaks, properties = find_peaks(signal[start:stop], prominence=max(floor, weak * depth))
        for peak, prominence in zip(peaks + start, properties["prominences"]):
            if not (current < peak < inner if side < 0 else inner < peak < current):
                continue
            distance = abs(_dv(x[peak], x[inner]))
            if distance < min_dv or g[peak] < -near * depth:
                continue
            if distance > reach and prominence < max(floor, strong * depth):
                continue
            # The farther a bump sits below the line, the more it must stand
            # out: a kink on the trough wall is not a shoulder.
            if prominence < deficit_share * max(0.0, -g[peak]):
                continue
            # ...nor is the lower half of a shoulder split by a narrow line.
            outward = np.arange(peak, peak + side * (dominance + 1), side)
            outward = outward[(outward >= 0) & (outward < len(x))]
            if g[peak] < np.max(g[outward]) - 0.25 * max(floor, weak * depth):
                continue
            candidates.append(int(peak))
    if not candidates:
        return None
    return max(candidates) if side < 0 else min(candidates)


def _knee(x, y, trough, side, limit_index, min_dv, scale=2.5, significance=0.15):
    """Sharpest downward bend between the trough and a search limit.

    The bend must be a fair fraction as sharp as the trough bottom itself;
    otherwise the wall merges smoothly into the spectrum and ``None`` is
    returned.
    """
    step = float(np.median(np.diff(np.log(x)))) * C_1E3
    smooth = gaussian_filter1d(y, max(1.0, scale / step / 2.355), mode="nearest")
    curvature = np.gradient(np.gradient(smooth))
    if side < 0:
        window = np.arange(limit_index, trough)
    else:
        window = np.arange(trough + 1, limit_index + 1)
    window = window[np.abs(_dv(x[window], x[trough])) >= min_dv]
    if window.size == 0:
        return None
    best = int(window[np.argmin(curvature[window])])
    near_trough = np.abs(_dv(x, x[trough])) <= max(min_dv, 3.0)
    if -curvature[best] < significance * np.max(curvature[near_trough]):
        return None
    return best


def bitangent(x, y, trough, blue_limit, red_limit, *, noise=0.0, prominence=0.05,
              weak_prominence=0.015, near=0.35, min_dv=3.0, reach=(16.0, 18.0),
              fixed_red=None, core=None, knee=True, wall=1.2, knee_near=0.10, plateau=0.02,
              max_iter=20):
    """Anchors of the line touching both shoulders of the trough at ``trough``.

    The search starts from the upper convex hull of the allowed region, the
    widest line that stays above the spectrum, and only ever moves an anchor
    inward to a qualifying shoulder.  This makes the iteration monotone: it
    cannot oscillate, and a steep spectrum whose shoulder is followed by a
    further decline keeps the shoulder instead of sliding down the slope.

    Parameters are in the units of ``x`` (Å) and 10^3 km/s.  ``prominence``,
    ``weak_prominence`` and ``near`` are fractions of the trough depth below
    the current line; ``reach`` is the (blue, red) distance from the trough
    within which the weaker prominence is enough.  ``core`` is an optional
    ``(blue_index, red_index)`` interval known to be absorption: anchors lie
    outside it and distances are measured from its edges.
    Returns ``(blue_index, red_index, blue_status, red_status)``.
    """
    step = float(np.median(np.diff(np.log(x)))) * C_1E3
    margin = int(round(4.0 / step))
    inner_blue, inner_red = core if core is not None else (trough, trough)
    first = int(np.searchsorted(x, blue_limit, side="left"))
    last = int(np.searchsorted(x, red_limit, side="right")) - 1
    if fixed_red is not None:
        last = fixed_red
    if not first < inner_blue <= trough <= inner_red < last:
        raise ValueError("Continuum search region does not bracket the trough")
    # Hull vertices inside the core are absorption structure, not continuum.
    keep = (np.r_[first:inner_blue, inner_red + 1:last + 1] if core is not None
            else np.r_[first:last + 1])
    hull = keep[_upper_hull(x[keep], y[keep])]
    right = min(max(int(np.searchsorted(hull, trough, side="left")), 1), len(hull) - 1)
    blue, red = int(hull[right - 1]), int(hull[right])
    if fixed_red is not None:
        red = fixed_red
    status_blue = status_red = "automatic"
    for _ in range(max_iter):
        g = y - _line(x, blue, red, y)
        depth = max(float(-np.min(g[blue:red + 1])), 1e-12)
        # A shoulder cannot sit on the trough wall: keep clear of the minimum
        # by a multiple of the half width at half depth on that side.
        gap_blue = max(min_dv, wall * _half_width(x, g, inner_blue, -1, blue))
        gap_red = max(min_dv, wall * _half_width(x, g, inner_red, +1, red))
        new_blue = _side_anchor(x, y, g, inner_blue, -1, blue, depth, noise,
                                (prominence, weak_prominence, near, gap_blue, reach[0]), margin)
        new_red = None if fixed_red is not None else _side_anchor(
            x, y, g, inner_red, +1, red, depth, noise,
            (prominence, weak_prominence, near, gap_red, reach[1]), margin)
        if new_blue is None and new_red is None:
            break
        blue = blue if new_blue is None else new_blue
        red = red if new_red is None else new_red
    # A flat stretch beyond the feature lets the line touch anywhere along it.
    # The shoulder is where the trough wall first reaches the line.
    g = y - _line(x, blue, red, y)
    depth = max(float(-np.min(g[blue:red + 1])), 1e-12)
    for side, anchor, inner in ((-1, blue, inner_blue), (+1, red, inner_red)):
        if side > 0 and fixed_red is not None:
            continue
        gap = max(min_dv, wall * _half_width(x, g, inner, side, anchor))
        between = np.arange(inner + side, anchor, side)
        between = between[(np.abs(_dv(x[between], x[inner])) >= gap)
                          & (g[between] >= -plateau * depth)]
        if between.size and abs(_dv(x[between[0]], x[anchor])) > 2.0:
            if side < 0:
                blue = int(between[0])
            else:
                red = int(between[0])
    # A flat or steadily falling spectrum beyond the feature has no maximum
    # where a shoulder belongs, and the line slides along it.  If the anchor
    # ended up farther than a shoulder normally lies, take the bend where the
    # trough wall ends, provided the spectrum has already reached the line.
    if knee:
        g = y - _line(x, blue, red, y)
        depth = max(float(-np.min(g[blue:red + 1])), 1e-12)
        moves = []
        for side, anchor, inner, span in ((-1, blue, inner_blue, reach[0]),
                                          (+1, red, inner_red, reach[1])):
            if (side > 0 and fixed_red is not None) or abs(_dv(x[anchor], x[inner])) <= span:
                continue
            gap = max(min_dv, wall * _half_width(x, g, inner, side, anchor))
            stop = int(np.clip(np.searchsorted(x, _shift(x[inner], side * span)), first, last))
            found = _knee(x, y, inner, side, stop, gap)
            if found is not None and g[found] >= -knee_near * depth and (
                    anchor < found < inner if side < 0 else inner < found < anchor):
                moves.append((side, found))
        for side, found in moves:
            if side < 0:
                blue, status_blue = found, "knee"
            else:
                red, status_red = found, "knee"
    # An anchor still on the search limit touches nothing: the spectrum keeps
    # rising or falling past the feature.  Use the bend where the trough wall
    # ends, and say so.
    if blue == first:
        status_blue = "no_distinct_shoulder"
        found = _knee(x, y, inner_blue, -1, first, min_dv) if knee else None
        blue = blue if found is None else found
    if fixed_red is None and red == last:
        status_red = "no_distinct_shoulder"
        found = _knee(x, y, inner_red, +1, last, min_dv) if knee else None
        red = red if found is None else found
    return blue, red, status_blue, status_red


def _half_width(x, g, start, side, stop):
    """Distance (10^3 km/s) from ``start`` to where the trough is half as deep."""
    half = 0.5 * g[start]
    index = start
    while index != stop and g[index] < half:
        index += side
    return float(abs(_dv(x[index], x[start])))


def _half_depth_width(x, g, trough):
    """Full width (10^3 km/s) of the trough at half its depth below the line."""
    half = 0.5 * g[trough]
    left = trough
    while left > 0 and g[left] < half:
        left -= 1
    right = trough
    while right < len(x) - 1 and g[right] < half:
        right += 1
    return float(_dv(x[right], x[left]))


def _figure(x, y, anchors, title, expected=()):
    fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(6, 6))
    top.plot(x, y, color="k", label="smoothed")
    for (blue, red), color in anchors:
        line = _line(x, blue, red, y)
        span = slice(blue, red + 1)
        top.plot(x[span], line[span], color=color, ls="--")
        top.plot(x[[blue, red]], y[[blue, red]], "o", mfc="none", mew=2, color=color)
        bottom.plot(x[span], (y / line - 1)[span], color=color)
    for wavelength in expected:
        top.axvline(wavelength, color="tab:orange", ls=":", lw=1)
    bottom.axhline(0, color="tab:red", ls="--")
    top.set_title(title)
    top.set_ylabel("norm. flux")
    bottom.set_ylabel("flux / continuum - 1")
    bottom.set_xlabel("Rest wavelength (Å)")
    fig.tight_layout()
    plt.close(fig)
    return fig


def find_si_background(spectrum, config):
    """Si II 6355 and 5972 continuum anchors from the observed shoulders."""
    red_bound = vel_to_wl(config.sil6355_red_bounds_kms[1] / 1e3, core.rest_wl_sil1_1)
    blue_bound = min(
        vel_to_wl(config.sil5972_blue_bounds_kms[0] / 1e3, core.rest_wl_sil2_1),
        vel_to_wl(config.sil6355_blue_bounds_kms[0] / 1e3, core.rest_wl_sil1_1),
    )
    x, y = resample(spectrum, blue_bound - 150.0, red_bound + 150.0)
    noise = smooth_noise(spectrum, x[0], x[-1])
    # Fastest and slowest plausible 6355 minima; 5972 at ordinary velocities
    # also lies in this range, hence the preference for the redder trough.
    trough = find_trough(
        x, y, _shift(SI6355_BLEND, config.tangent_si_trough_range[0]),
        _shift(SI6355_BLEND, config.tangent_si_trough_range[1]), prefer_red=0.35,
        outer=tuple(_shift(SI6355_BLEND, limit) for limit in config.tangent_si_trough_limits),
        secondary_bounds=config.k_vel12_bounds,
    )
    span = config.tangent_search_span
    blue1, red1, status_blue1, status_red1 = bitangent(
        x, y, trough, max(x[0], _shift(x[trough], -span[0])),
        min(red_bound, _shift(x[trough], span[1])),
        noise=noise, prominence=config.tangent_prominence,
        weak_prominence=config.tangent_weak_prominence, near=config.tangent_near,
        reach=config.tangent_shoulder_reach,
    )
    g1 = y - _line(x, blue1, red1, y)
    trough = blue1 + int(np.argmin(g1[blue1:red1 + 1]))
    velocity = float(wl_to_vel(x[trough], SI6355_BLEND))
    width = _half_depth_width(x, g1, trough)
    depth1 = float(-g1[trough] / _line(x, blue1, red1, y)[trough])

    # 5972 shares the 6355 blue shoulder.  Its own blue shoulder is the next
    # prominent maximum; when the line is absent the default offset is used.
    status_blue2 = "no_distinct_shoulder"
    blue2 = None
    half = config.tangent_si5972_min_width / 2.0
    reach2 = config.tangent_si5972_reach
    # 5972 follows 6355 within the velocity ratio allowed in the fit, so its
    # minimum can only lie in this interval; S II further blue cannot be taken
    # for it.
    ratio_low, ratio_high = config.k_vel12_bounds
    trough_blue = float(vel_to_wl(1.1 * ratio_high * velocity - 1.0, SI5972_BLEND))
    trough_red = float(vel_to_wl(0.9 * ratio_low * velocity, SI5972_BLEND))
    inside = np.arange(1, blue1)
    offset = _dv(x[inside], x[blue1])
    inside = inside[(offset < -half) & (x[inside] >= trough_blue) & (x[inside] <= trough_red)]
    if inside.size >= 3:
        lower2 = max(x[0], _shift(x[blue1], -reach2))
        first2 = int(np.searchsorted(x, lower2))
        hull = _upper_hull(x[first2:blue1 + 1], y[first2:blue1 + 1]) + first2
        sag = np.interp(x[inside], x[hull], y[hull]) - y[inside]
        if np.max(sag) > 0:
            try:
                blue2, _, status_blue2, _ = bitangent(
                    x, y, int(inside[np.argmax(sag)]), lower2, x[blue1], noise=noise,
                    prominence=config.tangent_prominence,
                    weak_prominence=config.tangent_weak_prominence,
                    near=config.tangent_near, min_dv=half, fixed_red=blue1,
                    reach=(reach2, 0.0),
                )
            except ValueError:
                blue2 = None
    if blue2 is None or blue2 >= blue1:
        target = _shift(x[blue1], -2.0 * config.tangent_si5972_min_width)
        blue2 = int(np.clip(np.searchsorted(x, target), 0, max(0, blue1 - 2)))
        status_blue2 = "no_distinct_shoulder"

    background = {
        "sil6355_red_side": float(wl_to_vel(x[red1], core.rest_wl_sil1_1)),
        "sil6355_blue_side": float(wl_to_vel(x[blue1], core.rest_wl_sil1_1)),
        "sil5972_red_side": float(wl_to_vel(x[blue1], core.rest_wl_sil2_1)),
        "sil5972_blue_side": float(wl_to_vel(x[blue2], core.rest_wl_sil2_1)),
        "status_sil6355_red_side": status_red1,
        "status_sil6355_blue_side": status_blue1,
        "status_sil5972_red_side": "derived_from_sil6355_blue",
        "status_sil5972_blue_side": status_blue2,
        "sil6355_velocity_initial": min(velocity, -0.5),
        "sil6355_fwhm_initial": float(np.clip(width, 3.0, 25.0)),
        "sil6355_amp_initial": -float(np.clip(depth1 / 1.6, 0.01, 0.95)),
        "filter_continuum": True,
        "sil6355_trough_wavelength": float(x[trough]),
        "sil6355_trough_depth": depth1,
    }
    figure = _figure(
        x, y, [((blue1, red1), "tab:red"), ((blue2, blue1), "tab:blue")],
        "Si II continuum shoulders", expected=(x[trough],),
    )
    return background, figure


def find_ca_background(spectrum, config, si_velocity):
    """Ca II NIR continuum anchors bracketing photospheric and fast absorption.

    The Si II velocity fixes a core that must lie inside the feature: from the
    8498 Å line of a component ``tangent_ca_core_hvf`` faster than Si II to the
    8662 Å line at the slowest allowed photospheric velocity. A second search
    protects blue absorption beyond that core when it can be distinguished
    from the expected O I region.
    """
    from . import ca

    v_si = -abs(float(si_velocity))
    if not np.isfinite(v_si) or v_si == 0:
        raise ValueError("Ca requires a finite, nonzero Si velocity")
    blue_bound = vel_to_wl(config.ca_blue_bounds_kms[0] / 1e3, ca.rest_wl_ca_ref)
    red_bound = vel_to_wl(config.ca_red_bounds_kms[1] / 1e3, ca.rest_wl_ca_ref)
    x, y = resample(spectrum, blue_bound - 150.0, red_bound + 150.0)
    noise = smooth_noise(spectrum, x[0], x[-1])
    core_blue = float(vel_to_wl(v_si - config.tangent_ca_core_hvf, ca.rest_wl_ca1))
    core_red = float(vel_to_wl(v_si * (1.0 - config.ca_pvf_vel_frac), ca.rest_wl_ca3))
    core_blue = max(core_blue, x[2])
    core_red = min(core_red, x[-3])
    if core_red <= core_blue:
        raise ValueError("Spectrum does not cover the Ca II NIR absorption")
    trough = find_trough(x, y, core_blue, core_red)
    index_blue = int(np.searchsorted(x, core_blue))
    index_red = int(np.searchsorted(x, core_red, side="right")) - 1
    reach_blue, reach_red = config.tangent_ca_reach
    lower = max(blue_bound, x[0], _shift(core_blue, -reach_blue))
    upper = min(red_bound, x[-1], _shift(core_red, reach_red))
    settings = dict(
        noise=noise, prominence=config.tangent_prominence,
        weak_prominence=config.tangent_weak_prominence, near=config.tangent_near,
        min_dv=config.tangent_ca_min_gap, reach=config.tangent_ca_shoulder_reach,
    )
    blue, red, status_blue, status_red = bitangent(
        x, y, trough, lower, upper, core=(index_blue, index_red), **settings,
    )
    first = int(np.searchsorted(x, lower))
    hull = first + _upper_hull(x[first:red + 1], y[first:red + 1])
    depth = np.interp(x, x[hull], y[hull]) - y
    peaks, _ = find_peaks(
        depth[first:blue + 1], prominence=max(3 * noise, config.tangent_prominence * depth[trough]),
    )
    # O I near the photospheric velocity may produce a separate blue trough.
    # Require evidence redward of its expected range before widening the fit.
    oxygen_red = vel_to_wl(v_si * (1.0 - config.ca_pvf_vel_frac), 7774.0)
    peaks += first
    candidates = peaks[x[peaks] > oxygen_red]
    hvf_velocity_initial = None
    if candidates.size:
        extra = int(candidates[np.argmax(depth[candidates])])
        blue_pair_wavelength = (ca.rest_wl_ca1 + ca.rest_wl_ca2) / 2.0
        hvf_velocity_initial = float(wl_to_vel(x[extra], blue_pair_wavelength))
        blue, red, status_blue, status_red = bitangent(
            x, y, trough, lower, upper, core=(extra, index_red), **settings,
        )
        status_blue = "possible_hvf"
    background = {
        "ca_red_side": float(wl_to_vel(x[red], ca.rest_wl_ca_ref)),
        "ca_blue_side": float(wl_to_vel(x[blue], ca.rest_wl_ca_ref)),
        "status_ca_red_side": status_red,
        "status_ca_blue_side": status_blue,
        "filter_continuum": True,
    }
    if hvf_velocity_initial is not None:
        background["ca_hvf_velocity_initial"] = hvf_velocity_initial
    figure = _figure(
        x, y, [((blue, red), "tab:red")], "Ca II NIR continuum shoulders",
        expected=(core_blue, core_red),
    )
    return background, figure

"""Trough-based Si continuum selection on the supplied wavelength grid."""

from __future__ import annotations

import json

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths

from . import core
from .coverage import RequiredInterval, validate_coverage


def trough_candidates(spectrum, rest_wavelength, *, upper_wavelength=np.inf, smoothing=1.0):
    """Enumerate resolved absorption troughs and their prominence bases.

    One extra pixel of smoothing is used only for detection; fits use original
    flux and uncertainties. Scores compare morphology, not chi-square from
    differently sized fitting windows.
    """
    wavelength = spectrum['rest_wl'].to_numpy()
    velocity = core.wl_to_vel(wavelength, rest_wavelength)
    mask = (velocity > -55) & (velocity < 25) & (wavelength < upper_wavelength)
    v = velocity[mask]
    y = gaussian_filter1d(spectrum['norm_fl_smooth_flattened'].to_numpy()[mask], smoothing)
    if len(v) < 12:
        return []
    noise = np.median(np.abs(np.diff(y))) / 0.954
    minima, props = find_peaks(-y, prominence=max(0.015, 2 * noise))
    if not len(minima):
        return []
    widths = peak_widths(-y, minima, rel_height=0.5)
    edges = peak_widths(-y, minima, rel_height=0.98)
    candidates = []
    for j, index in enumerate(minima):
        left, right = int(np.floor(edges[2][j])), int(np.ceil(edges[3][j]))
        width = np.interp(widths[3][j], np.arange(len(v)), v) - np.interp(
            widths[2][j], np.arange(len(v)), v)
        if not -35 < v[index] < 1.0 or width < 2 or v[right] - v[left] < 6:
            continue
        if left == 0 or right == len(v) - 1:
            continue  # an observed continuum edge is required on both sides
        candidates.append(dict(
            velocity=float(v[index]), blue=float(v[left]), red=float(v[right]),
            width=float(width), prominence=float(props['prominences'][j]),
            score=float(props['prominences'][j] * np.sqrt(width)),
        ))
    return sorted(candidates, key=lambda candidate: candidate['score'], reverse=True)


def _candidate_background(spectrum, config):
    """Choose a morphological candidate and retain alternatives for review.

    Candidates are ranked by trough prominence and width.
    """
    candidates = trough_candidates(spectrum, core.rest_wl_sil1_1)
    if not candidates:
        raise ValueError('Adaptive Si selection requires a resolved trough and two continuum edges')
    chosen = candidates[0]
    blue_wavelength = core.vel_to_wl(chosen['blue'], core.rest_wl_sil1_1)
    secondary = trough_candidates(
        spectrum, core.rest_wl_sil2_1, upper_wavelength=blue_wavelength,
    )
    secondary = [candidate for candidate in secondary
                 if abs(candidate['velocity'] - chosen['velocity']) < max(4, abs(chosen['velocity']) * 0.4)]
    weak_secondary = not secondary
    if secondary:
        second = secondary[0]
    else:
        # A weak 5972 line must not invalidate an identifiable 6355 line. Use
        # observed upper-envelope samples outside the predicted core, record
        # that this region is conditional on 6355, and require review of 5972.
        w = spectrum['rest_wl'].to_numpy()
        v = core.wl_to_vel(w, core.rest_wl_sil2_1)
        y = spectrum['norm_fl_smooth_flattened'].to_numpy()
        center, width = chosen['velocity'], chosen['width']
        eligible = w < blue_wavelength - np.median(np.diff(w))
        blue = np.flatnonzero(eligible & (v > center - 2.5 * width) & (v < center - width))
        red = np.flatnonzero(eligible & (v > center + width) & (v < center + 2.5 * width))
        if not len(blue) or not len(red):
            raise ValueError('Adaptive Si selection lacks sampled Si II 5972 continuum shoulders')
        second = dict(blue=float(v[blue[np.argmax(y[blue])]]),
                      red=float(v[red[np.argmax(y[red])]]))
    ambiguous = any(candidate['score'] >= chosen['score'] * 0.65
                    and abs(candidate['velocity'] - chosen['velocity']) > 3
                    for candidate in candidates[1:])
    background = {
        'sil6355_blue_side': chosen['blue'], 'sil6355_red_side': chosen['red'],
        'sil5972_blue_side': second['blue'], 'sil5972_red_side': second['red'],
        'status_sil6355_blue_side': 'adaptive_ambiguous' if ambiguous else 'automatic',
        'status_sil6355_red_side': 'adaptive_ambiguous' if ambiguous else 'automatic',
        'status_sil5972_blue_side': 'adaptive_weak_5972' if weak_secondary else 'automatic',
        'status_sil5972_red_side': 'adaptive_weak_5972' if weak_secondary else 'automatic',
        'sil6355_velocity_initial': chosen['velocity'],
        'sil6355_velocity_min': chosen['blue'] + 0.5,
        'sil6355_velocity_max': min(-0.01, chosen['red'] - 1.5),
        'adaptive_candidates': json.dumps(candidates),
        'adaptive_ambiguous': int(ambiguous),
        'adaptive_weak_5972': int(weak_secondary),
        'anchor_strategy': 'adaptive',
    }
    for label, rest, blue, red in (
        ('Si II 6355 selected continuum', core.rest_wl_sil1_1, chosen['blue'], chosen['red']),
        ('Si II 5972 selected continuum', core.rest_wl_sil2_1, second['blue'], second['red']),
    ):
        validate_coverage(spectrum['rest_wl'], 0, [RequiredInterval(
            label, core.vel_to_wl(blue - config.vel_width / 2, rest),
            core.vel_to_wl(red + config.vel_width / 2, rest),
        )])
    return background, core.preview_background_all(spectrum, background, config.vel_width)


def find_background(spectrum, config):
    """Conservatively rescue truncated troughs and preserve plausible legacy fits."""
    legacy, figure = core.find_background_regions(
        spectrum, config.sil6355_red_initial, config.sil6355_blue_initial,
        config.sil5972_blue_initial, v_window_kms=config.background_search_window_kms,
        max_attempts=config.background_search_attempts,
        sil6355_red_bounds_kms=config.sil6355_red_bounds_kms,
        sil6355_blue_bounds_kms=config.sil6355_blue_bounds_kms,
        sil5972_blue_bounds_kms=config.sil5972_blue_bounds_kms,
    )
    candidates = trough_candidates(spectrum, core.rest_wl_sil1_1)
    legacy.update(anchor_strategy='adaptive', adaptive_candidates=json.dumps(candidates),
                  adaptive_ambiguous=0, adaptive_weak_5972=0, adaptive_rescued=0)
    if not candidates:
        core.plt.close(figure)
        raise ValueError('Adaptive Si selection requires a resolved trough and two continuum edges')
    chosen = candidates[0]
    legacy_fit = None
    legacy_supported = False
    try:
        metrics, fit_figure, *_ = core.fit_features(
            spectrum, legacy, n_resol_neighbors=config.n_resol_neighbors,
            k_vel12_bounds=config.k_vel12_bounds, k_fwhm12_bounds=config.k_fwhm12_bounds,
        )
        legacy_fit = metrics['sil6355_vel']
        legacy_supported = metrics['sil6355_snr'] >= 3 and all(
            legacy[key] == 'automatic'
            for key in ('status_sil6355_blue_side', 'status_sil6355_red_side')
        )
        legacy['sil6355_velocity_initial'] = legacy_fit
        core.plt.close(fit_figure)
    except (ValueError, RuntimeError):
        pass
    # Agreement with the observed trough is sufficient to retain the existing
    # continuum. A different candidate alone is not evidence of a better fit.
    agreement = legacy_fit is not None and abs(legacy_fit - chosen['velocity']) < max(3, .4 * chosen['width'])
    truncated = (abs(legacy['sil6355_blue_side'] - chosen['velocity']) < .75 * chosen['width']
                 or abs(legacy['sil6355_red_side'] - chosen['velocity']) < .75 * chosen['width']
                 or not legacy['sil6355_blue_side'] < chosen['velocity'] < legacy['sil6355_red_side'])
    failed_anchor = any(legacy[key] not in {'automatic', 'derived_from_sil6355_blue'}
                        for key in core_background_status_keys())
    if legacy_supported or (not truncated and not failed_anchor and (agreement or (legacy_fit is not None and legacy_fit < -1))):
        if not agreement:
            legacy['adaptive_ambiguous'] = 1
            legacy['status_sil6355_blue_side'] = 'adaptive_ambiguous'
        return legacy, figure
    try:
        selected, selected_figure = _candidate_background(spectrum, config)
    except ValueError as exc:
        # Preserve an inspectable result when the joint-line continuum cannot
        # be resolved. Explicitly mark this rescue as unavailable.
        legacy['adaptive_ambiguous'] = 1
        legacy['adaptive_failure_reason'] = str(exc)
        legacy['status_sil6355_blue_side'] = 'adaptive_unavailable'
        return legacy, figure
    core.plt.close(figure)
    selected['adaptive_rescued'] = 1
    return selected, selected_figure


def core_background_status_keys():
    return ('status_sil6355_blue_side', 'status_sil6355_red_side',
            'status_sil5972_blue_side', 'status_sil5972_red_side')

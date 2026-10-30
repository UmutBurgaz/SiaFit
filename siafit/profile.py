"""Automatic Si profiles with simultaneous continuum and independent line fits."""

from __future__ import annotations

import numpy as np
from lmfit import Parameters, minimize
from scipy.integrate import trapezoid

from . import common, core
from .adaptive import trough_candidates
from .quality import optimizer_diagnostics


def fit_line(spectrum, rest, companion, blue, red, initial, *, flux=None, bounds=None,
             width_bounds=None, n_resol_neighbors=3, vel_width=4.):
    """Fit a linear continuum and tied doublet together on observed pixels."""
    w = spectrum.rest_wl.to_numpy()
    y = spectrum.norm_fl.to_numpy() if flux is None else np.asarray(flux)
    e = spectrum.norm_flerr.to_numpy()
    v = core.wl_to_vel(w, rest)
    mask = (v >= blue - vel_width/2) & (v <= red + vel_width/2)
    w, y, e, v = w[mask], y[mask], e[mask], v[mask]
    if (len(w) < 8 or not np.all(np.isfinite(y)) or not np.all(np.isfinite(e))
            or np.any(e <= 0) or np.any(np.diff(w) <= 0)):
        raise ValueError('Too few valid pixels for an automatic Si profile')
    origin = float(np.mean(w))
    x = (w - origin) / 1000
    edges = (v < blue + vel_width/4) | (v > red - vel_width/4)
    if np.count_nonzero(edges) < 2:
        raise ValueError('Si profile lacks sampled continuum shoulders')
    slope, intercept = np.polyfit(x[edges], y[edges], 1, w=1/e[edges])
    resolution = max(.1, float((n_resol_neighbors-1) * np.median(np.diff(v))))
    lower, upper = bounds if bounds is not None else (blue + .3, min(-.01, red - 1))
    if lower >= upper:
        raise ValueError('Automatic Si velocity window is empty')
    def model(p, wl):
        continuum = p['level'].value + p['slope'].value * (wl - origin) / 1000
        sigma = p['fwhm'].value / 2.35482
        absorption = sum(p['amplitude'].value * np.exp(
            -.5 * ((core.wl_to_vel(wl, line) - p['velocity'].value) / sigma)**2,
        ) for line in (rest, companion))
        return continuum * (1 + absorption)
    attempts = []
    for start in (initial - .55, initial - 2, initial + 1):
        p = Parameters()
        p.add('level', value=max(intercept, .01), min=1e-8)
        p.add('slope', value=slope)
        p.add('amplitude', value=-.15, min=-.65, max=0)
        p.add('velocity', value=np.clip(start, lower + 1e-6, upper - 1e-6), min=lower, max=upper)
        width_min, width_max = width_bounds or (resolution, max(resolution * 1.01, min(35, (red-blue)*1.5)))
        width_min = max(resolution, width_min)
        if width_min >= width_max:
            raise ValueError('Si profile width bounds conflict with the sampling resolution')
        p.add('fwhm', value=np.clip(8, width_min, width_max), min=width_min, max=width_max)
        def residual(pars):
            prediction = model(pars, w)
            prediction[edges] = pars['level'].value + pars['slope'].value * x[edges]
            return (y - prediction) / e
        fit = minimize(residual, p, method='least_squares', max_nfev=800)
        if fit.success and np.all(np.isfinite(fit.residual)):
            attempts.append(fit)
    if not attempts:
        raise RuntimeError('Automatic Si profile optimizer did not converge')
    fit = min(attempts, key=lambda attempt: attempt.chisqr)
    continuum = fit.params['level'].value + fit.params['slope'].value * x
    if np.any(continuum <= 0):
        raise ValueError('Automatic Si continuum is not positive')
    null = np.polyval(np.polyfit(x, y, 1, w=1/e), x)
    improvement = max(0., float(np.sum(((y-null)/e)**2) - fit.chisqr))
    dense = np.linspace(w[0], w[-1], 500)
    dense_continuum = fit.params['level'].value + fit.params['slope'].value * (dense-origin)/1000
    absorption = model(fit.params, dense) / dense_continuum - 1
    fitted_model = model(fit.params, w)
    fitted_model[edges] = continuum[edges]
    return dict(fit=fit, wavelength=w, flux=y, error=e, continuum=continuum,
                model=fitted_model, ew=abs(trapezoid(absorption, x=dense)),
                ew_raw=abs(trapezoid(y/continuum-1, x=w)), evidence=improvement,
                velocity=fit.params['velocity'].value,
                fwhm=fit.params['fwhm'].value)


def select_primary(spectrum, *, n_resol_neighbors=3, vel_width=4.):
    """Compare line-detection improvement over a local linear continuum."""
    candidates = []
    identity, identity_radius = None, None
    legacy_reference = None
    stable_legacy = False
    morphology = trough_candidates(spectrum, core.rest_wl_sil1_1)
    # A well-supported existing line supplies identity, not the final velocity
    # or continuum. This prevents a strong 5972/other blue absorption from being
    # relabelled as 6355 solely because it has more detection power.
    try:
        legacy, legacy_figure = core.find_background_regions(
            spectrum, 0., -20., -20., v_window_kms=20000.,
        )
        core.plt.close(legacy_figure)
        metrics, figure, *_ = core.fit_features(
            spectrum, legacy, n_resol_neighbors=n_resol_neighbors, vel_width=vel_width,
        )
        core.plt.close(figure)
        if (metrics['sil6355_snr'] < 3 or metrics['sil6355_vel_err'] is None
                or (morphology and abs(metrics['sil6355_vel']-morphology[0]['velocity']) > 4)):
            probe, probe_figure = core.fit_features_mc(
                spectrum, legacy, n_iter=7, mode='resample', seed=72819,
                n_resol_neighbors=n_resol_neighbors, vel_width=vel_width,
            )
            core.plt.close(probe_figure)
            center = probe['sil6355_vel_med_resample']
            if probe['n_success_resample'] >= 5 and -35 < center < -1:
                stable_legacy = probe['sil6355_vel_std_resample'] < 2
                legacy['sil6355_velocity_initial'] = center
                metrics, figure, *_ = core.fit_features(
                    spectrum, legacy, n_resol_neighbors=n_resol_neighbors, vel_width=vel_width,
                )
                core.plt.close(figure)
        legacy_reference = (legacy, metrics)
        truncated = bool(morphology and morphology[0]['velocity'] < metrics['sil6355_vel']-4
                         and abs(legacy['sil6355_blue_side']-morphology[0]['velocity']) < morphology[0]['width'])
        secondary_misidentified = bool(morphology
            and morphology[0]['velocity'] > metrics['sil6355_vel']+10
            and abs(core.wl_to_vel(core.vel_to_wl(metrics['sil6355_vel'], core.rest_wl_sil1_1),
                                   core.rest_wl_sil2_1) - (morphology[0]['velocity']-.55)) < 3)
        if ((metrics['sil6355_snr'] >= .5 or stable_legacy)
                and abs(metrics['sil6355_amp']) >= (.001 if stable_legacy else .005)
                and -35 < metrics['sil6355_vel'] < -1 and not truncated and not secondary_misidentified):
            identity = metrics['sil6355_vel']
            identity_radius = max(4.5, min(8., metrics['sil6355_fwhm']/2))
            candidates.append(dict(velocity=identity, blue=legacy['sil6355_blue_side'],
                                   red=legacy['sil6355_red_side']))
    except (ValueError, RuntimeError):
        pass
    partner_identity = None
    if morphology:
        blue_feature = morphology[0]
        red_velocity = core.wl_to_vel(
            core.vel_to_wl(blue_feature['velocity']-.55, core.rest_wl_sil1_1),
            core.rest_wl_sil2_1,
        )
        if -25 < red_velocity < -1:
            try:
                blue_fit = fit_line(spectrum, core.rest_wl_sil1_1, core.rest_wl_sil1_2,
                                    blue_feature['blue'], blue_feature['red'], blue_feature['velocity'],
                                    n_resol_neighbors=n_resol_neighbors, vel_width=vel_width)
                partners = []
                for width in (6., 10., 14.):
                    b, r = max(-55., red_velocity-1.5*width), min(25., red_velocity+1.5*width)
                    fitted = fit_line(spectrum, core.rest_wl_sil1_1, core.rest_wl_sil1_2,
                                      b, r, red_velocity, n_resol_neighbors=n_resol_neighbors,
                                      vel_width=vel_width)
                    amplitude = fitted['fit'].params['amplitude']
                    if (amplitude.stderr is not None and amplitude.stderr > 0
                            and abs(amplitude.value)/amplitude.stderr >= 2
                            and abs(amplitude.value) >= .02
                            and fitted['evidence'] >= max(9., .1*blue_fit['evidence'])
                            and abs(fitted['velocity']-red_velocity) < max(3., .3*abs(red_velocity))):
                        partners.append((fitted['evidence'], fitted, b, r))
                if partners:
                    _, partner, b, r = max(partners, key=lambda item: item[0])
                    partner_identity = partner['velocity']
                    identity, identity_radius = partner_identity, max(3., min(6., partner['fwhm']/2))
                    candidates.append(dict(velocity=identity, blue=b, red=r))
            except (ValueError, RuntimeError):
                pass
    if identity is None and morphology:
        identity = morphology[0]['velocity'] - .55
        identity_radius = max(4., min(8., .75 * morphology[0]['width']))
    for smoothing in (1., 2., 3.):
        for candidate in trough_candidates(spectrum, core.rest_wl_sil1_1, smoothing=smoothing)[:3]:
            if not any(abs(candidate['blue']-old['blue']) < 1 and abs(candidate['red']-old['red']) < 1
                       for old in candidates):
                candidates.append(candidate)
    # Derivative extrema can disappear in a tilted or blended continuum. Scan
    # velocity and width as well, using a linear template calculation to retain
    # several distinct hypotheses for nonlinear fitting.
    w = spectrum.rest_wl.to_numpy()
    y = spectrum.norm_fl.to_numpy()
    e = spectrum.norm_flerr.to_numpy()
    v = core.wl_to_vel(w, core.rest_wl_sil1_1)
    grid = []
    for center in np.arange(-35., -1., 3.):
        for width in (4., 8., 12., 16.):
            blue, red = max(-55., center - 1.5*width), min(25., center + 1.5*width)
            mask = (v > blue) & (v < red)
            if mask.sum() < 8:
                continue
            x = (w[mask] - np.mean(w[mask])) / 1000
            gaussian = sum(np.exp(-.5*((core.wl_to_vel(w[mask], rest)-center)/(width/2.35482))**2)
                           for rest in (core.rest_wl_sil1_1, core.rest_wl_sil1_2))
            null_design = np.column_stack((np.ones(len(x)), x)) / e[mask, None]
            design = np.column_stack((np.ones(len(x)), x, gaussian)) / e[mask, None]
            target = y[mask]/e[mask]
            parameters = np.linalg.lstsq(design, target, rcond=None)[0]
            if parameters[-1] >= 0:
                continue
            null = np.linalg.lstsq(null_design, target, rcond=None)[0]
            score = np.sum((target-null_design@null)**2)-np.sum((target-design@parameters)**2)
            grid.append((score, dict(velocity=float(center), blue=float(blue), red=float(red))))
    selected_grid = []
    for _, candidate in sorted(grid, key=lambda item: item[0], reverse=True):
        if all(abs(candidate['velocity']-other['velocity']) > 4 for other in selected_grid):
            selected_grid.append(candidate)
        if len(selected_grid) == 4:
            break
    candidates.extend(selected_grid)
    fits = []
    for candidate in candidates:
        try:
            result = fit_line(spectrum, core.rest_wl_sil1_1, core.rest_wl_sil1_2,
                              candidate['blue'], candidate['red'], candidate['velocity'],
                              n_resol_neighbors=n_resol_neighbors, vel_width=vel_width)
            if identity is not None and abs(result['velocity']-identity) > identity_radius:
                continue
            result['legacy_reference'] = legacy_reference
            result['partner_identity'] = partner_identity
            result['stable_legacy'] = stable_legacy
            fits.append((result['evidence'], candidate, result))
        except (ValueError, RuntimeError):
            continue
    if not fits:
        raise ValueError('No sampled Si II 6355 profile could be fitted')
    return max(fits, key=lambda item: item[0]), fits


def select_background(spectrum, config):
    import json

    (_, candidate, primary), alternatives = select_primary(
        spectrum, n_resol_neighbors=config.n_resol_neighbors, vel_width=config.vel_width,
    )
    velocity, width = primary['velocity'], primary['fwhm']
    reference = primary.get('legacy_reference')
    morphology = trough_candidates(spectrum, core.rest_wl_sil1_1)
    if reference is not None and morphology:
        legacy, metrics = reference
        observed = morphology[0]
        old_velocity = metrics['sil6355_vel']
        truncated = (observed['velocity'] < old_velocity-4
                     and abs(legacy['sil6355_blue_side']-observed['velocity']) < observed['width'])
        same_line = abs(old_velocity - (observed['velocity']-.55)) < max(3., .4*observed['width'])
        if primary.get('partner_identity') is not None:
            same_line |= abs(old_velocity-primary['partner_identity']) < 3
        secondary_velocity = core.wl_to_vel(
            core.vel_to_wl(observed['velocity'], core.rest_wl_sil1_1), core.rest_wl_sil2_1,
        )
        secondary_alias = abs(secondary_velocity-old_velocity) < max(3., .3*abs(old_velocity))
        if (primary.get('stable_legacy') and secondary_velocity < -2
                and .3 < secondary_velocity/old_velocity < 1.6):
            secondary_alias = True
            truncated = False
        reverse_alias = (observed['velocity'] > old_velocity+10
                         and abs(core.wl_to_vel(core.vel_to_wl(old_velocity, core.rest_wl_sil1_1),
                                                core.rest_wl_sil2_1) - (observed['velocity']-.55)) < 3)
        new_error = primary['fit'].params['velocity'].stderr
        old_error = metrics['sil6355_vel_err']
        unstable_replacement = bool(primary.get('stable_legacy') and not truncated
            and primary['velocity'] > old_velocity-3 and new_error is not None
            and new_error > max(1.5, 2*(old_error or 0)))
        proposed_secondary = core.wl_to_vel(
            core.vel_to_wl(primary['velocity'], core.rest_wl_sil1_1), core.rest_wl_sil2_1,
        )
        proposed_alias = proposed_secondary < -2 and .3 < proposed_secondary/old_velocity < 1.6
        # A resolved fast primary must not be held near a normal-velocity
        # solution by a weak secondary line or a truncated old continuum.
        fast_primary = primary['velocity'] < -17 and not proposed_alias
        if (abs(metrics['sil6355_amp']) >= (.001 if primary.get('stable_legacy') else .005)
                and (metrics['sil6355_snr'] >= .5 or primary.get('stable_legacy'))
                and not fast_primary and not truncated and not reverse_alias
                and (same_line or secondary_alias or unstable_replacement)):
            legacy.update(anchor_strategy='legacy', adaptive_policy='validated_legacy',
                          adaptive_ambiguous=int(secondary_alias and not same_line),
                          adaptive_rescued=0, adaptive_weak_5972=0,
                          adaptive_candidates=json.dumps([dict(velocity=r['velocity'], evidence=s)
                                                          for s, _, r in alternatives]))
            if legacy['adaptive_ambiguous']:
                legacy['status_sil6355_blue_side'] = 'profile_ambiguous'
            return legacy, dict(velocity=old_velocity, uncertainty=metrics['sil6355_vel_err'],
                                redchi=metrics['optimizer_redchi'], evidence=np.nan,
                                method='joint_legacy_validated', blue=legacy['sil6355_blue_side'],
                                red=legacy['sil6355_red_side'])
    # A separate window for 5972 is allowed to be weak or absent. It cannot
    # move the already identified 6355 profile onto a different trough.
    blue2 = max(-55., velocity - max(8., 1.5 * width))
    red2 = min(8., velocity + max(5., width))
    ambiguous = any(score >= .7 * primary['evidence'] and abs(result['velocity']-velocity) > 3
                    for score, _, result in alternatives if result is not primary)
    background = dict(
        sil6355_blue_side=candidate['blue'], sil6355_red_side=candidate['red'],
        sil5972_blue_side=blue2, sil5972_red_side=red2,
        sil6355_velocity_initial=velocity,
        anchor_strategy='profile',
        adaptive_candidates=json.dumps([dict(velocity=result['velocity'], evidence=score)
                                        for score, _, result in alternatives]),
        adaptive_ambiguous=int(ambiguous), adaptive_rescued=1, adaptive_weak_5972=0,
    )
    for label in ('sil6355_blue_side', 'sil6355_red_side', 'sil5972_blue_side', 'sil5972_red_side'):
        background[f'status_{label}'] = 'profile_ambiguous' if ambiguous else 'automatic'
    return background, dict(velocity=velocity, uncertainty=primary['fit'].params['velocity'].stderr,
                            redchi=primary['fit'].redchi, evidence=primary['evidence'],
                            method='independent_profiles', blue=candidate['blue'], red=candidate['red'])


def find_background(spectrum, config):
    background, _ = select_background(spectrum, config)
    return background, core.preview_background_all(spectrum, background, config.vel_width)


def fit_once(spectrum, background, *, flux=None, k_vel12_bounds=(.6, 1.3),
             k_fwhm12_bounds=(.6, 1.25), n_resol_neighbors=3, vel_width=4.):
    primary = fit_line(spectrum, core.rest_wl_sil1_1, core.rest_wl_sil1_2,
                       background['sil6355_blue_side'], background['sil6355_red_side'],
                       background['sil6355_velocity_initial'], flux=flux,
                       n_resol_neighbors=n_resol_neighbors, vel_width=vel_width)
    secondary, failure = None, ''
    v, width = primary['velocity'], primary['fwhm']
    try:
        secondary = fit_line(
            spectrum, core.rest_wl_sil2_1, core.rest_wl_sil2_2,
            background['sil5972_blue_side'], background['sil5972_red_side'], v,
            flux=flux, bounds=tuple(sorted(v * np.asarray(k_vel12_bounds))),
            width_bounds=tuple(width * np.asarray(k_fwhm12_bounds)),
            n_resol_neighbors=n_resol_neighbors, vel_width=vel_width,
        )
    except (ValueError, RuntimeError) as exc:
        failure = str(exc)
    metrics = optimizer_diagnostics(primary['fit'])
    # Keep the established diagnostic name for the primary line velocity.
    for old, new in [('velocity', 'v_sil1_1'), ('fwhm', 'fwhm_sil1_1'), ('amplitude', 'a_sil1_1')]:
        metrics[f'at_bound_{new}'] = metrics.pop(f'at_bound_{old}')
    metrics.update(optimizer_aic=primary['fit'].aic, optimizer_bic=primary['fit'].bic,
                   si_fit_method='independent_profiles', sil5972_fit_failure=failure,
                   sil5972_fit_success=int(secondary is not None),
                   sil6355_profile_evidence=primary['evidence'],
                   sil5972_profile_evidence=secondary['evidence'] if secondary else np.nan,
                   sil5972_profile_review_flag=int(secondary is None or secondary['evidence'] < 9))
    for label, result, wavelengths in (
        ('sil6355', primary, [('6347', core.rest_wl_sil1_1), ('6371', core.rest_wl_sil1_2)]),
        ('sil5972', secondary, [('5958', core.rest_wl_sil2_1), ('5979', core.rest_wl_sil2_2)]),
    ):
        for name in ('vel', 'vel_err', 'fwhm', 'fwhm_err', 'amp', 'ew', 'ew_raw', 'snr'):
            metrics[f'{label}_{name}'] = np.nan
        if result is not None:
            p = result['fit'].params
            rms = np.sqrt(np.mean(((result['flux']-result['model'])/result['continuum'])**2))
            depth = 2 * abs(p['amplitude'].value)
            metrics.update({f'{label}_vel': result['velocity'], f'{label}_vel_err': p['velocity'].stderr,
                            f'{label}_fwhm': result['fwhm'], f'{label}_fwhm_err': p['fwhm'].stderr,
                            f'{label}_amp': p['amplitude'].value, f'{label}_ew': result['ew'],
                            f'{label}_ew_raw': result['ew_raw'], f'{label}_snr': depth/rms if rms else np.inf})
        for line, rest in wavelengths:
            metrics[f'{label}_{line}_fwhm_angstrom'] = (
                common.velocity_fwhm_to_angstrom(result['velocity'], result['fwhm'], rest)
                if result else np.nan)
    metrics['r_sil'] = metrics['sil5972_ew']/metrics['sil6355_ew'] if metrics['sil6355_ew'] > 0 else np.nan
    metrics['r_sil_raw'] = metrics['sil5972_ew_raw']/metrics['sil6355_ew_raw'] if metrics['sil6355_ew_raw'] > 0 else np.nan
    return metrics, primary, secondary


def fit_features(spectrum, background, **settings):
    metrics, primary, secondary = fit_once(spectrum, background, **settings)
    fig, axes = core.plt.subplots(2, 1, figsize=(8, 7))
    for ax, result, label in zip(axes, (primary, secondary), ('Si II 6355', 'Si II 5972')):
        if result is None:
            ax.text(.5, .5, 'No valid secondary fit', transform=ax.transAxes, ha='center')
        else:
            ax.plot(result['wavelength'], result['flux'], color='.6', label='spectrum')
            ax.plot(result['wavelength'], result['model'], color='tab:blue', label='profile')
            ax.plot(result['wavelength'], result['continuum'], color='tab:red', ls='--', label='continuum')
            ax.set_title(f"{label}: {result['velocity']:.2f} × 1000 km/s")
            ax.legend(fontsize=8)
        ax.set_xlabel('Rest wavelength (Å)')
    fig.tight_layout()
    core.plt.close(fig)
    return metrics, fig, primary['fit'], secondary['fit'] if secondary else None, primary['fit']


def fit_features_mc(spectrum, background, n_iter=100, mode='resample', continuum_shift=1.3,
                    seed=None, **settings):
    if mode not in {'resample', 'shift', 'shift_resample', 'all'}:
        raise ValueError('invalid mc_mode')
    rng = np.random.default_rng(seed)
    names = ['sil6355_vel', 'sil5972_vel', 'sil6355_fwhm', 'sil5972_fwhm',
             'sil6355_ew', 'sil5972_ew', 'sil6355_ew_raw', 'sil5972_ew_raw', 'r_sil', 'r_sil_raw',
             'any_at_bound', 'sil5972_fit_success']
    modes = ['resample', 'shift', 'shift_resample'] if mode == 'all' else [mode]
    keys = ('sil6355_blue_side', 'sil6355_red_side', 'sil5972_blue_side', 'sil5972_red_side')
    def draw(draw_mode):
        flux = common.mc_draw_flux(rng, spectrum.norm_fl.to_numpy(), spectrum.norm_flerr.to_numpy(), draw_mode)
        bg = common.mc_shift_background(rng, background, keys, continuum_shift, draw_mode)
        return fit_once(spectrum, bg, flux=flux, **settings)[0]
    results, summaries, failures, reasons = common.collect_mc_draws(names, modes, n_iter, draw)
    summary = dict(mc_mode=mode, mc_n_iter=n_iter, mc_continuum_shift=continuum_shift,
                   mc_profile_selection='conditional_on_selected_continuum')
    fig, axes = core.plt.subplots(2, 2, figsize=(8, 6))
    for draw_mode in modes:
        summary[f'n_success_{draw_mode}'] = summaries[draw_mode]['sil6355_vel']['n']
        summary[f'n_fail_{draw_mode}'] = failures[draw_mode]
        summary[f'failure_reasons_{draw_mode}'] = ' | '.join(f'{count}x {reason}' for reason, count in reasons[draw_mode].items())
        summary[f'n_secondary_success_{draw_mode}'] = int(np.sum(results[draw_mode]['sil5972_fit_success']))
        summary[f'any_at_bound_fraction_{draw_mode}'] = summaries[draw_mode]['any_at_bound']['mean']
        for name in names[:10]:
            for stat in ('med', 'hi', 'lo', 'mean', 'std', 'gap', 'flag'):
                summary[f'{name}_{stat}_{draw_mode}'] = summaries[draw_mode][name][stat]
        for ax, name in zip(axes.flat, names[:4]):
            values = results[draw_mode][name]
            values = values[np.isfinite(values)]
            if len(values):
                ax.hist(values, bins=common._mc_plot_bin_edges(values), alpha=.4, label=draw_mode)
            ax.set_title(name)
    fig.tight_layout()
    core.plt.close(fig)
    return summary, fig

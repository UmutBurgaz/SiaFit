"""Ca II and joint Si II to Ca II catalogue workflows."""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import ca, core
from . import pipeline as si
from .catalog import load_catalog, portable_spectrum_path
from .config import FitConfig

CA_KEYS = ("ca_red_side", "ca_blue_side")
CA_STATUS_KEYS = ("status_ca_red_side", "status_ca_blue_side")


def _prior_velocity(record, config):
    raw = record.metadata.get(
        "selected_si_velocity", record.metadata.get("si_velocity", config.si_velocity)
    )
    try:
        velocity = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Ca-only fitting requires a finite si_velocity in the catalogue "
            "or --si-velocity on the command line (10³ km/s)"
        ) from exc
    if not np.isfinite(velocity):
        raise ValueError("si_velocity must be finite (10³ km/s)")
    return velocity


def _select_ca_background(df_spec, record, config, si_velocity):
    background, anchor_figure = ca.find_background_regions(
        df_spec,
        ca_red_side_init=config.ca_red_initial,
        ca_blue_side_init=config.ca_blue_initial,
        v_window_kms=config.ca_background_search_window_kms,
        dv_kms_red=config.ca_background_search_step_kms,
        dv_kms_blue=-config.ca_background_search_step_kms,
        max_attempts=config.ca_background_search_attempts,
        ca_red_bounds_kms=config.ca_red_bounds_kms,
        ca_blue_bounds_kms=config.ca_blue_bounds_kms,
    )
    if record.mode == "auto":
        preview = ca.preview_background(
            df_spec, background, vel_width=config.vel_width, si_vel_1e3=si_velocity
        )
        return si.BackgroundSelection(background, config.vel_width, preview, anchor_figure)
    background, preview, metrics, fit_figure, width, quality = (
        ca.interactive_refine_background_and_fit(
            df_spec, background, si_velocity,
            spec_meta={
                "name": record.name, "spectrum": record.spectrum,
                "redshift": record.redshift, "phase": record.phase,
                "telescope": record.metadata.get("telescope"),
                "instrument": record.metadata.get("instrument"),
            },
            vel_width=config.vel_width,
            n_resol_neighbors=config.ca_n_resol_neighbors,
            hvf_min_sep_kms=config.ca_hvf_min_sep_kms,
            pvf_vel_frac=config.ca_pvf_vel_frac,
            manual_start=record.mode == "manual",
        )
    )
    if quality == 0:
        raise ValueError("Spectrum rejected during Ca II review")
    return si.BackgroundSelection(background, width, preview, anchor_figure, metrics, fit_figure)


def _fit_ca(df_spec, selection, si_velocity, config, seed):
    if selection.fit_metrics is not None and selection.fit_figure is not None:
        metrics, fit_figure = selection.fit_metrics, selection.fit_figure
    else:
        metrics, fit_figure, _, _ = ca.fit_features(
            df_spec, selection.background, si_velocity,
            vel_width=selection.vel_width,
            n_resol_neighbors=config.ca_n_resol_neighbors,
            hvf_min_sep_kms=config.ca_hvf_min_sep_kms,
            pvf_vel_frac=config.ca_pvf_vel_frac,
        )
    mc_summary = {"mc_mode": "disabled", "mc_n_iter": 0}
    mc_figure = None
    if config.n_iterations:
        mc_summary, mc_figure = ca.fit_features_mc(
            df_spec, selection.background, si_velocity,
            n_iter=config.n_iterations,
            vel_width=selection.vel_width,
            n_resol_neighbors=config.ca_n_resol_neighbors,
            hvf_min_sep_kms=config.ca_hvf_min_sep_kms,
            pvf_vel_frac=config.ca_pvf_vel_frac,
            mode=config.mc_mode,
            continuum_shift=config.continuum_shift,
            seed=seed,
        )
    return metrics, fit_figure, mc_summary, mc_figure


def _background_row(record, output_dir, df_spec, config, clip, si_selection, ca_selection,
                    si_velocity, velocity_source):
    row = si._base_row(record, output_dir, df_spec, mode=record.mode, clip_outliers=clip)
    row["features"] = config.features
    row["selected_si_velocity"] = si_velocity
    row["selected_si_velocity_source"] = velocity_source
    if si_selection is not None:
        row["background_vel_width"] = si_selection.vel_width
        for key in si.BACKGROUND_KEYS + si.BACKGROUND_STATUS_KEYS:
            row[f"background_{key}"] = si_selection.background.get(key)
        row["background_qc_flag"] = int(any(
            si_selection.background.get(key) not in {"automatic", "manual", "derived_from_sil6355_blue"}
            for key in si.BACKGROUND_STATUS_KEYS
        ))
    row["ca_background_vel_width"] = ca_selection.vel_width
    for key in CA_KEYS + CA_STATUS_KEYS:
        row[f"background_{key}"] = ca_selection.background.get(key)
    row["ca_background_qc_flag"] = int(any(
        ca_selection.background.get(key) not in {"automatic", "manual"}
        for key in CA_STATUS_KEYS
    ))
    return row


def _saved_ca_selection(record, df_spec, si_velocity, config):
    background = {}
    for key in CA_KEYS:
        column = f"background_{key}"
        if column not in record.metadata:
            raise ValueError(f"Background catalogue is missing {column}")
        background[key] = float(record.metadata[column])
        if not np.isfinite(background[key]):
            raise ValueError(f"{column} must be finite")
    for key in CA_STATUS_KEYS:
        column = f"background_{key}"
        if column in record.metadata:
            background[key] = str(record.metadata[column])
    width = float(record.metadata.get("ca_background_vel_width", config.vel_width))
    if not np.isfinite(width) or width <= 0:
        raise ValueError("ca_background_vel_width must be finite and positive")
    preview = ca.preview_background(df_spec, background, width, si_velocity)
    return si.BackgroundSelection(background, width, preview)


def _save_plot(record, output_dir, panels, suffix=""):
    figures = [figure for _, figure in panels if figure is not None]
    titles = [title for title, figure in panels if figure is not None]
    subtitle = Path(record.spectrum).name
    if record.phase is not None:
        subtitle += f" | phase={record.phase:.2f} d"
    core.save_summary_figure(
        figures,
        output_dir / "plots" / f"{si._safe_name(record.record_id)}{suffix}.png",
        title=record.name, subtitle=subtitle, panel_titles=titles,
        width_per_panel=4.0, height=5.0,
    )


def _record_fit(row, df_spec, config, seed, si_selection, ca_selection, si_velocity):
    panels = []
    if si_selection is not None:
        si_metrics, si_figure, si_mc, si_mc_figure = si._fit_spectrum(
            df_spec, si_selection.background, si_selection.vel_width, config,
            seed=seed, existing_metrics=si_selection.fit_metrics,
            existing_figure=si_selection.fit_figure,
        )
        # This object is passed directly to Ca; no output-table lookup occurs.
        si_velocity = float(si_metrics["sil6355_vel"])
        row["selected_si_velocity"] = si_velocity
        row.update(si._flatten("best_fit_", si_metrics))
        row.update(si._flatten("si_", si_mc))
        row["si_mc_success_fraction"], row["si_mc_qc_flag"] = si._mc_quality(
            si_mc, config.n_iterations
        )
        panels.extend([
            ("Si II continuum", si_selection.background_figure),
            ("Si II fit", si_figure),
            ("Si II Monte Carlo", si_mc_figure),
        ])
    ca_metrics, ca_figure, ca_mc, ca_mc_figure = _fit_ca(
        df_spec, ca_selection, si_velocity, config, seed
    )
    row.update(si._flatten("best_fit_", ca_metrics))
    row.update(si._flatten("ca_", ca_mc))
    row["ca_mc_success_fraction"], row["ca_mc_qc_flag"] = si._mc_quality(
        ca_mc, config.n_iterations
    )
    panels.extend([
        ("Ca II continuum", ca_selection.background_figure),
        ("Ca II fit", ca_figure),
        ("Ca II Monte Carlo", ca_mc_figure),
    ])
    row["mc_seed"] = seed
    return row, panels


def _run(catalog_path, output_dir, *, config, spectra_dir, stage, start, end, resume,
         saved_backgrounds):
    config.validate()
    if config.features not in {"ca", "all"}:
        raise ValueError("Combined workflow requires Ca II")
    output_dir = Path(output_dir).expanduser().resolve()
    names = ("fit_results.csv", "fit_log.csv") if saved_backgrounds else (
        "backgrounds.csv", "fit_results.csv", "fit_log.csv", "background_log.csv"
    )
    si._protect_existing_outputs(output_dir, names, resume)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)
    records = load_catalog(
        catalog_path, spectra_dir=spectra_dir,
        default_mode=config.mode, default_input_format=config.input_format,
    )[start:end]
    background_path = output_dir / "backgrounds.csv"
    result_path = output_dir / "fit_results.csv"
    log_path = output_dir / ("background_log.csv" if stage == "background" else "fit_log.csv")
    background_rows = si._existing_rows(background_path, resume) if not saved_backgrounds else []
    result_rows = si._existing_rows(result_path, resume) if stage == "all" else []
    log_rows = si._existing_rows(log_path, resume)
    completed_rows = background_rows if stage == "background" else result_rows
    completed = {str(row.get("record_id")) for row in completed_rows}
    succeeded = failed = skipped = 0
    for position, record in enumerate(records, 1):
        if record.record_id in completed:
            skipped += 1
            continue
        started = time.monotonic()
        try:
            df_spec, clip = si._prepare(record, config)
            seed = None if config.seed is None else config.seed + record.catalog_index
            si_selection = None
            if config.features == "all":
                if saved_backgrounds:
                    si_background, si_width = si._background_from_record(
                        record, default_vel_width=config.vel_width
                    )
                    si_preview = core.preview_background_all(df_spec, si_background, si_width)
                    si_selection = si.BackgroundSelection(si_background, si_width, si_preview)
                else:
                    si_selection = si.select_background(df_spec, mode=record.mode, config=config)
                if si_selection.fit_metrics is not None:
                    si_velocity = float(si_selection.fit_metrics["sil6355_vel"])
                else:
                    si_metrics, si_temp_figure, *_ = core.fit_features(
                        df_spec, si_selection.background, vel_width=si_selection.vel_width,
                        n_resol_neighbors=config.n_resol_neighbors,
                        k_vel12_bounds=config.k_vel12_bounds,
                        k_fwhm12_bounds=config.k_fwhm12_bounds,
                    )
                    si_velocity = float(si_metrics["sil6355_vel"])
                    si_selection.fit_metrics = si_metrics
                    si_selection.fit_figure = si_temp_figure
                velocity_source = "joint_si_fit"
            else:
                si_velocity = _prior_velocity(record, config)
                velocity_source = "catalogue_or_option"
            if saved_backgrounds:
                ca_selection = _saved_ca_selection(record, df_spec, si_velocity, config)
            else:
                ca_selection = _select_ca_background(df_spec, record, config, si_velocity)
            row = _background_row(
                record, output_dir, df_spec, config, clip, si_selection, ca_selection,
                si_velocity, velocity_source,
            )
            if stage == "background":
                panels = []
                if si_selection is not None:
                    panels.extend([
                        ("Si II anchor search", si_selection.anchor_figure),
                        ("Si II continuum", si_selection.background_figure),
                    ])
                panels.extend([
                    ("Ca II anchor search", ca_selection.anchor_figure),
                    ("Ca II continuum", ca_selection.background_figure),
                ])
                _save_plot(record, output_dir, panels, suffix="_background")
                si._checkpoint_row(background_rows, row, background_path)
            else:
                if not saved_backgrounds:
                    si._checkpoint_row(background_rows, row, background_path)
                result, panels = _record_fit(
                    row, df_spec, config, seed, si_selection, ca_selection, si_velocity
                )
                _save_plot(record, output_dir, panels)
                si._checkpoint_row(result_rows, result, result_path)
            log_rows.append({
                "record_id": record.record_id, "name": record.name,
                "spectrum": portable_spectrum_path(record.spectrum_path, output_dir),
                "stage": stage, "features": config.features, "success": 1,
                "error_type": "", "error_message": "",
                "runtime_seconds": round(time.monotonic() - started, 3),
            })
            succeeded += 1
            print(f"[{position}/{len(records)}] OK   {record.record_id}")
        except (KeyboardInterrupt, EOFError):
            raise
        except Exception as exc:
            log_rows.append({
                "record_id": record.record_id, "name": record.name,
                "spectrum": portable_spectrum_path(record.spectrum_path, output_dir),
                "stage": stage, "features": config.features, "success": 0,
                "error_type": type(exc).__name__, "error_message": str(exc),
                "runtime_seconds": round(time.monotonic() - started, 3),
            })
            failed += 1
            print(f"[{position}/{len(records)}] FAIL {record.record_id}: {exc}")
        finally:
            plt.close("all")
            si._atomic_csv(log_rows, log_path)
    return si.RunSummary(len(records), succeeded, failed, skipped, output_dir)


def run_catalog(catalog_path, output_dir, *, config: FitConfig, spectra_dir=None,
                stage="all", start=0, end=None, resume=False):
    if stage not in {"all", "background"}:
        raise ValueError("stage must be 'all' or 'background'")
    return _run(catalog_path, output_dir, config=config, spectra_dir=spectra_dir,
                stage=stage, start=start, end=end, resume=resume, saved_backgrounds=False)


def fit_background_catalog(background_catalog, output_dir, *, config: FitConfig,
                           spectra_dir=None, start=0, end=None, resume=False):
    return _run(background_catalog, output_dir, config=config, spectra_dir=spectra_dir,
                stage="all", start=start, end=end, resume=resume, saved_backgrounds=True)

"""High-level, restartable Si II background-selection and fitting workflow."""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import core
from .catalog import SpectrumRecord, load_catalog, portable_spectrum_path
from .config import FitConfig
from .coverage import describe_requirements, validate_coverage

BACKGROUND_KEYS = (
    "sil6355_red_side",
    "sil6355_blue_side",
    "sil5972_red_side",
    "sil5972_blue_side",
)
BACKGROUND_STATUS_KEYS = (
    "status_sil6355_red_side",
    "status_sil6355_blue_side",
    "status_sil5972_red_side",
    "status_sil5972_blue_side",
)


@dataclass
class BackgroundSelection:
    background: dict[str, Any]
    vel_width: float
    background_figure: Any
    anchor_figure: Any | None = None
    fit_metrics: dict[str, Any] | None = None
    fit_figure: Any | None = None


@dataclass(frozen=True)
class RunSummary:
    total: int
    succeeded: int
    failed: int
    skipped: int
    output_dir: Path


def _as_bool(value: Any, default: bool) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    return default


def _safe_name(record_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", record_id).strip("._")
    digest = hashlib.sha256(record_id.encode("utf-8")).hexdigest()[:12]
    return f"{cleaned[:120] or 'spectrum'}-{digest}"


def _atomic_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(path)


def _existing_rows(path: Path, resume: bool) -> list[dict[str, Any]]:
    if not resume or not path.is_file():
        return []
    return pd.read_csv(path, converters={"record_id": str}).to_dict(orient="records")


def _protect_existing_outputs(output_dir: Path, names: tuple[str, ...], resume: bool) -> None:
    if resume:
        return
    existing = [name for name in names if (output_dir / name).exists()]
    if existing:
        raise FileExistsError(
            f"Output directory already contains {', '.join(existing)}. "
            "Choose a new output directory for a new analysis, or use resume=True / "
            "--resume to continue the same analysis."
        )


def _upsert(rows: list[dict[str, Any]], new_row: dict[str, Any]) -> None:
    """Replace an existing record_id row or append a new one."""
    record_id = str(new_row["record_id"])
    for index, row in enumerate(rows):
        if str(row.get("record_id")) == record_id:
            rows[index] = new_row
            return
    rows.append(new_row)


def _checkpoint_row(rows: list[dict[str, Any]], new_row: dict[str, Any], path: Path) -> None:
    """Commit a row in memory only after its output table was written successfully."""
    updated_rows = list(rows)
    _upsert(updated_rows, new_row)
    _atomic_csv(updated_rows, path)
    rows[:] = updated_rows


def _json_safe(value: Any):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return [_json_safe(item) for item in list(value)]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)


def _flatten(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}{key}": _json_safe(value) for key, value in values.items()}


def _base_row(
    record: SpectrumRecord,
    output_dir: Path,
    df_spec,
    *,
    mode: str,
    clip_outliers: bool,
) -> dict[str, Any]:
    # Preserve user-supplied optional metadata, then overwrite aliases with a
    # predictable canonical schema.
    result = {str(key): _json_safe(value) for key, value in record.metadata.items()}
    result.update(
        {
            "record_id": record.record_id,
            "name": record.name,
            "spectrum": portable_spectrum_path(record.spectrum_path, output_dir),
            "redshift": record.redshift,
            "mwebv": record.mwebv,
            "phase": record.phase,
            "input_format": df_spec.attrs.get("input_format", record.input_format),
            "vexp": df_spec.attrs.get("vexp"),
            "snr_estimate": df_spec.attrs.get("snr_estimate"),
            "uncertainty_source": df_spec.attrs.get("uncertainty_source"),
            "n_estimated_uncertainties": df_spec.attrs.get("n_estimated_uncertainties"),
            "coverage_status": df_spec.attrs.get("coverage_status"),
            "clip_outliers": clip_outliers,
            "mode": mode,
        }
    )
    return result


def _prepare(record: SpectrumRecord, config: FitConfig):
    clip = _as_bool(record.metadata.get("clip_outliers"), config.clip_outliers)
    df_spec = core.prepare_spectrum(
        record.spectrum_path,
        record.redshift,
        record.mwebv,
        vexp=record.vexp if record.vexp is not None else config.vexp,
        type_of_spec=record.input_format,
        try_clip=clip,
    )
    intervals = []
    if config.features in {"si", "all"}:
        intervals.extend(core.required_rest_intervals(config))
    if config.features in {"ca", "all"}:
        from . import ca

        intervals.extend(ca.required_rest_intervals(config))
        df_spec.loc[:, "vel_ca_ref"] = ca.wl_to_vel(df_spec["rest_wl"], ca.rest_wl_ca_ref)
    validate_coverage(df_spec["rest_wl"], record.redshift, intervals)
    df_spec.attrs["coverage_status"] = describe_requirements(intervals, record.redshift)
    return df_spec, clip


def _find_background(df_spec, config: FitConfig):
    return core.find_background_regions(
        df_spec,
        config.sil6355_red_initial,
        config.sil6355_blue_initial,
        config.sil5972_blue_initial,
        v_window_kms=config.background_search_window_kms,
        dv_kms_red=config.background_search_step_kms,
        dv_kms_blue=-config.background_search_step_kms,
        max_attempts=config.background_search_attempts,
        sil6355_red_bounds_kms=config.sil6355_red_bounds_kms,
        sil6355_blue_bounds_kms=config.sil6355_blue_bounds_kms,
        sil5972_blue_bounds_kms=config.sil5972_blue_bounds_kms,
    )


def select_background(
    df_spec,
    *,
    mode: str,
    config: FitConfig,
) -> BackgroundSelection:
    """Select automatic, reviewed, or fully manual background anchors."""
    background, anchor_figure = _find_background(df_spec, config)
    if mode == "auto":
        background_figure = core.preview_background_all(
            df_spec, background, vel_width=config.vel_width
        )
        return BackgroundSelection(
            background=background,
            vel_width=config.vel_width,
            background_figure=background_figure,
            anchor_figure=anchor_figure,
        )

    background, background_figure, metrics, fit_figure, vel_width = (
        core.interactive_refine_background_and_fit(
            df_spec,
            background,
            vel_width=config.vel_width,
            n_resol_neighbors=config.n_resol_neighbors,
            k_vel12_bounds=config.k_vel12_bounds,
            k_fwhm12_bounds=config.k_fwhm12_bounds,
            manual_start=mode == "manual",
        )
    )
    return BackgroundSelection(
        background=background,
        vel_width=vel_width,
        background_figure=background_figure,
        anchor_figure=anchor_figure,
        fit_metrics=metrics,
        fit_figure=fit_figure,
    )


def _background_row(
    record: SpectrumRecord,
    output_dir: Path,
    df_spec,
    selection: BackgroundSelection,
    *,
    mode: str,
    clip_outliers: bool,
) -> dict[str, Any]:
    row = _base_row(record, output_dir, df_spec, mode=mode, clip_outliers=clip_outliers)
    row["background_vel_width"] = selection.vel_width
    for key in BACKGROUND_KEYS + BACKGROUND_STATUS_KEYS:
        row[f"background_{key}"] = selection.background.get(key)
    statuses = [selection.background.get(key) for key in BACKGROUND_STATUS_KEYS]
    row["background_qc_flag"] = int(
        any(
            status not in {"automatic", "manual", "derived_from_sil6355_blue"}
            for status in statuses
        )
    )
    return row


def _fit_spectrum(
    df_spec,
    background: dict[str, Any],
    vel_width: float,
    config: FitConfig,
    *,
    seed: int | None,
    existing_metrics: dict[str, Any] | None = None,
    existing_figure=None,
):
    if existing_metrics is None or existing_figure is None:
        metrics, fit_figure, _, _, _ = core.fit_features(
            df_spec,
            background,
            vel_width=vel_width,
            n_resol_neighbors=config.n_resol_neighbors,
            k_vel12_bounds=config.k_vel12_bounds,
            k_fwhm12_bounds=config.k_fwhm12_bounds,
        )
    else:
        metrics, fit_figure = existing_metrics, existing_figure

    mc_summary: dict[str, Any] = {"mc_mode": "disabled", "mc_n_iter": 0}
    mc_figure = None
    if config.n_iterations:
        mc_summary, mc_figure = core.fit_features_mc(
            df_spec,
            background,
            n_iter=config.n_iterations,
            vel_width=vel_width,
            n_resol_neighbors=config.n_resol_neighbors,
            k_vel12_bounds=config.k_vel12_bounds,
            k_fwhm12_bounds=config.k_fwhm12_bounds,
            mode=config.mc_mode,
            continuum_shift=config.continuum_shift,
            seed=seed,
        )
    return metrics, fit_figure, mc_summary, mc_figure


def _mc_quality(mc_summary: dict[str, Any], n_iterations: int) -> tuple[float, int]:
    if n_iterations == 0:
        return 1.0, 0
    success_counts = [
        int(value) for key, value in mc_summary.items() if key.startswith("n_success_")
    ]
    if not success_counts:
        return 0.0, 1
    fraction = sum(success_counts) / (n_iterations * len(success_counts))
    return float(fraction), int(fraction < 0.8)


def _background_from_record(
    record: SpectrumRecord, *, default_vel_width: float = 4.0
) -> tuple[dict[str, Any], float]:
    background = {}
    for key in BACKGROUND_KEYS:
        column = f"background_{key}"
        if column not in record.metadata or pd.isna(record.metadata[column]):
            raise ValueError(f"Background catalogue is missing {column}")
        background[key] = float(record.metadata[column])
        if not np.isfinite(background[key]):
            raise ValueError(f"{column} must be finite")
    for key in BACKGROUND_STATUS_KEYS:
        column = f"background_{key}"
        if column in record.metadata and pd.notna(record.metadata[column]):
            background[key] = str(record.metadata[column])
    try:
        vel_width = float(record.metadata.get("background_vel_width", default_vel_width))
    except (TypeError, ValueError) as exc:
        raise ValueError("background_vel_width must be finite and positive") from exc
    if not np.isfinite(vel_width) or vel_width <= 0:
        raise ValueError("background_vel_width must be finite and positive")
    return background, vel_width


def _save_fit_plot(
    record: SpectrumRecord,
    output_dir: Path,
    background_figure,
    fit_figure,
    mc_figure,
) -> None:
    figures = [background_figure, fit_figure]
    titles = ["Continuum", "Best fit"]
    if mc_figure is not None:
        figures.append(mc_figure)
        titles.append("Monte Carlo")
    subtitle = Path(record.spectrum).name
    if record.phase is not None:
        subtitle += f" | phase={record.phase:.2f} d"
    core.save_summary_figure(
        figures,
        output_dir / "plots" / f"{_safe_name(record.record_id)}.png",
        title=record.name,
        subtitle=subtitle,
        panel_titles=titles,
        width_per_panel=4.0,
        height=5.0,
    )


def _close_selection(selection: BackgroundSelection) -> None:
    for figure in (selection.anchor_figure, selection.background_figure, selection.fit_figure):
        if figure is not None:
            plt.close(figure)


def run_catalog(
    catalog_path: str | Path,
    output_dir: str | Path,
    *,
    config: FitConfig | None = None,
    spectra_dir: str | Path | None = None,
    stage: str = "all",
    start: int = 0,
    end: int | None = None,
    resume: bool = False,
) -> RunSummary:
    """Run background selection alone or the complete end-to-end pipeline."""
    config = config or FitConfig()
    config.validate()
    if config.features in {"ca", "all"}:
        from .combined import run_catalog as run_combined_catalog

        return run_combined_catalog(
            catalog_path, output_dir, config=config, spectra_dir=spectra_dir,
            stage=stage, start=start, end=end, resume=resume,
        )
    if stage not in {"all", "background"}:
        raise ValueError("stage must be 'all' or 'background'")
    output_dir = Path(output_dir).expanduser().resolve()
    _protect_existing_outputs(
        output_dir,
        ("backgrounds.csv", "fit_results.csv", "fit_log.csv", "background_log.csv"),
        resume,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)

    records = load_catalog(
        catalog_path,
        spectra_dir=spectra_dir,
        default_mode=config.mode,
        default_input_format=config.input_format,
    )[start:end]
    background_path = output_dir / "backgrounds.csv"
    result_path = output_dir / "fit_results.csv"
    log_path = output_dir / ("fit_log.csv" if stage == "all" else "background_log.csv")
    background_rows = _existing_rows(background_path, resume)
    result_rows = _existing_rows(result_path, resume) if stage == "all" else []
    log_rows = _existing_rows(log_path, resume)
    completed_rows = result_rows if stage == "all" else background_rows
    completed = {str(row.get("record_id")) for row in completed_rows}
    if stage == "background":
        latest_attempts = {
            str(row.get("record_id")): _as_bool(row.get("success"), False)
            for row in log_rows
        }
        completed = {record_id for record_id in completed if latest_attempts.get(record_id, False)}

    succeeded = failed = skipped = 0
    for position, record in enumerate(records, start=1):
        if record.record_id in completed:
            skipped += 1
            print(f"[{position}/{len(records)}] SKIP {record.record_id}")
            continue
        started = time.monotonic()
        selection = None
        try:
            df_spec, clip = _prepare(record, config)
            selection = select_background(df_spec, mode=record.mode, config=config)
            background_row = _background_row(
                record,
                output_dir,
                df_spec,
                selection,
                mode=record.mode,
                clip_outliers=clip,
            )
            if stage == "background":
                figures = [selection.anchor_figure, selection.background_figure]
                titles = ["Anchor search", "Continuum"]
                if selection.fit_figure is not None:
                    figures = [selection.background_figure, selection.fit_figure]
                    titles = ["Continuum", "Best fit"]
                    plt.close(selection.anchor_figure)
                core.save_summary_figure(
                    figures,
                    output_dir / "plots" / f"{_safe_name(record.record_id)}_background.png",
                    title=record.name,
                    subtitle=Path(record.spectrum).name,
                    panel_titles=titles,
                    width_per_panel=4.0,
                    height=5.0,
                )
                _checkpoint_row(background_rows, background_row, background_path)
            else:
                _checkpoint_row(background_rows, background_row, background_path)
                seed = None if config.seed is None else config.seed + record.catalog_index
                metrics, fit_figure, mc_summary, mc_figure = _fit_spectrum(
                    df_spec,
                    selection.background,
                    selection.vel_width,
                    config,
                    seed=seed,
                    existing_metrics=selection.fit_metrics,
                    existing_figure=selection.fit_figure,
                )
                result = dict(background_row)
                result["mc_seed"] = seed
                result.update(_flatten("best_fit_", metrics))
                result.update(_flatten("", mc_summary))
                result["mc_success_fraction"], result["mc_qc_flag"] = _mc_quality(
                    mc_summary, config.n_iterations
                )
                _save_fit_plot(
                    record,
                    output_dir,
                    selection.background_figure,
                    fit_figure,
                    mc_figure,
                )
                plt.close(selection.anchor_figure)
                _checkpoint_row(result_rows, result, result_path)

            elapsed = time.monotonic() - started
            log_rows.append(
                {
                    "record_id": record.record_id,
                    "name": record.name,
                    "spectrum": portable_spectrum_path(record.spectrum_path, output_dir),
                    "stage": stage,
                    "success": 1,
                    "error_type": "",
                    "error_message": "",
                    "runtime_seconds": round(elapsed, 3),
                }
            )
            succeeded += 1
            print(f"[{position}/{len(records)}] OK   {record.record_id}")
        except (KeyboardInterrupt, EOFError):
            if selection is not None:
                _close_selection(selection)
            _atomic_csv(background_rows, background_path)
            _atomic_csv(result_rows, result_path)
            _atomic_csv(log_rows, log_path)
            raise
        except Exception as exc:
            elapsed = time.monotonic() - started
            log_rows.append(
                {
                    "record_id": record.record_id,
                    "name": record.name,
                    "spectrum": portable_spectrum_path(record.spectrum_path, output_dir),
                    "stage": stage,
                    "success": 0,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "runtime_seconds": round(elapsed, 3),
                }
            )
            failed += 1
            print(f"[{position}/{len(records)}] FAIL {record.record_id}: {exc}")
        finally:
            plt.close("all")
            _atomic_csv(log_rows, log_path)

    return RunSummary(len(records), succeeded, failed, skipped, output_dir)


def fit_background_catalog(
    background_catalog: str | Path,
    output_dir: str | Path,
    *,
    config: FitConfig | None = None,
    spectra_dir: str | Path | None = None,
    start: int = 0,
    end: int | None = None,
    resume: bool = False,
) -> RunSummary:
    """Fit spectra using a previously saved/editable backgrounds.csv file."""
    config = config or FitConfig()
    config.validate()
    if config.features in {"ca", "all"}:
        from .combined import fit_background_catalog as fit_combined_background_catalog

        return fit_combined_background_catalog(
            background_catalog, output_dir, config=config, spectra_dir=spectra_dir,
            start=start, end=end, resume=resume,
        )
    output_dir = Path(output_dir).expanduser().resolve()
    _protect_existing_outputs(output_dir, ("fit_results.csv", "fit_log.csv"), resume)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)
    records = load_catalog(
        background_catalog,
        spectra_dir=spectra_dir,
        default_mode=config.mode,
        default_input_format=config.input_format,
    )[start:end]

    result_path = output_dir / "fit_results.csv"
    log_path = output_dir / "fit_log.csv"
    result_rows = _existing_rows(result_path, resume)
    log_rows = _existing_rows(log_path, resume)
    completed = {str(row.get("record_id")) for row in result_rows}
    succeeded = failed = skipped = 0

    for position, record in enumerate(records, start=1):
        if record.record_id in completed:
            skipped += 1
            print(f"[{position}/{len(records)}] SKIP {record.record_id}")
            continue
        started = time.monotonic()
        try:
            df_spec, clip = _prepare(record, config)
            background, vel_width = _background_from_record(record, default_vel_width=config.vel_width)
            seed = None if config.seed is None else config.seed + record.catalog_index
            metrics, fit_figure, mc_summary, mc_figure = _fit_spectrum(
                df_spec, background, vel_width, config, seed=seed
            )
            background_figure = core.preview_background_all(
                df_spec, background, vel_width=vel_width
            )
            result = _base_row(
                record,
                output_dir,
                df_spec,
                mode=record.mode,
                clip_outliers=clip,
            )
            result["background_vel_width"] = vel_width
            for key in BACKGROUND_KEYS + BACKGROUND_STATUS_KEYS:
                result[f"background_{key}"] = background.get(key)
            result["mc_seed"] = seed
            result.update(_flatten("best_fit_", metrics))
            result.update(_flatten("", mc_summary))
            result["mc_success_fraction"], result["mc_qc_flag"] = _mc_quality(
                mc_summary, config.n_iterations
            )
            _save_fit_plot(record, output_dir, background_figure, fit_figure, mc_figure)
            _checkpoint_row(result_rows, result, result_path)
            elapsed = time.monotonic() - started
            log_rows.append(
                {
                    "record_id": record.record_id,
                    "name": record.name,
                    "spectrum": portable_spectrum_path(record.spectrum_path, output_dir),
                    "stage": "fit",
                    "success": 1,
                    "error_type": "",
                    "error_message": "",
                    "runtime_seconds": round(elapsed, 3),
                }
            )
            succeeded += 1
            print(f"[{position}/{len(records)}] OK   {record.record_id}")
        except Exception as exc:
            elapsed = time.monotonic() - started
            log_rows.append(
                {
                    "record_id": record.record_id,
                    "name": record.name,
                    "spectrum": portable_spectrum_path(record.spectrum_path, output_dir),
                    "stage": "fit",
                    "success": 0,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "runtime_seconds": round(elapsed, 3),
                }
            )
            failed += 1
            print(f"[{position}/{len(records)}] FAIL {record.record_id}: {exc}")
        finally:
            plt.close("all")
            _atomic_csv(log_rows, log_path)

    return RunSummary(len(records), succeeded, failed, skipped, output_dir)

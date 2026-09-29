"""Catalogue validation and portable spectrum-path resolution."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MODES = {"auto", "review", "manual"}
INPUT_FORMATS = {"auto", "flux", "flux-error"}

COLUMN_ALIASES = {
    "spectrum": ("spectrum", "spec_name", "spectrum_file", "file"),
    "redshift": ("redshift", "z"),
    "mwebv": ("mwebv", "mw_ebv", "ebv"),
    "name": ("name", "object", "object_name", "ztfname", "sn_name"),
    "record_id": ("record_id", "id", "spectrum_id"),
    "phase": ("phase", "spec_phase"),
    "input_format": ("input_format", "type_of_spec", "spectrum_format"),
    "mode": ("mode", "fit_mode"),
    "vexp": ("vexp", "smoothing_width"),
}


@dataclass(frozen=True)
class SpectrumRecord:
    """One validated input spectrum and its optional metadata."""

    record_id: str
    name: str
    spectrum: str
    spectrum_path: Path
    redshift: float
    mwebv: float
    catalog_index: int
    phase: float | None = None
    input_format: str = "auto"
    mode: str = "auto"
    vexp: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _first_present(row: pd.Series, aliases: tuple[str, ...], default=None):
    for column in aliases:
        if column in row.index and pd.notna(row[column]):
            value = row[column]
            if not isinstance(value, str) or value.strip():
                return value
    return default


def _resolve_spectrum(
    value: str,
    catalog_dir: Path,
    spectra_dir: Path | None,
) -> Path:
    supplied = Path(value).expanduser()
    candidates: list[Path] = []
    if supplied.is_absolute():
        candidates.append(supplied)
    else:
        candidates.append(catalog_dir / supplied)
        if spectra_dir is not None:
            candidates.append(spectra_dir / supplied)
        candidates.append(catalog_dir / "spectra" / supplied)

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Spectrum {value!r} was not found; searched: {searched}")


def _finite_float(value, field_name: str, row_number: int) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Row {row_number}: {field_name} must be numeric, got {value!r}") from exc
    if not np.isfinite(number):
        raise ValueError(f"Row {row_number}: {field_name} must be finite")
    return number


def load_catalog(
    catalog_path: str | Path,
    *,
    spectra_dir: str | Path | None = None,
    default_mode: str = "auto",
    default_input_format: str = "auto",
) -> list[SpectrumRecord]:
    """Load a CSV catalogue using canonical or legacy column names.

    Required values are spectrum, redshift, and mwebv. ``name`` defaults to
    the spectrum filename stem. Relative spectrum paths are checked relative
    to the catalogue, an optional spectra directory, then ``catalogue/spectra``.
    """
    catalog_path = Path(catalog_path).expanduser().resolve()
    if not catalog_path.is_file():
        raise FileNotFoundError(f"Catalogue not found: {catalog_path}")
    if default_mode not in MODES:
        raise ValueError(f"default_mode must be one of {sorted(MODES)}")
    if default_input_format not in INPUT_FORMATS:
        raise ValueError(f"default_input_format must be one of {sorted(INPUT_FORMATS)}")

    spectra_path = None
    if spectra_dir is not None:
        spectra_path = Path(spectra_dir).expanduser()
        if not spectra_path.is_absolute():
            spectra_path = (Path.cwd() / spectra_path).resolve()

    # Restart keys are identifiers, even when they look numeric or like an NA token.
    table = pd.read_csv(
        catalog_path,
        comment="#",
        converters={column: str for column in COLUMN_ALIASES["record_id"]},
    )
    if table.empty:
        raise ValueError(f"Catalogue contains no spectra: {catalog_path}")

    missing = [
        canonical
        for canonical in ("spectrum", "redshift", "mwebv")
        if not any(alias in table.columns for alias in COLUMN_ALIASES[canonical])
    ]
    if missing:
        raise ValueError(
            "Catalogue is missing required column(s): "
            + ", ".join(missing)
            + ". Accepted aliases are documented in the README."
        )

    records: list[SpectrumRecord] = []
    used_ids: set[str] = set()
    for zero_index, (_, row) in enumerate(table.iterrows()):
        row_number = zero_index + 2  # CSV header occupies line 1
        spectrum = str(_first_present(row, COLUMN_ALIASES["spectrum"], "")).strip()
        if not spectrum:
            raise ValueError(f"Row {row_number}: spectrum path is missing")
        redshift = _finite_float(
            _first_present(row, COLUMN_ALIASES["redshift"]), "redshift", row_number
        )
        mwebv = _finite_float(_first_present(row, COLUMN_ALIASES["mwebv"]), "mwebv", row_number)
        if redshift <= -1:
            raise ValueError(f"Row {row_number}: redshift must be greater than -1")
        if mwebv < 0:
            raise ValueError(f"Row {row_number}: mwebv must be non-negative")

        spectrum_path = _resolve_spectrum(spectrum, catalog_path.parent, spectra_path)
        name = str(_first_present(row, COLUMN_ALIASES["name"], spectrum_path.stem)).strip()
        record_id = str(
            _first_present(
                row,
                COLUMN_ALIASES["record_id"],
                f"{name}:{spectrum_path.stem}",
            )
        ).strip()
        if record_id in used_ids:
            raise ValueError(
                f"Row {row_number}: duplicate record_id {record_id!r}; add a unique id column"
            )
        used_ids.add(record_id)

        mode = str(_first_present(row, COLUMN_ALIASES["mode"], default_mode)).strip().lower()
        if mode not in MODES:
            raise ValueError(f"Row {row_number}: mode must be one of {sorted(MODES)}, got {mode!r}")

        raw_format = (
            str(_first_present(row, COLUMN_ALIASES["input_format"], default_input_format))
            .strip()
            .lower()
        )
        format_aliases = {
            "original": "flux",
            "homogenised": "flux-error",
            "homogenized": "flux-error",
        }
        input_format = format_aliases.get(raw_format, raw_format)
        if input_format not in INPUT_FORMATS:
            raise ValueError(
                f"Row {row_number}: input_format must be one of "
                f"{sorted(INPUT_FORMATS)}, got {raw_format!r}"
            )

        phase_value = _first_present(row, COLUMN_ALIASES["phase"])
        phase = None
        if phase_value is not None:
            phase = _finite_float(phase_value, "phase", row_number)

        vexp_value = _first_present(row, COLUMN_ALIASES["vexp"])
        vexp = None
        if vexp_value is not None:
            vexp = _finite_float(vexp_value, "vexp", row_number)
            if vexp <= 0:
                raise ValueError(f"Row {row_number}: vexp must be positive")

        metadata = {
            str(column): value
            for column, value in row.items()
            if pd.notna(value) and not isinstance(value, (list, dict, tuple, set))
        }
        records.append(
            SpectrumRecord(
                record_id=record_id,
                name=name,
                spectrum=spectrum,
                spectrum_path=spectrum_path,
                redshift=redshift,
                mwebv=mwebv,
                catalog_index=zero_index,
                phase=phase,
                input_format=input_format,
                mode=mode,
                vexp=vexp,
                metadata=metadata,
            )
        )
    return records


def portable_spectrum_path(spectrum_path: Path, output_dir: Path) -> str:
    """Represent a spectrum path relative to an output catalogue when possible."""
    spectrum_path = spectrum_path.resolve()
    output_dir = output_dir.resolve()
    try:
        return spectrum_path.relative_to(output_dir).as_posix()
    except ValueError:
        import os

        try:
            return Path(os.path.relpath(spectrum_path, output_dir)).as_posix()
        except ValueError:
            # Windows cannot express a relative path between different drives.
            return spectrum_path.as_posix()

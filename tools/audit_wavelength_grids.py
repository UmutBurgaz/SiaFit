"""Compare historical fitting inputs with current spectra without editing either."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


def read_spectrum(path):
    data = Path(path).read_bytes()
    rows = []
    for line in data.decode('utf-8-sig', errors='replace').splitlines():
        if line.lstrip().startswith('#'):
            continue
        try:
            values = [float(value) for value in line.replace(',', ' ').split()[:2]]
        except ValueError:
            continue
        if len(values) == 2 and np.isfinite(values).all():
            rows.append(values)
    return np.asarray(rows), hashlib.sha256(data).hexdigest()


def compare(current_path, historical_path):
    result = dict(current_path=str(current_path), historical_path=str(historical_path))
    if not Path(historical_path).is_file():
        return {**result, 'status': 'historical_input_missing'}
    current, current_hash = read_spectrum(current_path)
    historical, historical_hash = read_spectrum(historical_path)
    result.update(current_sha256=current_hash, historical_sha256=historical_hash)
    if current.shape != historical.shape or len(current) < 8:
        return {**result, 'status': 'sampling_differs'}
    if np.any(current[:, 0] <= 0):
        return {**result, 'status': 'invalid_wavelengths'}
    ratio = historical[:, 0] / current[:, 0]
    scale = float(np.median(ratio))
    deviation = float(np.max(np.abs(ratio - scale)))
    correlation = float(np.corrcoef(current[:, 1], historical[:, 1])[0, 1])
    result.update(wavelength_scale=scale, scale_max_deviation=deviation,
                  flux_correlation=correlation)
    if deviation > 1e-5 or not np.isfinite(correlation) or correlation < .99:
        status = 'input_difference_requires_inspection'
    else:
        status = 'wavelength_scale_differs' if abs(scale - 1) > 1e-5 else 'equivalent_grid'
    return {**result, 'status': status}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('comparison', type=Path)
    parser.add_argument('historical_directory', type=Path)
    parser.add_argument('output_directory', type=Path)
    args = parser.parse_args()
    rows = pd.read_csv(args.comparison).drop_duplicates('manual_row_number')

    def audit(row):
        try:
            result = compare(row.resolved_spectrum, args.historical_directory / Path(row.manual_spec_name).name)
        except (OSError, ValueError) as exc:
            result = {'status': 'read_error', 'reason': str(exc)}
        return {'manual_row_number': row.manual_row_number, 'record_id': row.record_id, **result}

    with ThreadPoolExecutor(max_workers=4) as pool:
        report = pd.DataFrame(pool.map(audit, rows.itertuples(index=False)))
    args.output_directory.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output_directory / 'wavelength_audit.csv', index=False)
    summary = report.status.value_counts().to_dict()
    (args.output_directory / 'wavelength_audit_summary.json').write_text(
        json.dumps(summary, indent=2), encoding='utf-8',
    )
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

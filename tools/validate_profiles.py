"""Benchmark automatic primary profiles on every available equivalent input grid."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


def measure(row):
    from siafit import FitConfig, core
    from siafit.profile import select_background

    result = {key: row[key] for key in ('manual_row_number', 'record_id', 'name', 'manual', 'baseline')}
    for key in ('manual_bestfit', 'baseline_bestfit'):
        if key in row:
            result[key] = row[key]
    try:
        spectrum = core.prepare_spectrum(
            row['resolved_spectrum'], row['redshift'], row['mwebv'],
            vexp=row['vexp'], type_of_spec='flux', try_clip=row['clip_outliers'],
        )
        background, measurement = select_background(spectrum, FitConfig())
        result.update(measurement, alternatives=background.get('adaptive_candidates', '[]'))
    except (ValueError, RuntimeError, TypeError) as exc:
        result['error'] = str(exc)
    finally:
        core.plt.close('all')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('comparison', type=Path)
    parser.add_argument('automatic', type=Path)
    parser.add_argument('audit', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--manual-fits', type=Path)
    args = parser.parse_args()
    data = pd.read_csv(args.comparison)
    data = data[data.metric.eq('Si II 6355 velocity')].copy()
    audit = pd.read_csv(args.audit)
    data = data.merge(audit[['manual_row_number', 'status']], on='manual_row_number')
    data = data[data.status.eq('equivalent_grid')]
    data = data.rename(columns={'manual_median': 'manual', 'auto_median': 'baseline', 'ztfname': 'name'})
    automatic = pd.read_csv(args.automatic, low_memory=False)
    data = data.merge(automatic[['record_id', 'redshift', 'mwebv', 'vexp', 'clip_outliers',
                                'best_fit_sil6355_vel']].rename(columns={'best_fit_sil6355_vel': 'baseline_bestfit'}),
                      on='record_id', validate='many_to_one')
    if args.manual_fits:
        manual = pd.read_csv(args.manual_fits, low_memory=False)
        manual['manual_row_number'] = np.arange(2, len(manual)+2)
        data = data.merge(manual[['manual_row_number', 'best_fit_sil6355_vel']].rename(
            columns={'best_fit_sil6355_vel': 'manual_bestfit'}), on='manual_row_number', validate='one_to_one')
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(measure, data.to_dict('records'), chunksize=2):
            results.append(row)
            if len(results) % 50 == 0:
                print(f'{len(results)}/{len(data)} spectra evaluated', flush=True)
    frame = pd.DataFrame(results)
    frame.to_csv(args.output/'rows.csv', index=False)
    summaries = []
    for name, mask in [('fast', frame.manual < -17), ('ordinary', frame.manual.between(-17, -8)),
                       ('slow', frame.manual > -8)]:
        rows = frame[mask]
        before = (rows.baseline-rows.manual).abs() <= 1.5
        after = (rows.velocity-rows.manual).abs() <= 1.5
        summaries.append(dict(group=name, total=len(rows), baseline_within_1500=int(before.sum()),
                              profile_within_1500=int(after.sum()), improvements=int((~before & after).sum()),
                              regressions=int((before & ~after).sum()),
                              failures=int(rows.velocity.isna().sum()),
                              baseline_median_absolute_error=float(np.nanmedian(abs(rows.baseline-rows.manual))),
                              profile_median_absolute_error=float(np.nanmedian(abs(rows.velocity-rows.manual)))))
    (args.output/'summary.json').write_text(json.dumps(summaries, indent=2), encoding='utf-8')
    if 'manual_bestfit' in frame:
        point_summary = []
        for row in summaries:
            mask = frame.manual < -17 if row['group'] == 'fast' else (
                frame.manual > -8 if row['group'] == 'slow' else frame.manual.between(-17, -8))
            part = frame[mask]
            old = abs(part.baseline_bestfit-part.manual_bestfit) <= 1.5
            new = abs(part.velocity-part.manual_bestfit) <= 1.5
            point_summary.append(dict(group=row['group'], total=len(part), baseline_within_1500=int(old.sum()),
                                      automatic_within_1500=int(new.sum()), improvements=int((~old & new).sum()),
                                      regressions=int((old & ~new).sum())))
        (args.output/'bestfit_summary.json').write_text(json.dumps(point_summary, indent=2), encoding='utf-8')
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == '__main__':
    main()

"""Evaluate adaptive selection on a fixed stratified sample of verified inputs."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from siafit import FitConfig, core, pipeline


def main():
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('comparison', type=Path)
    parser.add_argument('automatic_results', type=Path)
    parser.add_argument('audit', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--sampling-seed', type=int, default=142)
    parser.add_argument('--exclude-sample', type=Path)
    parser.add_argument('--strategy', choices=('legacy', 'adaptive', 'profile'), default='profile')
    parser.add_argument('--per-group', type=int, default=8)
    args = parser.parse_args()
    comparison = pd.read_csv(args.comparison)
    comparison = comparison[comparison.metric.eq('Si II 6355 velocity')]
    audit = pd.read_csv(args.audit)
    comparison = comparison.merge(audit[['manual_row_number', 'status']], on='manual_row_number')
    verified = comparison[comparison.status.eq('equivalent_grid')].copy()
    if args.exclude_sample:
        prior = pd.read_csv(args.exclude_sample)
        verified = verified[~verified.ztfname.isin(prior['name'])]
    discrepancy = (verified.auto_median - verified.manual_median).abs() > 1.5
    groups = {
        'fast_discrepant': (verified.manual_median < -17) & discrepancy,
        'fast_agreement': (verified.manual_median < -17) & ~discrepancy,
        'slow_discrepant': (verified.manual_median > -8) & discrepancy,
        'slow_agreement': (verified.manual_median > -8) & ~discrepancy,
        'ordinary_agreement': verified.manual_median.between(-17, -8) & ~discrepancy,
    }
    automatic = pd.read_csv(args.automatic_results, low_memory=False).set_index('record_id')
    results = []
    cfg = FitConfig(anchor_strategy=args.strategy)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, mask in groups.items():
        sample = verified[mask].sort_values('manual_row_number')
        sample = sample.sample(n=min(args.per_group, len(sample)), random_state=args.sampling_seed)
        for row in sample.itertuples(index=False):
            record = automatic.loc[row.record_id]
            result = dict(group=name, manual_row_number=row.manual_row_number,
                          record_id=row.record_id, name=row.ztfname,
                          manual=row.manual_median, baseline=row.auto_median)
            try:
                spectrum = core.prepare_spectrum(
                    row.resolved_spectrum, record.redshift, record.mwebv,
                    vexp=record.vexp, type_of_spec='flux', try_clip=True,
                )
                background, _ = pipeline._find_background(spectrum, cfg)
                metrics, figure, *_ = core.fit_features(spectrum, background)
                result.update(velocity=metrics['sil6355_vel'],
                              rescued=background.get('adaptive_rescued', 0),
                              ambiguous=background.get('adaptive_ambiguous', 0),
                              weak_5972=metrics.get('sil5972_profile_review_flag', background.get('adaptive_weak_5972', 0)),
                              redchi=metrics['optimizer_redchi'],
                              candidates=background.get('adaptive_candidates', '[]'))
                figure.savefig(args.output / f'{row.record_id}.png', dpi=80)
            except (ValueError, RuntimeError) as exc:
                result['error'] = str(exc)
            finally:
                core.plt.close('all')
            results.append(result)
            print(json.dumps(result), flush=True)
    frame = pd.DataFrame(results)
    frame.to_csv(args.output / 'validation_rows.csv', index=False)
    summary = []
    for name, group in frame.groupby('group'):
        finite = np.isfinite(group.get('velocity', np.nan))
        previous_good = (group.baseline - group.manual).abs() <= 1.5
        now_good = finite & ((group.velocity - group.manual).abs() <= 1.5)
        summary.append(dict(group=name, total=len(group), returned=int(finite.sum()),
                            baseline_within_1500=int(previous_good.sum()),
                            adaptive_within_1500=int(now_good.sum()),
                            ambiguous=int(group.get('ambiguous', pd.Series(dtype=float)).fillna(0).sum()),
                            weak_5972=int(group.get('weak_5972', pd.Series(dtype=float)).fillna(0).sum()),
                            regressions=int((previous_good & ~now_good).sum())))
    (args.output / 'validation_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

# SiaFit

SiaFit measures Si II λ6355/λ5972 and the Ca II near-infrared triplet in supernova spectra. It supports automatic batch runs, interactive review, manual continuum selection, and separate background and fitting stages. The `all` workflow fits Si first and passes its selected λ6355 velocity directly to Ca.

## Install

Python 3.10 or newer is required. The distribution name is `siafit` and the command is `siafit`.

Once published to PyPI, installation will be:

```bash
python -m pip install siafit
```

Until then, install from a checkout of this repository:

```bash
python -m pip install -e .
```

For development, use `python -m pip install -e '.[dev]'`.

Python imports use `siafit`, for example `from siafit import FitConfig, run_catalog`.

## Bundled examples

[example_catalog.csv](example_catalog.csv) contains three spectra with sampling for both fitting regions:

| Spectrum | Instrument |
| --- | --- |
| `ZTF18aagrtxs_0_20180328_SEDM.DAT` | SEDM |
| `ZTF18abtnbys_0_20180913_DBSP.DAT` | DBSP |
| `ZTF20aatzwgk_1_20200414_SEDM.DAT` | SEDM |

The catalogue supplies each spectrum's redshift, Milky Way reddening, phase, optional smoothing width, and an example `si_velocity` prior for Ca-only runs. That prior is ignored in `all`: the fitted Si velocity is used instead. See [spectra/README.md](spectra/README.md) for observation details.

## Run a fit

```bash
siafit run example_catalog.csv --features all --output-dir results --niter 20 --seed 42
siafit run example_catalog.csv --features si --output-dir si_results
siafit run example_catalog.csv --features ca --output-dir ca_results
```

The default feature selection is `si` to preserve existing Si-only fitting. Use `--features all` for the Si II and Ca II NIR fitting.

For Ca-only data without a `si_velocity` catalogue column, pass `--si-velocity VALUE`, in units of 10³ km/s. The Ca photospheric and high-velocity priors require that value. In `all`, no prior file or intermediate results CSV is needed.

`--mode auto` uses automatic continuum anchors. `--mode review` offers interactive approval and adjustment. `--mode manual` starts with manual continuum selection. Interactive modes require a graphical Matplotlib backend.

Version 1.1.0 improves automatic fitting of fast, blue-side Si II absorption.
Use `--anchor-strategy legacy` to retain the previous automatic fitting method.

## Input spectra and catalogue

Spectra are whitespace- or comma-separated ASCII tables. The first two columns are observed wavelength in Å and flux. By default, a third column is treated as 1σ flux uncertainty; extra columns are ignored. Use `input_format=flux` in the catalogue or `--input-format flux` when the third column has another meaning. Invalid, missing, or non-positive uncertainties are estimated from the flux. Incomplete rows, including rows before the first complete pixel, are skipped.

The catalogue requires `spectrum`, `redshift`, and `mwebv` columns. `name`, `record_id`, `phase`, `mode`, `input_format`, `vexp`, and `si_velocity` are optional. Spectrum paths may be absolute, relative to the catalogue, or filenames inside its `spectra/` directory. `--spectra-dir` supplies another lookup directory. A Ca-only run must provide a finite `si_velocity` through the catalogue or command line.

```csv
spectrum,redshift,mwebv,si_velocity
my_spectrum.dat,0.031,0.018,-11.0
```

## Wavelength coverage 

`--features si` checks only Si. `--features ca` checks only Ca. `--features all` checks both intervals independently, so a gap between the two fitting regions is allowed. Within either interval, the preflight checks endpoint sampling, requires at least eight pixels, and rejects gaps larger than five times the local median pixel spacing. Successful output rows record required rest-frame and corresponding observed-frame intervals in `coverage_status`. Failures report both intervals and the sampling problem.

The coverage settings can be inspected or changed through `FitConfig`, including `sil6355_*_bounds_kms`, `sil5972_blue_bounds_kms`, `ca_*_initial`, `ca_*_bounds_kms`, `ca_background_search_window_kms`, and `vel_width`.

## Results

A run writes `backgrounds.csv`, `fit_results.csv`, `fit_log.csv`, and diagnostic PNGs under `plots/`. A background-only run writes `background_log.csv`. Si measurements use `best_fit_sil...` columns; Ca measurements use `best_fit_ca...` columns. Joint Monte Carlo summaries are separated with `si_` and `ca_` prefixes. The output also records the selected Si velocity passed into Ca, continuum anchor values and statuses, uncertainty provenance, and coverage status.

A successful run means the workflow completed; it does not guarantee that every
measurement passes scientific quality checks. Inspect `background_qc_flag`,
`mc_qc_flag`, the joint-workflow `si_`/`ca_` quality flags, successful draw counts,
and failure reasons before using the results. Flagged continuum anchors may need
review or manual selection. Do not treat missing uncertainties as valid measurements.

Velocities are in 10³ km/s; equivalent widths are in Å, FWHM are available both in 10³ km/s and Å. Ca is modeled as photospheric and high-velocity triplets. The three lines within each triplet share an amplitude, velocity, and width; the two triplets have separate parameters. The photospheric velocity is constrained to within 25% of Si II λ6355. The high-velocity component is at least 2,000 km/s faster than Si II λ6355, which does not guarantee that separation from the fitted Ca photospheric component.

### Fit diagnostics and review flags

Non-converged Ca optimizations now fail the spectrum or the individual MC draw,
as Si optimizations already do. Ca MC summaries include `failure_reasons_*`;
joint catalogues prefix those columns with `ca_`. Best fits and MC draws use the
same Ca numerical implementation and validate continuum windows, uncertainties,
and positive finite continua.

Best-fit diagnostics include `best_fit_at_bound_<parameter>` for Si and
`best_fit_ca_at_bound_<parameter>` for Ca, plus `any_at_bound` and
`covariance_available` under the same prefixes. A parameter is near a finite
bound when its distance is at most `1e-6 * max(1, abs(bound))` in that parameter's
units. Tied parameters are excluded. A finite covariance does not override a
boundary warning.

MC columns `at_bound_<parameter>_fraction_<mode>` and
`any_at_bound_fraction_<mode>` report fractions of successful draws; joint runs
add the `si_` or `ca_` prefix. With no successful draws these fractions are NaN,
not zero. Draws that converge at a bound remain in the uncertainty summaries.

`best_fit_fit_qc_flag` (Si) and `best_fit_ca_fit_qc_flag` (Ca) mark results needing
review. Their corresponding `fit_qc_reasons` columns explain boundary solutions,
missing covariance, high reduced chi-square, estimated/mixed uncertainties,
boundary-limited MC draws, or a success fraction below 80% in any requested MC
mode. Continue to inspect the separate background-quality flags as well.
These are review indicators, not automatic scientific rejection criteria.

The default reduced chi-square warning threshold is 10. Change it with
`--redchi-warning-threshold VALUE` or `FitConfig(redchi_warning_threshold=VALUE)`;
set the Python field to `None` to disable that warning. This is a configurable
heuristic, not a calibrated significance test, especially when uncertainties
were estimated or pixels are correlated. Flagged measurements remain in the CSV.

Ca MC uncertainties are conditional on the fixed selected Si velocity; they do
not propagate Si velocity uncertainty. The output records this as
`best_fit_ca_uncertainty_conditioning=fixed_si_velocity`. Resample mode retains
the existing shared full-spectrum noise realizations in joint runs.

The legacy field `ca_hvf_min_sep_kms` retains its original units of **10³ km/s**:
`2.0` means 2,000 km/s. Existing configuration fields and CSV units are preserved.
Direct low-level fitter defaults now match `FitConfig` (including 3 resolution
neighbors and 100 MC draws). Ca physical constraints are unchanged; direct
callers needing historical settings can pass them explicitly.

## Python API

```python
from siafit import FitConfig, run_catalog

summary = run_catalog(
    "example_catalog.csv",
    "results",
    config=FitConfig(features="all", mode="auto", n_iterations=20, seed=42),
)
print(summary)
```

The [automatic](examples/automatic_example.py) and [manual](examples/manual_example.py) examples are directly runnable. Shared reader, preprocessing, smoothing, and Monte Carlo helpers live in `siafit/common.py`; Si and Ca feature code lives in `core.py` and `ca.py` respectively.

## Citation and license

The software citation is in [CITATION.cff](CITATION.cff). For the Si analysis and Ca fitting methodology, see [Burgaz et al. (2025)](https://doi.org/10.1051/0004-6361/202450386) and [Maguire et al. (2014)](https://doi.org/10.1093/mnras/stu1607), respectively. The code is released under the [MIT license](LICENSE).

## Development

```bash
python -m pytest -q
python -m ruff check .
python -m build
```

## Versioning and compatibility

SiaFit 1.0 defines the supported public interface as the `siafit` command and its
`run`, `background`, and `fit` subcommands; the documented options; and the Python
exports `FitConfig`, `SpectrumRecord`, `load_catalog`, `run_catalog`, and
`fit_background_catalog`.

Within 1.x, existing command syntax, documented configuration fields, required
catalogue columns, and existing CSV measurement column names and units will remain
compatible. New optional arguments and output columns may be added; consumers
should select columns by name and tolerate extra columns. Internal modules and
undocumented helpers are not covered by this promise. Numerical results can change
when fitting bugs are corrected, and such changes will be described in release
notes. Breaking changes to the supported interface require a new major version.

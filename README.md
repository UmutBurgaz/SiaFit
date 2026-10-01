# SN Ia Features Fitter

SN Ia Features Fitter measures Si II λ6355/λ5972 and the Ca II near-infrared triplet in supernova spectra. It supports automatic batch runs, interactive review, manual continuum selection, and separate background and fitting stages. The `all` workflow fits Si first and passes its selected λ6355 velocity directly to Ca.

## Install

Python 3.10 or newer is required. From a checkout of this repository:

```bash
python -m pip install -e .
```

For development, use `python -m pip install -e '.[dev]'`.

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

Velocities are in 10³ km/s; equivalent widths are in Å, FWHM are available both in 10³ km/s and Å. Ca is modeled as photospheric and high-velocity triplets. The three lines within each triplet share an amplitude, velocity, and width; the two triplets have separate parameters. The photospheric velocity is constrained to within 25% of Si II λ6355. The high-velocity component is at least 2,000 km/s faster than Si II λ6355, which does not guarantee that separation from the fitted Ca photospheric component.

## Python API

```python
from sn_ia_features_fitter import FitConfig, run_catalog

summary = run_catalog(
    "example_catalog.csv",
    "results",
    config=FitConfig(features="all", mode="auto", n_iterations=20, seed=42),
)
print(summary)
```

The [automatic](examples/automatic_example.py) and [manual](examples/manual_example.py) examples are directly runnable. Shared reader, preprocessing, smoothing, and Monte Carlo helpers live in `sn_ia_features_fitter/common.py`; Si and Ca feature code lives in `core.py` and `ca.py` respectively.

## Citation and license

The software citation is in [CITATION.cff](CITATION.cff). For the Si analysis and Ca fitting methodology, see [Burgaz et al. (2025)](https://doi.org/10.1051/0004-6361/202450386) and [Maguire et al. (2014)](https://doi.org/10.1093/mnras/stu1607), respectively. The code is released under the [MIT license](LICENSE).

## Development

```bash
python -m pytest -q
python -m ruff check .
python -m build
```

"""Open Si II and Ca II NIR manual continuum selectors for one spectrum."""

from pathlib import Path

from sn_ia_features_fitter import FitConfig, run_catalog

ROOT = Path(__file__).resolve().parents[1]

run_catalog(
    ROOT / "example_catalog.csv",
    ROOT / "example_output" / "manual",
    config=FitConfig(features="all", mode="manual", n_iterations=20, seed=42),
    start=0,
    end=1,
)

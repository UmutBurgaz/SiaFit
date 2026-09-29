"""Fit Si II and Ca II NIR in all bundled spectra without prompts."""

from pathlib import Path

from sn_ia_features_fitter import FitConfig, run_catalog

ROOT = Path(__file__).resolve().parents[1]

run_catalog(
    ROOT / "example_catalog.csv",
    ROOT / "example_output" / "automatic",
    config=FitConfig(features="all", mode="auto", n_iterations=20, seed=42),
)

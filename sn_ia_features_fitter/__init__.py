"""Public API for Si II and Ca II NIR feature fitting."""

from .catalog import SpectrumRecord, load_catalog
from .config import FitConfig

__all__ = [
    "FitConfig",
    "SpectrumRecord",
    "fit_background_catalog",
    "load_catalog",
    "run_catalog",
]

__version__ = "0.2.0"


def run_catalog(*args, **kwargs):
    """Lazily import and run the end-to-end/background workflow."""
    from .pipeline import run_catalog as _run_catalog

    return _run_catalog(*args, **kwargs)


def fit_background_catalog(*args, **kwargs):
    """Lazily import and fit a saved background catalogue."""
    from .pipeline import fit_background_catalog as _fit_background_catalog

    return _fit_background_catalog(*args, **kwargs)

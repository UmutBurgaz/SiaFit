from pathlib import Path

import pytest

from sn_ia_features_fitter import FitConfig
from sn_ia_features_fitter.cli import build_parser


@pytest.mark.parametrize(
    "settings",
    [
        {"vexp": float("nan")},
        {"vel_width": float("inf")},
        {"background_search_window_kms": float("nan")},
        {"background_search_step_kms": 0.0},
        {"sil6355_red_initial": float("nan")},
        {"sil6355_blue_initial": -300.0},
        {"sil5972_blue_initial": float("inf")},
        {"continuum_shift": float("nan")},
        {"k_vel12_bounds": (1.3, 0.6)},
        {"k_fwhm12_bounds": (0.0, 1.25)},
        {"k_vel12_bounds": (0.6, float("inf"))},
        {"sil6355_red_bounds_kms": (0.0, 0.0)},
        {"sil6355_blue_bounds_kms": (-40000.0,)},
        {"sil5972_blue_bounds_kms": (-400000.0, -10000.0)},
        {"background_search_attempts": 2.5},
        {"n_resol_neighbors": 1},
        {"n_iterations": 2.5},
        {"n_iterations": True},
        {"seed": -1},
    ],
)
def test_invalid_configuration_fails_before_fitting(settings):
    with pytest.raises(ValueError, match=next(iter(settings))):
        FitConfig(**settings).validate()


def test_default_and_disabled_mc_configurations_are_valid():
    FitConfig().validate()
    FitConfig(n_iterations=0, background_search_attempts=0, continuum_shift=0.0, seed=0).validate()


def test_ca_resolution_neighbors_default_matches_production_runners():
    config = FitConfig(features="ca")
    config.validate()
    assert config.ca_n_resol_neighbors == 3


@pytest.mark.parametrize("command", ["run", "background", "fit"])
def test_cli_default_output_directory(command):
    args = build_parser().parse_args([command, "catalog.csv"])
    assert args.output_dir == Path("sn_ia_fit_results")

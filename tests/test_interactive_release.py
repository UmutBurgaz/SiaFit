from pathlib import Path

import pandas as pd
import pytest
from matplotlib.figure import Figure

from siafit import FitConfig, ca, core, load_catalog, pipeline, run_catalog

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("mode", ["review", "manual"])
def test_joint_interactive_workflow_with_scripted_user_events(tmp_path, monkeypatch, mode):
    """Exercise real fits and persistence; only window display and user events are scripted."""
    config = FitConfig(features="all", mode=mode, n_iterations=0, seed=42)
    record = load_catalog(ROOT / "example_catalog.csv", default_mode=mode)[0]
    spectrum, _ = pipeline._prepare(record, config)
    si_background, si_figure = pipeline._find_background(spectrum, config)
    ca_background, ca_figure = ca.find_background_regions(
        spectrum, ca_red_side_init=config.ca_red_initial,
        ca_blue_side_init=config.ca_blue_initial,
    )
    core.plt.close(si_figure)
    core.plt.close(ca_figure)
    si_clicks = iter([
        core.vel_to_wl(si_background[key], rest)
        for key, rest in (
            ("sil6355_red_side", core.rest_wl_sil1_1),
            ("sil6355_blue_side", core.rest_wl_sil1_1),
            ("sil5972_red_side", core.rest_wl_sil2_1),
            ("sil5972_blue_side", core.rest_wl_sil2_1),
        )
    ])
    ca_clicks = iter([
        ca.vel_to_wl(ca_background[key], ca.rest_wl_ca_ref)
        for key in ("ca_red_side", "ca_blue_side")
    ])
    prompts = []

    def accept(prompt):
        prompts.append(prompt)
        assert len(prompts) <= 10, "Unexpected retry loop"
        return "y"

    monkeypatch.setattr("builtins.input", accept)
    monkeypatch.setattr(core.plt, "get_backend", lambda: "test-interactive")
    monkeypatch.setattr(Figure, "show", lambda *args, **kwargs: None)
    monkeypatch.setattr(core.plt, "pause", lambda *args, **kwargs: None)
    monkeypatch.setattr(Figure, "ginput", lambda *args, **kwargs: [(next(si_clicks), 1.0)])
    monkeypatch.setattr(ca.plt, "ginput", lambda *args, **kwargs: [(next(ca_clicks), 1.0)])
    summary = run_catalog(ROOT / "example_catalog.csv", tmp_path, config=config, end=1)
    assert summary.succeeded == 1 and summary.failed == 0
    results = pd.read_csv(tmp_path / "fit_results.csv")
    assert results.loc[0, "selected_si_velocity"] == results.loc[0, "best_fit_sil6355_vel"]
    assert len(list((tmp_path / "plots").glob("*.png"))) == 1
    assert any("Accept fit" in prompt for prompt in prompts)
    assert any("Ca II NIR fit" in prompt for prompt in prompts)
    if mode == "manual":
        backgrounds = pd.read_csv(tmp_path / "backgrounds.csv")
        status_columns = [c for c in backgrounds if "status_" in c and "side" in c]
        assert status_columns
        assert all(backgrounds.loc[0, c] == "manual" for c in status_columns)

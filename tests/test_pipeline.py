from pathlib import Path

import pandas as pd
import pytest

from sn_ia_features_fitter import FitConfig, fit_background_catalog, pipeline, run_catalog
from sn_ia_features_fitter.catalog import load_catalog

ROOT = Path(__file__).resolve().parents[1]


def test_end_to_end_and_two_stage_workflows(tmp_path):
    automatic_dir = tmp_path / "automatic"
    config = FitConfig(
        mode="auto",
        input_format="flux",
        n_iterations=0,
        seed=7,
    )
    summary = run_catalog(
        ROOT / "example_catalog.csv",
        automatic_dir,
        config=config,
        start=0,
        end=1,
    )
    assert summary.succeeded == 1
    assert summary.failed == 0
    assert (automatic_dir / "backgrounds.csv").is_file()
    assert (automatic_dir / "fit_results.csv").is_file()
    assert len(list((automatic_dir / "plots").glob("*.png"))) == 1

    results = pd.read_csv(automatic_dir / "fit_results.csv")
    assert results.loc[0, "mc_mode"] == "disabled"
    assert results.loc[0, "mc_qc_flag"] == 0
    assert results.loc[0, "uncertainty_source"] == "estimated"
    assert results.loc[0, "n_estimated_uncertainties"] > 0

    second_stage_dir = tmp_path / "second-stage"
    second_summary = fit_background_catalog(
        automatic_dir / "backgrounds.csv",
        second_stage_dir,
        config=config,
    )
    assert second_summary.succeeded == 1
    assert second_summary.failed == 0
    assert (second_stage_dir / "fit_results.csv").is_file()


@pytest.fixture
def isolated_catalog(tmp_path, monkeypatch):
    """Exercise batch persistence without repeating the numerical integration test."""
    spectrum = tmp_path / "one.dat"
    spectrum.write_text("5000 1\n6000 2\n", encoding="utf-8")
    background = {key: -10.0 for key in pipeline.BACKGROUND_KEYS}
    background.update({key: "automatic" for key in pipeline.BACKGROUND_STATUS_KEYS})
    catalog = tmp_path / "catalog.csv"
    pd.DataFrame(
        [
            {
                "record_id": record_id,
                "name": f"SN{record_id}",
                "spectrum": spectrum.name,
                "redshift": 0,
                "mwebv": 0,
                **{f"background_{key}": value for key, value in background.items()},
            }
            for record_id in ("001", "002")
        ]
    ).to_csv(catalog, index=False)
    monkeypatch.setattr(pipeline, "_prepare", lambda *args: (pd.DataFrame(), True))
    monkeypatch.setattr(
        pipeline,
        "select_background",
        lambda *args, **kwargs: pipeline.BackgroundSelection(background, 4.0, None),
    )
    monkeypatch.setattr(
        pipeline,
        "_fit_spectrum",
        lambda *args, **kwargs: ({}, None, {"mc_mode": "disabled", "mc_n_iter": 0}, None),
    )
    monkeypatch.setattr(pipeline.core, "preview_background_all", lambda *args, **kwargs: None)
    return catalog


@pytest.mark.parametrize("stage", ["all", "background", "fit"])
def test_failed_plot_is_retried_on_resume(isolated_catalog, tmp_path, monkeypatch, stage):
    output = tmp_path / "output"

    def save_plot(record, *args, **kwargs):
        if record.record_id == "001":
            raise OSError("simulated plot write failure")

    def save_background(*args, **kwargs):
        if kwargs["title"] == "SN001":
            raise OSError("simulated plot write failure")

    monkeypatch.setattr(pipeline, "_save_fit_plot", save_plot)
    monkeypatch.setattr(pipeline.core, "save_summary_figure", save_background)
    runner = fit_background_catalog if stage == "fit" else run_catalog
    options = {} if stage == "fit" else {"stage": stage}
    config = FitConfig(n_iterations=0)
    summary = runner(isolated_catalog, output, config=config, **options)
    assert (summary.succeeded, summary.failed) == (1, 1)
    checkpoint = output / ("backgrounds.csv" if stage == "background" else "fit_results.csv")
    assert pd.read_csv(checkpoint, dtype={"record_id": str})["record_id"].tolist() == ["002"]

    monkeypatch.setattr(pipeline, "_save_fit_plot", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline.core, "save_summary_figure", lambda *args, **kwargs: None)
    resumed = runner(isolated_catalog, output, config=config, resume=True, **options)
    assert (resumed.succeeded, resumed.failed, resumed.skipped) == (1, 0, 1)
    assert set(pd.read_csv(checkpoint, dtype={"record_id": str})["record_id"]) == {"001", "002"}


def test_background_resume_retries_row_logged_as_failed(isolated_catalog, tmp_path, monkeypatch):
    output = tmp_path / "output"
    output.mkdir()
    # Older checkpoints could contain an anchor row whose plot never saved.
    pd.DataFrame([{"record_id": "001"}, {"record_id": "002"}]).to_csv(
        output / "backgrounds.csv", index=False
    )
    pd.DataFrame(
        [{"record_id": "001", "success": 0}, {"record_id": "002", "success": 1}]
    ).to_csv(output / "background_log.csv", index=False)
    monkeypatch.setattr(pipeline.core, "save_summary_figure", lambda *args, **kwargs: None)

    summary = run_catalog(
        isolated_catalog, output, config=FitConfig(n_iterations=0), stage="background", resume=True
    )
    assert (summary.succeeded, summary.failed, summary.skipped) == (1, 0, 1)


def test_checkpoint_write_failure_does_not_insert_row(tmp_path, monkeypatch):
    rows = [{"record_id": "old"}]

    def fail(*args, **kwargs):
        raise OSError("simulated CSV write failure")

    monkeypatch.setattr(pipeline, "_atomic_csv", fail)
    with pytest.raises(OSError):
        pipeline._checkpoint_row(rows, {"record_id": "new"}, tmp_path / "results.csv")
    assert rows == [{"record_id": "old"}]


def test_plot_filenames_do_not_collide_or_use_windows_device_names():
    identifiers = ["a:b", "a_b", "A_B", "x" * 150 + "1", "x" * 150 + "2", "CON"]
    names = [pipeline._safe_name(identifier) for identifier in identifiers]
    assert len({name.casefold() for name in names}) == len(identifiers)
    assert all(len(name) <= 140 for name in names)
    assert names[-1].split(".")[0].casefold() != "con"


def test_fit_uses_configured_width_when_catalog_has_no_width(isolated_catalog, tmp_path, monkeypatch):
    widths = []

    def fit(df_spec, background, vel_width, config, **kwargs):
        widths.append(vel_width)
        return {}, None, {"mc_mode": "disabled", "mc_n_iter": 0}, None

    monkeypatch.setattr(pipeline, "_fit_spectrum", fit)
    monkeypatch.setattr(pipeline, "_save_fit_plot", lambda *args, **kwargs: None)
    summary = fit_background_catalog(
        isolated_catalog, tmp_path / "output", config=FitConfig(n_iterations=0, vel_width=6.0)
    )
    assert summary.succeeded == 2
    assert widths == [6.0, 6.0]


@pytest.mark.parametrize("width", [0, -1, float("inf"), float("nan"), "bad"])
def test_background_width_must_be_finite_and_positive(isolated_catalog, width):
    record = load_catalog(isolated_catalog)[0]
    record.metadata["background_vel_width"] = width
    with pytest.raises(ValueError, match="background_vel_width must be finite and positive"):
        pipeline._background_from_record(record)


def test_saved_background_width_overrides_fallback(isolated_catalog):
    record = load_catalog(isolated_catalog)[0]
    record.metadata["background_vel_width"] = 2.5
    _, width = pipeline._background_from_record(record, default_vel_width=6.0)
    assert width == 2.5


@pytest.mark.parametrize("stage", ["all", "background", "fit"])
def test_fresh_run_protects_existing_outputs(isolated_catalog, tmp_path, stage):
    output = tmp_path / "output"
    output.mkdir()
    checkpoint = output / ("backgrounds.csv" if stage == "background" else "fit_results.csv")
    original = b"record_id,measurement\n001,123\n"
    checkpoint.write_bytes(original)
    runner = fit_background_catalog if stage == "fit" else run_catalog
    options = {} if stage == "fit" else {"stage": stage}
    with pytest.raises(FileExistsError, match="Choose a new output directory"):
        runner(isolated_catalog, output, config=FitConfig(n_iterations=0), **options)
    assert checkpoint.read_bytes() == original

from pathlib import Path

import pandas as pd
import pytest

from siafit.catalog import load_catalog, portable_spectrum_path

ROOT = Path(__file__).resolve().parents[1]


def test_example_catalog_loads_all_public_spectra():
    records = load_catalog(ROOT / "example_catalog.csv")
    assert len(records) == 3
    assert all(record.spectrum_path.is_file() for record in records)
    assert records[0].name == "ZTF18aagrtxs"


def test_legacy_aliases_and_absolute_spectrum_path(tmp_path):
    spectrum = tmp_path / "one.dat"
    spectrum.write_text("5000 1\n6000 2\n", encoding="utf-8")
    catalog = tmp_path / "legacy.csv"
    pd.DataFrame([{"spec_name": str(spectrum), "z": 0.02, "mwebv": 0.01, "ztfname": "SN"}]).to_csv(
        catalog, index=False
    )

    [record] = load_catalog(catalog)
    assert record.name == "SN"
    assert record.redshift == pytest.approx(0.02)
    assert record.input_format == "auto"


def test_missing_required_column_has_clear_error(tmp_path):
    catalog = tmp_path / "bad.csv"
    pd.DataFrame([{"spectrum": "a.dat", "redshift": 0.02}]).to_csv(catalog, index=False)
    with pytest.raises(ValueError, match="mwebv"):
        load_catalog(catalog)


@pytest.mark.parametrize("id_column", ["record_id", "id", "spectrum_id"])
def test_catalog_preserves_exact_restart_ids(tmp_path, id_column):
    spectrum = tmp_path / "one.dat"
    spectrum.write_text("5000 1\n6000 2\n", encoding="utf-8")
    identifiers = ["001", "1", "1e3", "NA"]
    catalog = tmp_path / "ids.csv"
    pd.DataFrame(
        [
            {id_column: value, "spectrum": spectrum.name, "redshift": 0, "mwebv": 0}
            for value in identifiers
        ]
    ).to_csv(catalog, index=False)

    assert [record.record_id for record in load_catalog(catalog)] == identifiers


def test_portable_spectrum_path_uses_relative_forward_slashes(tmp_path):
    spectrum = tmp_path / "spectra" / "one.dat"
    assert portable_spectrum_path(spectrum, tmp_path / "results") == "../spectra/one.dat"


def test_portable_spectrum_path_falls_back_when_drives_differ(tmp_path, monkeypatch):
    spectrum = tmp_path / "spectra" / "one.dat"

    def different_drives(*args, **kwargs):
        raise ValueError("path is on another mount")

    monkeypatch.setattr("os.path.relpath", different_drives)
    assert portable_spectrum_path(spectrum, tmp_path / "results") == spectrum.resolve().as_posix()

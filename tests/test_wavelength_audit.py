"""Detect coordinate changes without changing historical measurements."""

import numpy as np

from tools.audit_wavelength_grids import compare


def test_scaled_grid_is_detected_even_with_identical_flux(tmp_path):
    current = tmp_path / 'current.dat'
    historical = tmp_path / 'historical.dat'
    wavelength = np.linspace(5000, 7000, 30)
    flux = 1 + np.sin(wavelength / 100)
    np.savetxt(current, np.column_stack((wavelength, flux)))
    np.savetxt(historical, np.column_stack((wavelength * 1.03, flux)))
    original = historical.read_bytes()
    result = compare(current, historical)
    assert result['status'] == 'wavelength_scale_differs'
    assert abs(result['wavelength_scale'] - 1.03) < 1e-10
    assert result['current_sha256'] != result['historical_sha256']
    assert historical.read_bytes() == original


def test_different_flux_cannot_be_certified_as_a_coordinate_change(tmp_path):
    current, historical = tmp_path / 'a.dat', tmp_path / 'b.dat'
    w = np.arange(5000, 6000, 10)
    np.savetxt(current, np.column_stack((w, np.sin(w / 100))))
    np.savetxt(historical, np.column_stack((w * 1.03, np.cos(w / 100))))
    assert compare(current, historical)['status'] == 'input_difference_requires_inspection'

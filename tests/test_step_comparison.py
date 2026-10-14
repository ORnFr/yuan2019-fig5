"""Check nested-grid comparison and equal physical marker spacing."""

from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from yuan2019_fig5.step_comparison import compare_step_sizes, make_step_comparison
from yuan2019_fig5.plot import load_trajectory
import matplotlib.pyplot as plt


def sample_curves():
    """Return simple nested curves with known errors at fine-only times."""
    coarse_t = .01 * np.arange(1001)
    fine_t = .005 * np.arange(2001)
    exact_coarse = {"t": coarse_t, "z0": np.zeros(1001)}
    exact_fine = {"t": fine_t, "z0": np.zeros(2001)}
    coarse = {"t": coarse_t, "z0": np.full(1001, .008)}
    fine = {"t": fine_t, "z0": np.full(2001, .004)}
    fine["z0"][7] = -.009
    return exact_coarse, coarse, exact_fine, fine


def test_native_fine_peak_and_common_difference():
    """Do not miss a fine-only peak by subsampling the error statistics."""
    arrays, metrics = compare_step_sizes(*sample_curves())
    assert metrics["fine"]["max_absolute_error"] == .009
    assert metrics["fine"]["time_at_max_absolute_error"] == .035
    assert metrics["fine_common_max_absolute_error"] == .004
    assert metrics["max_step_difference"] == .004
    np.testing.assert_array_equal(arrays["fine_minus_coarse_common"], np.full(1001, -.004))
    figure = make_step_comparison(arrays)
    try:
        axis = figure.axes[0]
        np.testing.assert_array_equal(axis.lines[1].get_xdata(), axis.lines[2].get_xdata())
        inset = axis.child_axes[0]
        np.testing.assert_array_equal(inset.lines[0].get_ydata(), arrays["coarse_error"])
        np.testing.assert_array_equal(inset.lines[2].get_ydata(), arrays["fine_error"])
        assert len(inset.lines[2].get_ydata()) == 2001
    finally:
        plt.close(figure)


def test_nonnested_grid_is_rejected():
    """Do not silently interpolate shifted fine data onto the old grid."""
    curves = sample_curves()
    curves[2]["t"] += .001
    curves[3]["t"] = curves[2]["t"].copy()
    with pytest.raises(ValueError, match="nest"):
        compare_step_sizes(*curves)


def test_fine_grid_reader(tmp_path):
    """Accept 2001 float64/complex128 states only with an explicit dt=.005."""
    data = {"t": .005 * np.arange(2001), "z0": np.ones(2001),
            "rho": np.tile(np.diag([1., 0, 0, 0]).astype(np.complex128), (2001, 1, 1)),
            "theta": np.zeros((2001, 6))}
    path = tmp_path / "fine.npz"
    np.savez_compressed(path, **data)
    loaded, _ = load_trajectory(path, variational=True, dt=.005)
    np.testing.assert_array_equal(loaded["t"], data["t"])
    with pytest.raises(ValueError):
        load_trajectory(path, variational=True)

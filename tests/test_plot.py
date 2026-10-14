"""Verify signed full-grid metrics, observable consistency and plot data selection."""

from pathlib import Path
import sys
import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from yuan2019_fig5.plot import load_trajectory, compare_trajectories, make_figure5
import matplotlib.pyplot as plt


def test_full_grid_signed_error_and_marker_thinning():
    """An error peak skipped by markers must remain in metrics and the inset."""
    times = .01 * np.arange(1001)
    exact = {"t": times, "z0": np.zeros(1001)}
    variational = {"t": times, "z0": np.zeros(1001)}
    variational["z0"][7], variational["z0"][8] = -.02, .01
    arrays, metrics = compare_trajectories(exact, variational)
    assert metrics["max_absolute_error"] == .02 and metrics["max_error_index"] == 7
    assert metrics["signed_error_at_max_absolute_error"] == -.02
    assert not metrics["strict_target_met"]
    assert metrics["rms_error"] == np.sqrt((.02**2 + .01**2) / 1001)
    figure = make_figure5(arrays, marker_stride=15)
    try:
        axis = figure.axes[0]
        inset = axis.child_axes[0]
        np.testing.assert_array_equal(axis.lines[0].get_ydata(), exact["z0"])
        assert len(axis.lines[1].get_ydata()) < 1001
        assert axis.lines[1].get_xdata()[[0, -1]].tolist() == [0, 10]
        np.testing.assert_array_equal(inset.lines[0].get_ydata(), arrays["error"])
        low, high = inset.get_ylim()
        assert low < -.02 and high > .01
    finally:
        plt.close(figure)


def test_target_is_strict_and_grids_cannot_be_shifted():
    """Exactly .01 fails the target; time alignment cannot be interpolated away."""
    exact = {"t": np.array([0., 1.]), "z0": np.zeros(2)}
    other = {"t": np.array([0., 1.]), "z0": np.array([.01, 0.])}
    assert not compare_trajectories(exact, other)[1]["strict_target_met"]
    other["t"] += .01
    with pytest.raises(ValueError, match="identical"):
        compare_trajectories(exact, other)


def test_saved_baselines_and_independent_observable_formula():
    """Compare all saved values with an explicit four-index trace of Z0 rho."""
    exact, _ = load_trajectory(PROJECT / "results/exact_20261013_220047_446893/exact.npz")
    variational, _ = load_trajectory(PROJECT / "results/vqs_20261014_161232_948193/vqs.npz", variational=True)
    operator = np.diag([1, 1, -1, -1])
    references = []
    for trajectory in (exact, variational):
        values = np.array([sum(operator[i, j] * density[j, i] for i in range(4) for j in range(4))
                           for density in trajectory["rho"]])
        np.testing.assert_allclose(values.real, trajectory["z0"], atol=1e-12, rtol=0)
        references.append(values.real)
    arrays, metrics = compare_trajectories(exact, variational)
    np.testing.assert_allclose(arrays["error"], references[1] - references[0], atol=1e-12, rtol=0)
    assert metrics["points"] == 1001
    assert metrics["strict_target_met"]


@pytest.mark.parametrize("mutation", ["grid", "z0", "rho", "precision"])
def test_corrupt_saved_trajectory_is_rejected(tmp_path, mutation):
    """Reject shifted grids, mismatched observables, invalid states and lower precision."""
    with np.load(PROJECT / "results/exact_20261013_220047_446893/exact.npz", allow_pickle=False) as original:
        data = {key: original[key].copy() for key in ("t", "rho", "z0")}
    if mutation == "grid":
        data["t"] += .01
    elif mutation == "z0":
        data["z0"][20] += .02
    elif mutation == "rho":
        data["rho"][20] *= 2
    else:
        data["rho"] = data["rho"].astype(np.complex64)
    path = tmp_path / "corrupt.npz"
    np.savez_compressed(path, **data)
    with pytest.raises(ValueError):
        load_trajectory(path)

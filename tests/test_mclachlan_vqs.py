"""Scientific checks for reduced-density equations and parameter Euler evolution."""

from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from yuan2019_fig5.mclachlan import evaluate_equations, solve_svd
from yuan2019_fig5.mclachlan_validation import check_equations, check_zero_solve, check_short_evolution


def test_independent_equations_and_density_lstsq():
    """Compare full-gate derivatives, QuTiP generator, trace products and direct LS."""
    check_equations()


def test_analytic_initial_solve():
    """The rank-two initial solution must preserve zero/redundant directions."""
    check_zero_solve()


def test_svd_inconsistent_and_zero_systems():
    """Check cutoff selection, minimum norm and honest unsatisfied residuals."""
    matrix = np.diag([4., 2., 1e-16, 0., 0., 0.])
    vector = np.array([8., -2., 3., 0., 0., 0.])
    velocity, info = solve_svd(matrix, vector)
    np.testing.assert_allclose(velocity, [2, -1, 0, 0, 0, 0], atol=1e-15, rtol=0)
    assert info["rank"] == 2 and info["retained_condition"] == 2
    assert np.isinf(info["full_condition"]) and info["linear_residual"] == 3
    velocity, info = solve_svd(np.zeros((6, 6)), np.ones(6))
    np.testing.assert_array_equal(velocity, np.zeros(6))
    assert info["rank"] == 0 and np.isinf(info["retained_condition"])
    assert info["linear_residual"] == np.sqrt(6)
    velocity, info = solve_svd(np.eye(6), np.arange(6.))
    np.testing.assert_array_equal(velocity, np.arange(6.))
    assert info["rank"] == 6 and info["full_condition"] == 1


def test_short_euler_alignment_and_sensitivity():
    """Verify densities at their saved parameter/time and short diagnostic runs."""
    check_short_evolution()


@pytest.mark.parametrize("theta,error", [(np.zeros(5), ValueError),
    (np.zeros(6, dtype=complex), TypeError), (np.full(6, np.nan), ValueError)])
def test_equation_parameter_domain(theta, error):
    """Reject incompatible parameters before an eager or compiled evaluation."""
    with pytest.raises(error):
        evaluate_equations(theta)

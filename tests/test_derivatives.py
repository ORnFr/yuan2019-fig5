"""Phase 4 Jacobian checks with analytic and finite-difference references."""

from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from yuan2019_fig5.derivatives import density_jacobian
from yuan2019_fig5.derivatives_validation import (
    check_zero_jacobian, check_derivative_samples, check_jvp_and_jit,
)


def test_zero_parameter_analytic_jacobian():
    """Check all six zero-point matrices and their exact redundant/zero directions."""
    check_zero_jacobian()


def test_product_rule_finite_difference_and_tangent_properties():
    """Check 17 cases, six parameter axes, four FD steps, Hermiticity and trace."""
    check_derivative_samples()


def test_directional_derivative_and_jit():
    """Compare the Jacobian contraction to a complex-output JVP and compiled output."""
    check_jvp_and_jit()


@pytest.mark.parametrize("theta,error", [(np.zeros(5), ValueError),
                                         (np.zeros((2, 3)), ValueError),
                                         (np.zeros(6, dtype=complex), TypeError)])
def test_invalid_parameter_domain(theta, error):
    """Reject ambiguous axes and complex inputs rather than assuming holomorphicity."""
    with pytest.raises(error):
        density_jacobian(theta)

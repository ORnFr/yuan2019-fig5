"""Phase 3 checks with independent matrices and analytic reduced states."""

from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from yuan2019_fig5.ansatz import purification_state, reduce_system_density
from yuan2019_fig5.ansatz_validation import (
    sample_parameters, check_gate_conventions, check_partial_trace_examples,
    check_ansatz_states, check_wire_mapping_and_jit,
)


def test_gate_angles_and_control():
    """Check full RX/CRX/RZZ matrices and their action on every basis state."""
    check_gate_conventions()


def test_partial_trace():
    """Compare analytic examples, full-density summation and QuTiP ptrace."""
    check_partial_trace_examples()


def test_full_ansatz_and_physical_states():
    """Compare the full Figure 4 circuit with dense 16x16 construction."""
    check_ansatz_states(sample_parameters())


def test_wire_mapping_and_forward_jit():
    """Keep ancilla/system labels and bit significance consistent."""
    check_wire_mapping_and_jit()


@pytest.mark.parametrize("theta,error", [(np.zeros(5), ValueError),
                                         (np.zeros((2, 3)), ValueError),
                                         (np.zeros(6, dtype=complex), TypeError)])
def test_invalid_parameter_shape_or_type(theta, error):
    """Reject ambiguous parameter shapes and complex parameter domains."""
    with pytest.raises(error):
        purification_state(theta)


@pytest.mark.parametrize("state", [np.zeros(8), np.zeros((4, 4))])
def test_invalid_purification_shape(state):
    """Do not silently reinterpret an incompatible state layout."""
    with pytest.raises(ValueError):
        reduce_system_density(state)

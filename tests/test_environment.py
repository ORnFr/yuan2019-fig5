"""Numerical environment checks with independent analytic/matrix references."""

import importlib.util
from pathlib import Path

# Load the standalone script by path without requiring an installed package.
script_path = Path(__file__).resolve().parents[1] / "scripts" / "check_environment.py"
spec = importlib.util.spec_from_file_location("check_environment", script_path)
environment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(environment)


def test_actual_cpu_precision():
    """Small increments must survive on real and complex CPU arrays."""
    environment.check_precision()


def test_circuit_and_automatic_differentiation():
    """Check analytic values, complex Jacobians, finite differences and JIT."""
    environment.check_rotation_and_ad()


def test_entangled_state():
    """Check a two-qubit circuit against the analytic Bell state."""
    environment.check_bell_circuit()


def test_qutip_and_matplotlib():
    """Check installed matrix arithmetic and headless PNG rendering."""
    environment.check_qutip_and_plotting()

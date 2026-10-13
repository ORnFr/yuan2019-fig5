"""Phase 2 scientific checks: independent matrices, analytic limits and dynamics."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from yuan2019_fig5.exact import evolve_exact
from yuan2019_fig5.validation import check_model_and_generator, check_analytic_damping, check_full_evolution


def test_model_and_generator():
    """Check complete matrices and lowering direction, not only visual trends."""
    check_model_and_generator()


def test_analytic_amplitude_damping():
    """Excited population must decay at the specified unit rate."""
    check_analytic_damping()


def test_full_exact_evolution():
    """Check physical constraints and all time points against independent evolution."""
    check_full_evolution(*evolve_exact())

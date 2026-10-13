"""Independent QuTiP reference evolution; no variational inputs are read."""

from .model import build_model, qutip
import numpy as np

SOLVER_OPTIONS = {
    "method": "dop853", "atol": 1e-12, "rtol": 1e-10, "nsteps": 2500,
    "normalize_output": False, "store_states": True, "progress_bar": "",
}


def solve_lindblad(hamiltonian, initial_density, collapse_operators, times, *, atol=1e-12, rtol=1e-10):
    r"""Solve d rho/dt = -i[H,rho] + sum_j D[c_j](rho) with mesolve.

    D[c](rho) = c rho c^dagger - {c^dagger c,rho}/2.
    Input Qobjs: square H/rho, compatible collapse operators; times is a
    strictly increasing float64 array (N,) starting at zero. Units hbar=1.
    Return complex128 density matrices (N,d,d); no renormalization,
    Hermitian projection or eigenvalue clipping is performed. The adaptive
    DOP853 internal steps differ from the requested output-time spacing.
    atol/rtol are local ODE tolerances, not certified global error bounds.
    """
    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1 or len(times) < 2 or times[0] != 0 or not np.all(np.diff(times) > 0):
        raise ValueError("times must be increasing, start at zero and contain at least two points.")
    if not np.all(np.isfinite(times)) or atol <= 0 or rtol <= 0:
        raise ValueError("Use finite times and positive solver tolerances.")
    options = dict(SOLVER_OPTIONS, atol=atol, rtol=rtol)
    result = qutip.mesolve(hamiltonian, initial_density, times,
                          c_ops=collapse_operators, e_ops=[], options=options)
    densities = np.stack([state.full() for state in result.states])
    if densities.shape != (len(times), *initial_density.shape) or densities.dtype != np.complex128:
        raise RuntimeError("Unexpected density shape or precision.")
    if not np.all(np.isfinite(densities)):
        raise RuntimeError("Evolution produced NaN/Inf.")
    return densities


def observable_values(densities, observable):
    r"""Return Re Tr(O rho_t), checking the discarded imaginary component.

    Input: complex128 densities (N,d,d), Hermitian Qobj observable (d,d).
    Output: float64 expectations (N,). Values are not clipped to [-1,1].
    """
    values = np.einsum("ij,tji->t", observable.full(), densities)
    if np.max(np.abs(values.imag)) > 1e-12:
        raise RuntimeError("Hermitian expectation has an unexpected imaginary component.")
    return values.real


def evolve_exact():
    """Return t (1001,), rho (1001,4,4), z0 (1001,) on t_n=0.01*n."""
    times = 0.01 * np.arange(1001, dtype=np.float64)
    hamiltonian, initial_density, collapse_operators, z0 = build_model()
    densities = solve_lindblad(hamiltonian, initial_density, collapse_operators, times)
    return times, densities, observable_values(densities, z0)


def density_diagnostics(densities):
    """Measure trace/Hermiticity/positivity errors without repairing states.

    Input (N,d,d) density matrices. Output scalar diagnostics; Frobenius
    norm measures Hermiticity. eigvalsh uses the Hermitian part only as a
    diagnostic after checking the unmodified matrices' Hermiticity.
    """
    adjoints = densities.conj().swapaxes(-1, -2)
    return {
        "max_trace_error": float(np.max(np.abs(np.trace(densities, axis1=1, axis2=2) - 1))),
        "max_hermiticity_frobenius_error": float(np.max(np.linalg.norm(densities - adjoints, axis=(1, 2)))),
        "minimum_eigenvalue": float(np.min(np.linalg.eigvalsh((densities + adjoints) / 2))),
        "all_finite": bool(np.all(np.isfinite(densities))),
    }

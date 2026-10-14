"""Reduced-density McLachlan equations and an explicit truncated SVD solve."""

from .ansatz import system_density, jax, jnp
from .derivatives import density_jacobian
from .model import build_model
import numpy as np

# Only fixed model operators are converted; no Exact states/data are read.
_hamiltonian, _, _collapse_operators, _z0 = build_model()
HAMILTONIAN = jnp.asarray(_hamiltonian.full(), dtype=jnp.complex128)
COLLAPSE_OPERATORS = tuple(jnp.asarray(operator.full(), dtype=jnp.complex128)
                           for operator in _collapse_operators)


def lindblad_rhs(density):
    r"""Return L(rho)=-i[H,rho]+sum(c rho c^dagger-{c^dagger c,rho}/2).

    Input/output: complex128 (4,4), system basis |00>,|01>,|10>,|11>.
    H=X0+X1+ZZ/4, c_j=sigma^-_j, gamma=1, hbar=1; dimensionless time.
    This is evaluated at the current ansatz density, never the Exact state.
    """
    derivative = -1j * (HAMILTONIAN @ density - density @ HAMILTONIAN)
    for collapse in COLLAPSE_OPERATORS:
        adjoint = jnp.conj(collapse).T
        rate = adjoint @ collapse
        derivative += collapse @ density @ adjoint - 0.5 * (rate @ density + density @ rate)
    return derivative


def raw_equations(theta):
    r"""Return rho,D,Lrho,M_complex,V_complex for Eq. (28)/(29).

    theta: six real paper parameters (6,). D has shape (6,4,4), M (6,6),
    V (6,). M_ij=Tr(D_i^dagger D_j), V_i=Tr(D_i^dagger Lrho).
    Summing conjugated matrix elements is the Hilbert-Schmidt inner
    product, including the adjoint. No real projection occurs here.
    """
    density = system_density(theta)
    derivatives = density_jacobian(theta)
    generator = lindblad_rhs(density)
    matrix = jnp.einsum("ist,jst->ij", jnp.conj(derivatives), derivatives)
    vector = jnp.einsum("ist,st->i", jnp.conj(derivatives), generator)
    return density, derivatives, generator, matrix, vector


_compiled_equations = jax.jit(raw_equations)


def evaluate_equations(theta, *, compiled=True):
    """Evaluate real M,V after verifying imaginary parts, symmetry and PSD.

    NumPy conversion is only after AD. Return rho,D,Lrho,M,V and measured
    identity errors. Matrices/states are not symmetrized or repaired.
    """
    theta = np.asarray(theta)
    if theta.shape != (6,):
        raise ValueError("Expected six real parameters, shape (6,).")
    if np.iscomplexobj(theta):
        raise TypeError("Paper parameters must be real.")
    if not np.all(np.isfinite(theta)):
        raise ValueError("Parameters must be finite.")
    function = _compiled_equations if compiled else raw_equations
    density, derivatives, generator, matrix, vector = (
        np.asarray(array) for array in function(jnp.asarray(theta, dtype=jnp.float64)))
    if not all(np.all(np.isfinite(array)) for array in (density, derivatives, generator, matrix, vector)):
        raise RuntimeError("McLachlan evaluation produced NaN/Inf.")
    errors = {"m_imaginary": float(np.max(np.abs(matrix.imag))),
              "v_imaginary": float(np.max(np.abs(vector.imag))),
              "m_symmetry": float(np.max(np.abs(matrix.real - matrix.real.T))),
              "m_minimum_eigenvalue": float(np.min(np.linalg.eigvalsh(matrix.real)))}
    if max(errors["m_imaginary"], errors["v_imaginary"], errors["m_symmetry"]) > 1e-12:
        raise RuntimeError(f"Invalid real McLachlan identities: {errors}")
    if errors["m_minimum_eigenvalue"] < -1e-12:
        raise RuntimeError(f"McLachlan matrix is not positive semidefinite: {errors}")
    return density, derivatives, generator, matrix.real, vector.real, errors


def solve_svd(matrix, vector, *, cutoff_multiplier=1.0):
    r"""Solve M theta_dot=V with a truncated, minimum-norm SVD solution.

    Inputs: real float64 M (6,6), V (6,). Cutoff is multiplier*6*eps*s_max;
    keep singular values strictly above it. Return theta_dot (6,) and
    diagnostic scalars/spectrum. No inverse, Tikhonov or special first step.
    Full condition is infinity when numerically rank deficient; retained
    condition describes only kept singular values. Zero M returns zero
    theta_dot and reports the actual unsatisfied residual.
    """
    matrix, vector = np.asarray(matrix), np.asarray(vector)
    if matrix.shape != (6, 6) or vector.shape != (6,):
        raise ValueError("Expected M (6,6) and V (6,).")
    if np.iscomplexobj(matrix) or np.iscomplexobj(vector):
        raise TypeError("Verify imaginary parts before passing real M,V to the solver.")
    if not np.isfinite(cutoff_multiplier) or cutoff_multiplier <= 0:
        raise ValueError("cutoff_multiplier must be finite and positive.")
    matrix, vector = matrix.astype(np.float64), vector.astype(np.float64)
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(vector)):
        raise ValueError("M,V must be finite.")
    left, singular_values, right_adjoint = np.linalg.svd(matrix, full_matrices=False)
    cutoff = cutoff_multiplier * 6 * np.finfo(np.float64).eps * singular_values[0]
    keep = singular_values > cutoff
    coefficients = np.zeros(6, dtype=np.float64)
    np.divide(left.T @ vector, singular_values, out=coefficients, where=keep)
    velocity = right_adjoint.T @ coefficients
    rank = int(np.count_nonzero(keep))
    retained_condition = float(singular_values[0] / singular_values[keep][-1]) if rank else np.inf
    diagnostics = {"singular_values": singular_values, "cutoff": float(cutoff), "rank": rank,
                   "full_condition": retained_condition if rank == 6 else np.inf,
                   "retained_condition": retained_condition,
                   "linear_residual": float(np.linalg.norm(matrix @ velocity - vector))}
    if not np.all(np.isfinite(velocity)):
        raise RuntimeError("SVD solution produced NaN/Inf.")
    return velocity, diagnostics

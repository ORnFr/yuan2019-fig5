"""Baseline parameter Forward Euler; no Exact data or state-Euler update."""

import numpy as np
from .mclachlan import evaluate_equations, solve_svd


def evolve_vqs(*, dt=0.01, stop=10.0, cutoff_multiplier=1.0):
    r"""Return aligned states/parameters/diagnostics for theta(0)=six zeros.

    theta_(n+1)=theta_n+dt*theta_dot_n. rho_(n+1) is reconstructed from
    the ansatz, never updated as rho+dt Lrho. Input times are dimensionless,
    hbar=1 and field coefficient=1. Default baseline has 1001 output points.
    All arrays include diagnostics at the same theta/time; parameter_step
    contains only the 1000 actually applied updates (not an extra final step).
    """
    if not np.isfinite(dt) or not np.isfinite(stop) or dt <= 0 or stop <= 0:
        raise ValueError("dt and stop must be finite and positive.")
    steps = round(stop / dt)
    if not np.isclose(steps * dt, stop, atol=1e-12, rtol=0):
        raise ValueError("stop must be an integer multiple of dt.")
    times = dt * np.arange(steps + 1, dtype=np.float64)
    arrays = {"t": times, "theta": np.zeros((steps + 1, 6)),
              "rho": np.empty((steps + 1, 4, 4), dtype=np.complex128),
              "z0": np.empty(steps + 1), "theta_dot": np.empty((steps + 1, 6)),
              "M": np.empty((steps + 1, 6, 6)), "V": np.empty((steps + 1, 6)),
              "singular_values": np.empty((steps + 1, 6)),
              "rank": np.empty(steps + 1, dtype=np.int64),
              "parameter_step": np.empty((steps, 6))}
    scalar_names = ("cutoff", "full_condition", "retained_condition", "linear_residual",
                    "tangent_residual", "trace_error", "hermiticity_error", "minimum_eigenvalue",
                    "m_imaginary", "v_imaginary", "m_symmetry", "m_minimum_eigenvalue")
    arrays.update({name: np.empty(steps + 1) for name in scalar_names})
    z0_matrix = np.diag([1, 1, -1, -1])
    for index in range(steps + 1):
        density, derivatives, generator, matrix, vector, identity_errors = evaluate_equations(arrays["theta"][index])
        velocity, solve_diagnostics = solve_svd(matrix, vector, cutoff_multiplier=cutoff_multiplier)
        tangent = np.einsum("i,ist->st", velocity, derivatives)
        observable = np.trace(z0_matrix @ density)
        trace_error = float(abs(np.trace(density) - 1))
        hermiticity_error = float(np.linalg.norm(density - density.conj().T))
        minimum_eigenvalue = float(np.min(np.linalg.eigvalsh((density + density.conj().T) / 2)))
        if trace_error > 1e-10 or hermiticity_error > 1e-10 or minimum_eigenvalue < -1e-10:
            raise RuntimeError(f"Invalid density at t={times[index]}.")
        if abs(observable.imag) > 1e-12 or abs(observable.real) > 1 + 1e-10:
            raise RuntimeError(f"Invalid Z0 at t={times[index]}.")
        arrays["rho"][index], arrays["z0"][index] = density, observable.real
        arrays["M"][index], arrays["V"][index], arrays["theta_dot"][index] = matrix, vector, velocity
        for name, value in {**solve_diagnostics, **identity_errors,
                            "tangent_residual": float(np.linalg.norm(tangent - generator)),
                            "trace_error": trace_error, "hermiticity_error": hermiticity_error,
                            "minimum_eigenvalue": minimum_eigenvalue}.items():
            arrays[name][index] = value
        if index < steps:
            arrays["parameter_step"][index] = dt * velocity
            arrays["theta"][index + 1] = arrays["theta"][index] + arrays["parameter_step"][index]
            if not np.all(np.isfinite(arrays["theta"][index + 1])):
                raise RuntimeError(f"Nonfinite parameter update after t={times[index]}.")
    return arrays

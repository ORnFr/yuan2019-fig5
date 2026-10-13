"""Independent references and physical checks for the Exact calculation."""

import numpy as np
from scipy.sparse.linalg import expm_multiply
from .model import build_model, qutip
from .exact import solve_lindblad, observable_values, density_diagnostics


def reference_operators():
    """Return explicitly written NumPy H, c0, c1 for the ordered four-state basis.

    These matrices are independent of the QuTiP tensor constructors used
    in build_model. H has units of the transverse field; damping rate=1.
    """
    hamiltonian = np.array([[0.25, 1, 1, 0], [1, -0.25, 0, 1],
                            [1, 0, -0.25, 1], [0, 1, 1, 0.25]], dtype=np.complex128)
    c0 = np.zeros((4, 4), dtype=np.complex128)
    c1 = np.zeros((4, 4), dtype=np.complex128)
    c0[0, 2] = c0[1, 3] = 1
    c1[0, 1] = c1[2, 3] = 1
    return hamiltonian, [c0, c1]


def lindblad_action(density, hamiltonian, collapse_operators):
    r"""Evaluate -i[H,rho]+sum(c rho c^dagger-{c^dagger c,rho}/2).

    Inputs/output: complex matrices (d,d), hbar=1. This direct NumPy
    multiplication is a validation reference, not a time integrator.
    """
    derivative = -1j * (hamiltonian @ density - density @ hamiltonian)
    for collapse in collapse_operators:
        rate_operator = collapse.conj().T @ collapse
        derivative += (collapse @ density @ collapse.conj().T
                       - 0.5 * (rate_operator @ density + density @ rate_operator))
    return derivative


def reference_liouvillian():
    r"""Return the independent 16x16 generator on column-stacked density matrices.

    Column k is vec_F(L(E_k)); explicitly applying the equation to all
    matrix units avoids depending on a library's Kronecker/vectorization
    formula. The basis order for rho remains |00>,|01>,|10>,|11>.
    """
    hamiltonian, collapse_operators = reference_operators()
    matrix_units = np.eye(16, dtype=np.complex128)
    columns = []
    for column in range(16):
        density = matrix_units[:, column].reshape((4, 4), order="F")
        derivative = lindblad_action(density, hamiltonian, collapse_operators)
        columns.append(derivative.reshape(16, order="F"))
    return np.column_stack(columns)


def check_model_and_generator():
    """Check matrices, lowering directions and the complete generator."""
    hamiltonian, initial_density, collapse_operators, z0 = build_model()
    expected_h, expected_c = reference_operators()
    np.testing.assert_array_equal(hamiltonian.full(), expected_h)
    assert hamiltonian.isherm
    np.testing.assert_array_equal(initial_density.full(), np.diag([1, 0, 0, 0]))
    np.testing.assert_array_equal(z0.full(), np.diag([1, 1, -1, -1]))
    for collapse, expected in zip(collapse_operators, expected_c):
        np.testing.assert_array_equal(collapse.full(), expected)
    generator = reference_liouvillian()
    qutip_generator = qutip.liouvillian(hamiltonian, collapse_operators).full()
    generator_error = float(np.max(np.abs(generator - qutip_generator)))
    trace_row = np.eye(4).reshape(16, order="F")
    trace_preservation_error = float(np.max(np.abs(trace_row @ generator)))
    rng = np.random.default_rng(2019)
    amplitudes = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    density = amplitudes @ amplitudes.conj().T
    density /= np.trace(density)
    derivative = lindblad_action(density, expected_h, expected_c)
    hermiticity_error = float(np.linalg.norm(derivative - derivative.conj().T))
    assert generator_error < 1e-14 and trace_preservation_error < 1e-14
    assert hermiticity_error < 1e-14
    return {"generator_max_absolute_error": generator_error,
            "generator_trace_preservation_error": trace_preservation_error,
            "derivative_hermiticity_frobenius_error": hermiticity_error,
            "absolute_tolerance": 1e-14, "random_seed": 2019}


def check_analytic_damping():
    r"""Check H=0, c=|0><1|, rho(0)=|1><1| against p_1(t)=exp(-t).

    Hilbert space C^2; time is dimensionless, gamma=1. This fixes the
    collapse direction and rate independently of the two-qubit curve.
    """
    times = np.linspace(0, 3, 61, dtype=np.float64)
    collapse = qutip.basis(2, 0) * qutip.basis(2, 1).dag()
    densities = solve_lindblad(0 * qutip.qeye(2), qutip.basis(2, 1).proj(), [collapse], times)
    expected_population = np.exp(-times)
    population_error = float(np.max(np.abs(densities[:, 1, 1] - expected_population)))
    z_error = float(np.max(np.abs(observable_values(densities, qutip.sigmaz())
                                 - (1 - 2 * expected_population))))
    assert population_error < 1e-9 and z_error < 1e-9
    return {"time_range": [0, 3], "points": 61, "gamma": 1,
            "population_max_absolute_error": population_error,
            "z_max_absolute_error": z_error, "absolute_tolerance": 1e-9}


def check_full_evolution(times, densities, values):
    """Check all 1001 states against an independent exponential and tighter ODE solve.

    Comparisons use maximum absolute matrix-element/observable errors.
    The reference propagates vec_F(rho) with exp(t L), without mesolve.
    State reconstruction transposes the column-stacked array back to rho.
    No output normalization, clipping, smoothing or time shift is used.
    """
    np.testing.assert_array_equal(times, 0.01 * np.arange(1001))
    assert densities.shape == (1001, 4, 4) and values.shape == (1001,)
    generator = reference_liouvillian()
    initial_vector = np.zeros(16, dtype=np.complex128)
    initial_vector[0] = 1
    vectors = expm_multiply(generator, initial_vector, start=0, stop=10,
                            num=1001, endpoint=True, traceA=np.trace(generator))
    reference = vectors.reshape((1001, 4, 4)).swapaxes(1, 2)
    hamiltonian, initial_density, collapse_operators, z0 = build_model()
    tighter = solve_lindblad(hamiltonian, initial_density, collapse_operators,
                             times, atol=1e-14, rtol=1e-12)
    reference_error = float(np.max(np.abs(densities - reference)))
    reference_z_error = float(np.max(np.abs(values - observable_values(reference, z0))))
    tighter_error = float(np.max(np.abs(densities - tighter)))
    tighter_z_error = float(np.max(np.abs(values - observable_values(tighter, z0))))
    diagnostics = density_diagnostics(densities)
    assert reference_error < 1e-9 and reference_z_error < 1e-9
    assert tighter_error < 1e-9 and tighter_z_error < 1e-9
    assert diagnostics["all_finite"]
    assert diagnostics["max_trace_error"] < 1e-10
    assert diagnostics["max_hermiticity_frobenius_error"] < 1e-10
    assert diagnostics["minimum_eigenvalue"] >= -1e-10
    assert np.max(np.abs(values)) <= 1 + 1e-10 and values[0] == 1
    np.testing.assert_array_equal(densities[0], initial_density.full())
    return {"exponential_rho_max_absolute_error": reference_error,
            "exponential_z0_max_absolute_error": reference_z_error,
            "tighter_rho_max_absolute_difference": tighter_error,
            "tighter_z0_max_absolute_difference": tighter_z_error,
            "comparison_absolute_tolerance": 1e-9,
            "physical_absolute_tolerance": 1e-10,
            "tighter_atol": 1e-14, "tighter_rtol": 1e-12,
            "density_diagnostics": diagnostics}

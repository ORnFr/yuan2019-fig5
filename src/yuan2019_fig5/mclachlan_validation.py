"""Independent density-space references and aligned Euler trajectory checks."""

import numpy as np
from .ansatz_validation import dense_ansatz_state, full_matrix_partial_trace, sample_parameters
from .derivatives_validation import dense_density_jacobian
from .mclachlan import evaluate_equations, solve_svd
from .model import build_model, qutip
from .vqs import evolve_vqs


def dense_equations(theta):
    r"""Return reference rho,D,Lrho,M,V using full gates and QuTiP superoperator.

    D: (6,4,4), rho/Lrho: (4,4), M: (6,6), V: (6,). The independent
    derivative uses gate-product rules and full-density partial trace.
    QuTiP's column-stacking convention is handled by its vector converters.
    Explicit Tr(D_i^dagger D_j) checks the production element contraction.
    """
    density = full_matrix_partial_trace(dense_ansatz_state(theta))
    derivatives = dense_density_jacobian(theta)
    hamiltonian, _, collapse, _ = build_model()
    density_object = qutip.Qobj(density, dims=[[2, 2], [2, 2]])
    generator = qutip.vector_to_operator(
        qutip.liouvillian(hamiltonian, collapse) @ qutip.operator_to_vector(density_object)).full()
    matrix = np.empty((6, 6), dtype=np.complex128)
    vector = np.empty(6, dtype=np.complex128)
    for i in range(6):
        vector[i] = np.trace(derivatives[i].conj().T @ generator)
        for j in range(6):
            matrix[i, j] = np.trace(derivatives[i].conj().T @ derivatives[j])
    assert max(np.max(abs(matrix.imag)), np.max(abs(vector.imag))) < 1e-12
    return density, derivatives, generator, matrix.real, vector.real


def check_equations():
    """Check all 17 accepted parameter cases against independent dense references."""
    names = ("rho", "D", "Lrho", "M", "V")
    errors = dict.fromkeys(names, 0.0)
    jit_error = velocity_error = 0.0
    for theta in sample_parameters():
        actual = evaluate_equations(theta, compiled=False)
        compiled = evaluate_equations(theta)
        reference = dense_equations(theta)
        for name, value, expected, accelerated in zip(names, actual[:5], reference, compiled[:5]):
            errors[name] = max(errors[name], float(np.max(abs(value - expected))))
            jit_error = max(jit_error, float(np.max(abs(value - accelerated))))
        # Solve the original real density least-squares problem B x ~= b,
        # without forming Gram normal equations. sqrt(rcond_M) matches the
        # singular-value threshold of B^T B. This is only a reference check.
        derivatives, generator = actual[1:3]
        basis = derivatives.reshape(6, 16).T
        design = np.concatenate((basis.real, basis.imag), axis=0)
        target = np.concatenate((generator.real.ravel(), generator.imag.ravel()))
        reference_velocity = np.linalg.lstsq(design, target,
            rcond=np.sqrt(6 * np.finfo(np.float64).eps))[0]
        velocity, _ = solve_svd(actual[3], actual[4])
        velocity_error = max(velocity_error, float(np.max(abs(velocity - reference_velocity))))
    assert max(errors.values()) < 1e-12 and jit_error < 1e-12
    assert velocity_error < 1e-9
    return {"cases": 17, "seed": 2019, "dense_max_absolute_errors": errors,
            "jit_max_absolute_error": jit_error, "absolute_tolerance": 1e-12,
            "density_lstsq_velocity_max_absolute_error": velocity_error,
            "velocity_absolute_tolerance": 1e-9}


def check_zero_solve():
    """Verify analytic rank-two M, V and the minimum-norm initial velocity."""
    rho, derivatives, generator, matrix, vector, _ = evaluate_equations(np.zeros(6))
    expected = np.zeros((6, 6))
    for first, second in ((0, 3), (1, 4)):
        expected[np.ix_([first, second], [first, second])] = [[0.5, 1], [1, 2]]
    expected_vector = np.array([1, 1, 0, 2, 2, 0.])
    expected_velocity = np.array([0.4, 0.4, 0, 0.8, 0.8, 0])
    velocity, info = solve_svd(matrix, vector)
    for actual, reference in ((matrix, expected), (vector, expected_vector),
                              (velocity, expected_velocity), (rho, np.diag([1, 0, 0, 0]))):
        np.testing.assert_allclose(actual, reference, atol=1e-12, rtol=0)
    assert info["rank"] == 2 and np.isinf(info["full_condition"])
    tangent_residual = float(np.linalg.norm(np.einsum("i,ist->st", velocity, derivatives) - generator))
    assert info["linear_residual"] < 1e-12 and tangent_residual < 1e-12
    return {"M": matrix.tolist(), "V": vector.tolist(), "theta_dot": velocity.tolist(),
            "singular_values": info["singular_values"].tolist(), "rank": info["rank"],
            "cutoff": info["cutoff"], "full_condition": "infinity (rank deficient)",
            "retained_condition": info["retained_condition"],
            "linear_residual": info["linear_residual"], "tangent_residual": tangent_residual,
            "absolute_tolerance": 1e-12}


def audit_trajectory(arrays, dt):
    """Check all states/time updates; independently reconstruct 11 sampled densities.

    Identity/evolution alignment tolerance is 1e-12; density validity uses
    1e-10. Infinity is permitted only in condition diagnostics where the
    saved rank justifies it. No state is normalized, clipped or repaired.
    """
    count = len(arrays["t"])
    assert arrays["theta"].shape == (count, 6) and arrays["rho"].shape == (count, 4, 4)
    assert arrays["theta"].dtype == np.float64 and arrays["rho"].dtype == np.complex128
    np.testing.assert_array_equal(arrays["t"], dt * np.arange(count))
    np.testing.assert_array_equal(arrays["theta"][0], np.zeros(6))
    np.testing.assert_allclose(np.diff(arrays["theta"], axis=0), dt * arrays["theta_dot"][:-1], atol=1e-12, rtol=0)
    np.testing.assert_array_equal(arrays["parameter_step"], dt * arrays["theta_dot"][:-1])
    np.testing.assert_allclose(arrays["theta"][1], dt * np.array([.4, .4, 0, .8, .8, 0]), atol=1e-12, rtol=0)
    for name, values in arrays.items():
        if name not in ("full_condition", "retained_condition"):
            assert np.all(np.isfinite(values)), name
    np.testing.assert_array_equal(np.isinf(arrays["full_condition"]), arrays["rank"] < 6)
    np.testing.assert_array_equal(np.isinf(arrays["retained_condition"]), arrays["rank"] == 0)
    assert np.all(arrays["full_condition"] >= 1) and np.all(arrays["retained_condition"] >= 1)
    trace_error = float(np.max(abs(np.trace(arrays["rho"], axis1=1, axis2=2) - 1)))
    hermiticity_error = float(np.max(np.linalg.norm(arrays["rho"] - arrays["rho"].conj().swapaxes(1, 2), axis=(1, 2))))
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(arrays["rho"])))
    assert trace_error < 1e-10 and hermiticity_error < 1e-10 and minimum_eigenvalue >= -1e-10
    z0 = np.trace(np.diag([1, 1, -1, -1]) @ arrays["rho"], axis1=1, axis2=2)
    assert np.max(abs(z0.imag)) < 1e-12
    np.testing.assert_allclose(arrays["z0"], z0.real, atol=1e-12, rtol=0)
    dense_error = 0.0
    for index in np.unique(np.linspace(0, count - 1, min(11, count), dtype=int)):
        reference = full_matrix_partial_trace(dense_ansatz_state(arrays["theta"][index]))
        dense_error = max(dense_error, float(np.max(abs(reference - arrays["rho"][index]))))
    assert dense_error < 1e-12
    ranks, counts = np.unique(arrays["rank"], return_counts=True)
    return {"points": count, "trace_max_absolute_error": trace_error,
            "hermiticity_max_frobenius_error": hermiticity_error,
            "minimum_density_eigenvalue": minimum_eigenvalue,
            "density_validity_tolerance": 1e-10, "dense_reconstruction_max_absolute_error": dense_error,
            "alignment_absolute_tolerance": 1e-12,
            "rank_counts": {str(rank): int(number) for rank, number in zip(ranks, counts)},
            "linear_residual_max": float(np.max(arrays["linear_residual"])),
            "tangent_residual_max": float(np.max(arrays["tangent_residual"])),
            "retained_condition_max": float(np.max(arrays["retained_condition"])),
            "parameter_step_norm_max": float(np.max(np.linalg.norm(arrays["parameter_step"], axis=1))),
            "final_z0": float(arrays["z0"][-1])}


def check_short_evolution():
    """Run threshold x1/x10/x100 and dt .01/.005/.0025 checks to t=.2.

    Return every diagnostic trajectory and measured differences. Time-step
    differences are observations, not a proof of convergence over t<=10.
    """
    baseline = evolve_vqs(stop=.2)
    checks = {"baseline": audit_trajectory(baseline, .01)}
    saved = {f"baseline_{key}": value for key, value in baseline.items()}
    sensitivity = []
    for multiplier in (10., 100.):
        other = evolve_vqs(stop=.2, cutoff_multiplier=multiplier)
        audit_trajectory(other, .01)
        difference = {"cutoff_multiplier": multiplier,
                      "z0_max_difference": float(np.max(abs(other["z0"] - baseline["z0"]))),
                      "rho_max_difference": float(np.max(abs(other["rho"] - baseline["rho"]))),
                      "theta_max_difference": float(np.max(abs(other["theta"] - baseline["theta"]))),
                      "rank_changes": int(np.count_nonzero(other["rank"] != baseline["rank"]))}
        assert difference["rho_max_difference"] < 1e-10
        sensitivity.append(difference)
        saved.update({f"cutoff{int(multiplier)}_{key}": value for key, value in other.items()})
    step_checks = []
    previous = baseline
    for dt, stride in ((.005, 2), (.0025, 4)):
        other = evolve_vqs(dt=dt, stop=.2)
        audit_trajectory(other, dt)
        sampled = other["z0"][::stride]
        step_checks.append({"dt": dt,
            "z0_max_difference_from_baseline": float(np.max(abs(sampled - baseline["z0"]))),
            "z0_max_difference_from_previous_dt": float(np.max(abs(other["z0"][::2] - previous["z0"])))})
        saved.update({f"dt{str(dt).replace('.', '_')}_{key}": value for key, value in other.items()})
        previous = other
    checks.update({"threshold_sensitivity": sensitivity, "short_step_diagnostics": step_checks,
                   "stop": .2, "cutoff_sensitivity_rho_tolerance": 1e-10,
                   "limitation": "Short-time diagnostics do not establish full-time convergence or the Figure 5 error target."})
    return saved, checks

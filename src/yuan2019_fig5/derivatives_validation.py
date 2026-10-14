"""Independent product-rule, analytic and finite-difference Jacobian checks."""

import numpy as np
from .ansatz import system_density, jax, jnp, tc
from .ansatz_validation import dense_paper_gates, tensor_four, sample_parameters
from .derivatives import density_jacobian

FD_STEPS = np.array([1e-3, 1e-4, 1e-5, 1e-6], dtype=np.float64)
FD_ABSOLUTE_TOLERANCES = np.array([2e-6, 2e-8, 2e-9, 2e-9])
FD_RELATIVE_TOLERANCES = np.array([5e-6, 5e-8, 5e-8, 5e-8])
RELATIVE_NORM_THRESHOLD = 1e-12


def dense_state_and_derivatives(theta):
    r"""Return psi (16,) and dpsi (6,16) using 16x16 product-rule matrices.

    RX'=-i X RX, RZZ'=-i ZZ RZZ; a controlled derivative has a zero
    control=0 block and RX' in control=1. Paper angles, no half factor.
    For each gate G, propagate psi_new=G psi_old and all tangents as
    dpsi_new=G dpsi_old; add G'_i psi_old only for its own parameter.
    This NumPy/SciPy reference does not use JAX AD or finite differences.
    """
    identity = np.eye(2, dtype=np.complex128)
    pauli_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    pauli_z = np.diag([1, -1]).astype(np.complex128)
    zz = np.kron(pauli_z, pauli_z)
    p0, p1 = np.diag([1, 0]), np.diag([0, 1])
    hadamard = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
    rotations = [dense_paper_gates(angle) for angle in theta]
    crx0 = (tensor_four(p0, identity, identity, identity)
            + tensor_four(p1, identity, rotations[0][0], identity))
    crx1 = (tensor_four(identity, p0, identity, identity)
            + tensor_four(identity, p1, identity, rotations[1][0]))
    gates = [tensor_four(hadamard, identity, identity, identity),
             tensor_four(identity, hadamard, identity, identity), crx0, crx1,
             np.kron(np.eye(4), rotations[2][2]),
             tensor_four(identity, identity, rotations[3][0], identity),
             tensor_four(identity, identity, identity, rotations[4][0]),
             np.kron(np.eye(4), rotations[5][2])]
    derivative_gates = [None, None,
                        tensor_four(p1, identity, -1j * pauli_x @ rotations[0][0], identity),
                        tensor_four(identity, p1, identity, -1j * pauli_x @ rotations[1][0]),
                        np.kron(np.eye(4), -1j * zz @ rotations[2][2]),
                        tensor_four(identity, identity, -1j * pauli_x @ rotations[3][0], identity),
                        tensor_four(identity, identity, identity, -1j * pauli_x @ rotations[4][0]),
                        np.kron(np.eye(4), -1j * zz @ rotations[5][2])]
    state = np.zeros(16, dtype=np.complex128)
    state[0] = 1
    derivatives = np.zeros((6, 16), dtype=np.complex128)
    for gate_index, gate in enumerate(gates):
        previous_state = state
        derivatives = (gate @ derivatives.T).T
        if gate_index >= 2:
            derivatives[gate_index - 2] += derivative_gates[gate_index] @ previous_state
        state = gate @ previous_state
    return state, derivatives


def dense_density_jacobian(theta):
    r"""Return Tr_A(|dpsi_i><psi|+|psi><dpsi_i|), shape (6,4,4).

    Explicit block summation of each full 16x16 derivative density avoids
    reusing the production reshape/einsum contraction. This is a density
    derivative, not a pure-state metric or a density time evolution.
    """
    state, derivatives = dense_state_and_derivatives(theta)
    jacobian = np.zeros((6, 4, 4), dtype=np.complex128)
    for parameter, derivative in enumerate(derivatives):
        full_derivative = np.outer(derivative, state.conj()) + np.outer(state, derivative.conj())
        for ancilla in range(4):
            for row in range(4):
                for column in range(4):
                    jacobian[parameter, row, column] += full_derivative[4 * ancilla + row, 4 * ancilla + column]
    return jacobian


def zero_parameter_reference():
    r"""Analytic D at theta=0: D0=-i[X0,rho0]/2, D3=2D0; likewise for parameters 1 and 4.

    D2=D5=0 because ZZ initially only changes the global phase. The factor
    1/2 for controlled rotations is the initial ancilla population in |1>.
    This independently fixes the parameter axis, signs and angle factors.
    """
    pauli_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    x0, x1 = np.kron(pauli_x, np.eye(2)), np.kron(np.eye(2), pauli_x)
    density = np.diag([1, 0, 0, 0]).astype(np.complex128)
    reference = np.zeros((6, 4, 4), dtype=np.complex128)
    reference[0] = -0.5j * (x0 @ density - density @ x0)
    reference[1] = -0.5j * (x1 @ density - density @ x1)
    reference[3] = 2 * reference[0]
    reference[4] = 2 * reference[1]
    return reference


def check_zero_jacobian():
    """Check the six zero-point derivative matrices against the analytic formula."""
    jacobian = density_jacobian(jnp.zeros(6, dtype=jnp.float64))
    assert jacobian.shape == (6, 4, 4) and jacobian.dtype == jnp.complex128
    jacobian = np.asarray(jacobian)
    reference = zero_parameter_reference()
    error = float(np.max(np.abs(jacobian - reference)))
    assert error < 1e-12
    relations = {"D2_zero": float(np.linalg.norm(jacobian[2])),
                 "D5_zero": float(np.linalg.norm(jacobian[5])),
                 "D3_minus_2D0": float(np.linalg.norm(jacobian[3] - 2 * jacobian[0])),
                 "D4_minus_2D1": float(np.linalg.norm(jacobian[4] - 2 * jacobian[1]))}
    assert max(relations.values()) < 1e-12
    return {"max_absolute_error": error, "absolute_tolerance": 1e-12,
            "derivative_frobenius_norms": np.linalg.norm(jacobian, axis=(1, 2)).tolist(),
            "relations_frobenius_errors": relations,
            "interpretation": "Two independent first-order density directions; redundancy/zero directions are preserved. No McLachlan matrix or solver computed."}


def central_difference(theta, step):
    """Return density central differences (6,4,4); validation only, not production AD."""
    jacobian = np.empty((6, 4, 4), dtype=np.complex128)
    for parameter in range(6):
        displacement = np.zeros(6, dtype=np.float64)
        displacement[parameter] = step
        plus = np.asarray(system_density(jnp.asarray(theta + displacement)))
        minus = np.asarray(system_density(jnp.asarray(theta - displacement)))
        jacobian[parameter] = (plus - minus) / (2 * step)
    return jacobian


def check_derivative_samples(parameters=None):
    """Check 17 zero/nonzero Jacobians against exact product-rule and multiple FD steps.

    Absolute errors: maximum complex element magnitude per parameter.
    Relative errors: Frobenius error divided by the reference derivative
    norm, only where that norm exceeds 1e-12; undefined entries use zero
    storage plus an explicit mask. No derivative is normalized or repaired.
    """
    if parameters is None:
        parameters = sample_parameters()
    jacobians, references, densities = [], [], []
    absolute_errors = np.zeros((len(parameters), len(FD_STEPS), 6))
    relative_errors = np.zeros_like(absolute_errors)
    relative_defined = np.zeros((len(parameters), 6), dtype=bool)
    max_analytic_error = max_trace_error = max_hermiticity_error = 0.0
    for case, theta in enumerate(parameters):
        jacobian = density_jacobian(jnp.asarray(theta))
        assert jacobian.shape == (6, 4, 4) and jacobian.dtype == jnp.complex128
        jacobian = np.asarray(jacobian)
        reference = dense_density_jacobian(theta)
        assert np.all(np.isfinite(jacobian)) and np.all(np.isfinite(reference))
        max_analytic_error = max(max_analytic_error, float(np.max(np.abs(jacobian - reference))))
        max_trace_error = max(max_trace_error, float(np.max(np.abs(np.trace(jacobian, axis1=1, axis2=2)))))
        max_hermiticity_error = max(max_hermiticity_error, float(np.max(np.linalg.norm(jacobian - jacobian.conj().swapaxes(1, 2), axis=(1, 2)))))
        reference_norms = np.linalg.norm(reference, axis=(1, 2))
        relative_defined[case] = reference_norms > RELATIVE_NORM_THRESHOLD
        for step_index, step in enumerate(FD_STEPS):
            difference = central_difference(theta, step) - jacobian
            absolute_errors[case, step_index] = np.max(np.abs(difference), axis=(1, 2))
            np.divide(np.linalg.norm(difference, axis=(1, 2)), reference_norms,
                      out=relative_errors[case, step_index], where=relative_defined[case])
        jacobians.append(jacobian)
        references.append(reference)
        densities.append(np.asarray(system_density(jnp.asarray(theta))))
    assert max_analytic_error < 1e-12
    assert max_trace_error < 1e-12 and max_hermiticity_error < 1e-12
    max_absolute_by_step = np.max(absolute_errors, axis=(0, 2))
    max_relative_by_step = np.max(relative_errors, axis=(0, 2))
    assert np.all(max_absolute_by_step < FD_ABSOLUTE_TOLERANCES)
    assert np.all(max_relative_by_step < FD_RELATIVE_TOLERANCES)
    summary = {"cases": len(parameters), "parameter_seed": 2019,
               "analytic_max_absolute_error": max_analytic_error, "analytic_absolute_tolerance": 1e-12,
               "derivative_trace_max_absolute_error": max_trace_error,
               "derivative_hermiticity_max_frobenius_error": max_hermiticity_error,
               "physical_identity_absolute_tolerance": 1e-12,
               "fd_steps": FD_STEPS.tolist(), "fd_max_absolute_errors": max_absolute_by_step.tolist(),
               "fd_absolute_tolerances": FD_ABSOLUTE_TOLERANCES.tolist(),
               "fd_max_defined_relative_errors": max_relative_by_step.tolist(),
               "fd_relative_tolerances": FD_RELATIVE_TOLERANCES.tolist(),
               "relative_definition": "Frobenius error / analytic derivative Frobenius norm; undefined when norm<=1e-12",
               "undefined_relative_directions": int(np.count_nonzero(~relative_defined)),
               "relative_norm_threshold": RELATIVE_NORM_THRESHOLD,
               "dtype": "complex128", "shape_per_case": [6, 4, 4]}
    arrays = {"theta": np.asarray(parameters), "rho": np.asarray(densities),
              "D_ad": np.asarray(jacobians), "D_dense": np.asarray(references),
              "fd_steps": FD_STEPS, "fd_max_absolute_errors": absolute_errors,
              "fd_relative_errors": relative_errors, "fd_relative_defined": relative_defined}
    return arrays, summary


def check_jvp_and_jit():
    """Compare split Jacobian, complex-output JVP and compiled output at zero/nonzero points."""
    direction = np.array([0.23, -0.41, 0.17, 0.37, -0.29, 0.53], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    compiled_jacobian = jax.jit(density_jacobian)
    jvp_error = jit_error = 0.0
    for theta in sample_parameters()[[0, 7, 8]]:
        parameters = jnp.asarray(theta)
        jacobian = density_jacobian(parameters)
        _, tangent = jax.jvp(system_density, (parameters,), (jnp.asarray(direction),))
        contraction = jnp.einsum("i,ist->st", jnp.asarray(direction), jacobian)
        compiled = compiled_jacobian(parameters)
        assert tangent.dtype == compiled.dtype == jnp.complex128
        jvp_error = max(jvp_error, float(np.max(np.abs(np.asarray(tangent - contraction)))))
        jit_error = max(jit_error, float(np.max(np.abs(np.asarray(compiled - jacobian)))))
    assert jvp_error < 1e-12 and jit_error < 1e-12
    assert jax.config.x64_enabled and tc.backend.name == "jax"
    assert all(device.platform == "cpu" for device in jax.devices())
    return {"cases": 3, "direction": direction.tolist(), "jvp_max_absolute_error": jvp_error,
            "jit_max_absolute_error": jit_error, "absolute_tolerance": 1e-12,
            "devices": [str(device) for device in jax.devices()]}

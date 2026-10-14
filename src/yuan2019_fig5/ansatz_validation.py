"""Independent matrix and partial-trace references for Phase 3."""

import numpy as np
from scipy.linalg import expm
from .ansatz import build_ansatz, purification_state, reduce_system_density, system_density, jax, jnp, tc
from .runtime import import_project_qutip

ABSOLUTE_TOLERANCE = 1e-12


def dense_paper_gates(angle):
    r"""Return RX, CRX and RZZ from independent NumPy/SciPy matrices.

    angle is a real scalar in radians. Outputs have shapes (2,2), (4,4),
    (4,4); CRX basis is |control target>, controlled on 1. RX=e^(-i a X)
    and RZZ=e^(-i a ZZ). No TensorCircuit gate constructors are used.
    """
    pauli_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    pauli_z = np.diag([1, -1]).astype(np.complex128)
    rx = expm(-1j * angle * pauli_x)
    crx = np.zeros((4, 4), dtype=np.complex128)
    crx[:2, :2] = np.eye(2)
    crx[2:, 2:] = rx
    rzz = expm(-1j * angle * np.kron(pauli_z, pauli_z))
    return rx, crx, rzz


def tensor_four(first, second, third, fourth):
    """Return four ordered tensor factors as a 16x16 dense matrix."""
    return np.kron(np.kron(np.kron(first, second), third), fourth)


def dense_ansatz_state(theta):
    """Apply Figure 4's independently embedded 16x16 matrices to |0000>.

    theta has shape (6,), all real; output is complex128 (16,) in the
    declared big-endian wire order. Projectors explicitly embed the two
    nonadjacent control/target pairs, avoiding any shared circuit helper.
    """
    identity = np.eye(2, dtype=np.complex128)
    hadamard = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
    p0, p1 = np.diag([1, 0]), np.diag([0, 1])
    rx0 = dense_paper_gates(theta[0])[0]
    rx1 = dense_paper_gates(theta[1])[0]
    controlled0 = (tensor_four(p0, identity, identity, identity)
                   + tensor_four(p1, identity, rx0, identity))
    controlled1 = (tensor_four(identity, p0, identity, identity)
                   + tensor_four(identity, p1, identity, rx1))
    gates = [tensor_four(hadamard, identity, identity, identity),
             tensor_four(identity, hadamard, identity, identity),
             controlled0, controlled1,
             np.kron(np.eye(4), dense_paper_gates(theta[2])[2]),
             tensor_four(identity, identity, dense_paper_gates(theta[3])[0], identity),
             tensor_four(identity, identity, identity, dense_paper_gates(theta[4])[0]),
             np.kron(np.eye(4), dense_paper_gates(theta[5])[2])]
    state = np.zeros(16, dtype=np.complex128)
    state[0] = 1
    for gate in gates:
        state = gate @ state
    return state


def full_matrix_partial_trace(state):
    r"""Explicitly sum diagonal ancilla blocks of the full 16x16 density.

    state: NumPy complex vector (16,). Return rho_S (4,4) from
    sum_a rho_full[4*a+s,4*a+t]. No reshape/einsum shortcut is reused.
    """
    full_density = np.outer(state, state.conj())
    reduced = np.zeros((4, 4), dtype=np.complex128)
    for ancilla in range(4):
        for row in range(4):
            for column in range(4):
                reduced[row, column] += full_density[4 * ancilla + row, 4 * ancilla + column]
    return reduced


def sample_parameters():
    """Return 17 fixed/reproducible parameter vectors (17,6), seed 2019."""
    zero = np.zeros((1, 6))
    isolated = 0.37 * np.eye(6)
    generic = np.array([[0.21, -0.43, 0.17, 0.38, -0.59, 0.26],
                        [np.pi / 2, np.pi / 2, -0.31, 0.23, -0.19, 0.47]])
    random = np.random.default_rng(2019).uniform(-1.2, 1.2, size=(8, 6))
    return np.concatenate((zero, isolated, generic, random)).astype(np.float64)


def check_gate_conventions():
    """Check full gate matrices, every basis column and control=0 behavior."""
    angles = [0, 0.17, -0.41, 0.73, np.pi / 2]
    errors = {"rx": 0.0, "crx": 0.0, "rzz": 0.0, "unitarity": 0.0,
              "basis_action": 0.0, "control_zero": 0.0}
    for angle in angles:
        references = dense_paper_gates(angle)
        gates = [tc.gates.rx(theta=2 * angle).tensor,
                 tc.gates.crx(theta=2 * angle).tensor,
                 tc.gates.rzz(theta=2 * angle, half=True).tensor]
        for name, tensor, reference in zip(("rx", "crx", "rzz"), gates, references):
            matrix = np.asarray(tensor).reshape(reference.shape)
            errors[name] = max(errors[name], float(np.max(np.abs(matrix - reference))))
            errors["unitarity"] = max(errors["unitarity"], float(np.max(np.abs(matrix.conj().T @ matrix - np.eye(len(matrix))))))
            for basis_index in range(len(matrix)):
                basis = np.eye(len(matrix), dtype=np.complex128)[:, basis_index]
                circuit = tc.Circuit(1 if name == "rx" else 2, inputs=jnp.asarray(basis))
                if name == "rx":
                    circuit.rx(0, theta=2 * angle)
                elif name == "crx":
                    circuit.crx(0, 1, theta=2 * angle)
                else:
                    circuit.rzz(0, 1, theta=2 * angle, half=True)
                error = float(np.max(np.abs(np.asarray(circuit.state()) - reference @ basis)))
                errors["basis_action"] = max(errors["basis_action"], error)
                if name == "crx" and basis_index < 2:
                    errors["control_zero"] = max(errors["control_zero"], float(np.max(np.abs(np.asarray(circuit.state()) - basis))))
    for error in errors.values():
        assert error < ABSOLUTE_TOLERANCE, errors
    # A nonzero angle must distinguish the paper gate from a missing factor 2.
    wrong_half_angle_gap = float(np.max(np.abs(np.asarray(tc.gates.rx(theta=0.37).tensor)
                                               - dense_paper_gates(0.37)[0])))
    assert wrong_half_angle_gap > 0.1
    return {"angles_radians": angles, "max_absolute_errors": errors,
            "absolute_tolerance": ABSOLUTE_TOLERANCE, "missing_factor_two_gap": wrong_half_angle_gap}


def check_partial_trace_examples():
    """Check product, complex, entangled and random states against two references."""
    qutip = import_project_qutip()
    product = np.zeros(16, dtype=np.complex128)
    product[0] = 1
    complex_product = np.zeros(16, dtype=np.complex128)
    complex_product[4:6] = [1 / np.sqrt(2), 1j / np.sqrt(2)]
    cross_bell = np.zeros(16, dtype=np.complex128)
    cross_bell[[0, 10]] = 1 / np.sqrt(2)
    maximally_entangled = np.zeros(16, dtype=np.complex128)
    maximally_entangled[[0, 5, 10, 15]] = 0.5
    random = np.random.default_rng(2026).normal(size=(2, 16, 2))
    random_states = random[:, :, 0] + 1j * random[:, :, 1]
    random_states /= np.linalg.norm(random_states, axis=1)[:, None]
    states = [product, complex_product, cross_bell, maximally_entangled, *random_states]
    expected = [np.diag([1, 0, 0, 0]),
                np.outer(complex_product[4:8], complex_product[4:8].conj()),
                np.diag([0.5, 0, 0.5, 0]), np.eye(4) / 4]
    matrix_error = qutip_error = analytic_error = 0.0
    for index, state in enumerate(states):
        density = np.asarray(reduce_system_density(jnp.asarray(state)))
        matrix_error = max(matrix_error, float(np.max(np.abs(density - full_matrix_partial_trace(state)))))
        qutip_density = qutip.Qobj(state, dims=[[2, 2, 2, 2], [1, 1, 1, 1]]).ptrace([2, 3]).full()
        qutip_error = max(qutip_error, float(np.max(np.abs(density - qutip_density))))
        if index < 4:
            analytic_error = max(analytic_error, float(np.max(np.abs(density - expected[index]))))
    # Trace should equal the input norm squared, not be silently normalized.
    scaled = np.asarray(reduce_system_density(2 * complex_product))
    scaling_error = float(np.max(np.abs(scaled - 4 * expected[1])))
    for error in (matrix_error, qutip_error, analytic_error, scaling_error):
        assert error < ABSOLUTE_TOLERANCE
    return {"examples": 6, "state_seed": 2026, "full_matrix_max_absolute_error": matrix_error,
            "qutip_ptrace_max_absolute_error": qutip_error, "analytic_max_absolute_error": analytic_error,
            "unnormalized_scaling_max_absolute_error": scaling_error, "absolute_tolerance": ABSOLUTE_TOLERANCE}


def check_ansatz_states(parameters):
    """Check complete states/densities, initial condition and physical validity."""
    errors = {"dense_state": 0.0, "dense_density": 0.0, "partial_trace": 0.0,
              "state_norm": 0.0, "trace": 0.0, "hermiticity": 0.0}
    minimum_eigenvalue = 1.0
    states, densities = [], []
    for theta in parameters:
        state = purification_state(jnp.asarray(theta))
        density = system_density(jnp.asarray(theta))
        assert state.dtype == density.dtype == jnp.complex128
        assert state.shape == (16,) and density.shape == (4, 4)
        state, density = np.asarray(state), np.asarray(density)
        assert np.all(np.isfinite(state)) and np.all(np.isfinite(density))
        reference = dense_ansatz_state(theta)
        measurements = {
            "dense_state": np.max(np.abs(state - reference)),
            "dense_density": np.max(np.abs(density - full_matrix_partial_trace(reference))),
            "partial_trace": np.max(np.abs(density - full_matrix_partial_trace(state))),
            "state_norm": abs(np.vdot(state, state) - 1), "trace": abs(np.trace(density) - 1),
            "hermiticity": np.linalg.norm(density - density.conj().T),
        }
        for name, error in measurements.items():
            errors[name] = max(errors[name], float(error))
        minimum_eigenvalue = min(minimum_eigenvalue, float(np.min(np.linalg.eigvalsh((density + density.conj().T) / 2))))
        states.append(state)
        densities.append(density)
    for error in errors.values():
        assert error < ABSOLUTE_TOLERANCE, errors
    assert minimum_eigenvalue >= -ABSOLUTE_TOLERANCE
    zero_state = np.zeros(16, dtype=np.complex128)
    zero_state[[0, 4, 8, 12]] = 0.5
    np.testing.assert_allclose(states[0], zero_state, atol=ABSOLUTE_TOLERANCE, rtol=0)
    np.testing.assert_allclose(densities[0], np.diag([1, 0, 0, 0]), atol=ABSOLUTE_TOLERANCE, rtol=0)
    # Maximal entanglement has a maximally mixed system, even after system gates.
    np.testing.assert_allclose(densities[8], np.eye(4) / 4, atol=ABSOLUTE_TOLERANCE, rtol=0)
    return np.asarray(states), np.asarray(densities), {
        "cases": len(parameters), "parameter_seed": 2019, "max_absolute_errors": errors,
        "minimum_eigenvalue": minimum_eigenvalue, "absolute_tolerance": ABSOLUTE_TOLERANCE,
        "state_dtype": str(states[0].dtype), "density_dtype": str(densities[0].dtype),
        "zero_state": "|++>_A tensor |00>_S", "zero_system_density": "|00><00|",
    }


def check_wire_mapping_and_jit():
    """Check bit significance, system Z0 mapping and forward-only JIT output."""
    wire_error = 0.0
    for wire in range(4):
        circuit = tc.Circuit(4)
        circuit.x(wire)
        expected = np.eye(16, dtype=np.complex128)[:, 2 ** (3 - wire)]
        wire_error = max(wire_error, float(np.max(np.abs(np.asarray(circuit.state()) - expected))))
    theta = sample_parameters()[7]
    circuit = build_ansatz(jnp.asarray(theta))
    density = np.asarray(system_density(jnp.asarray(theta)))
    z0 = np.trace(np.diag([1, 1, -1, -1]) @ density)
    z1 = np.trace(np.diag([1, -1, 1, -1]) @ density)
    z0_error = float(abs(np.asarray(circuit.expectation_ps(z=[2])) - z0))
    z1_error = float(abs(np.asarray(circuit.expectation_ps(z=[3])) - z1))
    assert abs(z0 - z1) > 1e-3, "Use a nondegenerate case to distinguish the two system wires."
    jit_density = jax.jit(system_density)(jnp.asarray(theta))
    assert jit_density.dtype == jnp.complex128
    jit_error = float(np.max(np.abs(np.asarray(jit_density) - density)))
    for error in (wire_error, z0_error, z1_error, jit_error):
        assert error < ABSOLUTE_TOLERANCE
    assert jax.config.x64_enabled and tc.backend.name == "jax"
    assert all(device.platform == "cpu" for device in jax.devices())
    return {"wire_order": ["a0", "a1", "s0", "s1"], "basis_wire_max_absolute_error": wire_error,
            "system_z0_wire": 2, "system_z1_wire": 3, "z0_max_absolute_error": z0_error,
            "z1_max_absolute_error": z1_error, "jit_density_max_absolute_error": jit_error,
            "devices": [str(device) for device in jax.devices()], "absolute_tolerance": ABSOLUTE_TOLERANCE,
            "derivatives_computed": False}

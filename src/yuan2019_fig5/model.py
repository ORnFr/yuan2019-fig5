"""Two-qubit Lindblad model: Yuan et al. (2019), Section 6.3, Eq. (62)."""

from .runtime import import_project_qutip

qutip = import_project_qutip()


def build_model():
    r"""Return H, rho0, collapse operators and Z0 as QuTiP Qobjs.

    H = X_0 + X_1 + Z_0 Z_1/4, rho0 = |00><00|, c_j = sigma^-_j,
    Z0 = Z tensor I. Hilbert space: C^2 tensor C^2; ordered basis
    |00>, |01>, |10>, |11>. Operators have shape (4,4), complex128.
    We set hbar=1 and measure dimensionless time in the transverse-field
    scale (X coefficient = 1). Independent damping rates are gamma_j=1.
    No input parameters or ancillary qubits are used in this fixed model.
    Source: local journal PDF, printed/PDF page 20, Eq. (62).
    """
    identity = qutip.qeye(2)
    pauli_x = qutip.sigmax()
    pauli_z = qutip.sigmaz()
    # Define the computational-basis lowering direction explicitly.
    sigma_minus = qutip.basis(2, 0) * qutip.basis(2, 1).dag()
    hamiltonian = (qutip.tensor(pauli_x, identity)
                   + qutip.tensor(identity, pauli_x)
                   + 0.25 * qutip.tensor(pauli_z, pauli_z))
    initial_ket = qutip.tensor(qutip.basis(2, 0), qutip.basis(2, 0))
    initial_density = initial_ket.proj()
    collapse_operators = [qutip.tensor(sigma_minus, identity),
                          qutip.tensor(identity, sigma_minus)]
    z0 = qutip.tensor(pauli_z, identity)
    return hamiltonian, initial_density, collapse_operators, z0

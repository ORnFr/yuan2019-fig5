"""Figure 4 purification ansatz and reduced system density (Phase 3 only)."""

from .runtime import configure_project

configure_project()

import jax

# Enable precision before importing TensorCircuit or creating any arrays.
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")

import jax.numpy as jnp
import tensorcircuit as tc

tc.set_backend("jax")
tc.set_dtype("complex128")

WIRE_ORDER = ("a0", "a1", "s0", "s1")


def build_ansatz(theta):
    r"""Return the four-qubit circuit in Yuan et al. (2019), Figure 4.

    Input: six finite real angles theta=(theta0,...,theta5), shape (6,),
    converted to float64. Angles are dimensionless (radians). Output:
    TensorCircuit circuit with wire order (a0,a1,s0,s1)=(0,1,2,3).
    Hilbert space: (C^2)^tensor4, basis |a0 a1 s0 s1>, initial |0000>.
    Paper gates are RX(theta)=exp(-i theta X), RZZ(theta)=exp(-i theta ZZ),
    and CRX(theta)=P0 tensor I + P1 tensor RX(theta), controlled on 1.
    Library RX/CRX use half angles; RZZ is explicitly called with half=True.
    Therefore every library angle below is twice the paper parameter.
    Source: printed/PDF page 20, Figure 4. No Hamiltonian 1/4 factor is
    inserted into the ansatz, and no parameter time evolution occurs here.
    """
    theta = jnp.asarray(theta)
    if theta.shape != (6,):
        raise ValueError("theta must have shape (6,).")
    if jnp.issubdtype(theta.dtype, jnp.complexfloating):
        raise TypeError("The six paper parameters must be real.")
    theta = theta.astype(jnp.float64)
    circuit = tc.Circuit(4)
    circuit.h(0)
    circuit.h(1)
    circuit.crx(0, 2, theta=2 * theta[0])
    circuit.crx(1, 3, theta=2 * theta[1])
    circuit.rzz(2, 3, theta=2 * theta[2], half=True)
    circuit.rx(2, theta=2 * theta[3])
    circuit.rx(3, theta=2 * theta[4])
    circuit.rzz(2, 3, theta=2 * theta[5], half=True)
    return circuit


def purification_state(theta):
    r"""Return |Psi(theta)> as a JAX complex128 vector of shape (16,).

    Input and basis conventions are those of build_ansatz. The pure state
    includes two ancillae; at theta=0 it is |++>_A tensor |00>_S.
    No NumPy conversion, normalization correction or fitting is applied.
    """
    return build_ansatz(theta).state()


def reduce_system_density(state):
    r"""Compute rho_S=Tr_A(|Psi><Psi|) by contracting the ancilla index.

    Input: complex state (16,) in |a0 a1 s0 s1> order. Output: JAX
    complex128 matrix (4,4), basis |s0 s1>=(|00>,|01>,|10>,|11>).
    Reshape to A[a,s] (4,4); rho[s,t]=sum_a A[a,s] conj(A[a,t]).
    Trace equals <Psi|Psi>; the function does not normalize its input.
    Only JAX operations are used, preserving the future differentiation
    path. This contraction assumes the declared, independently tested
    basis order; it is neither A A^dagger nor A^dagger A.
    """
    state = jnp.asarray(state, dtype=jnp.complex128)
    if state.shape != (16,):
        raise ValueError("state must have shape (16,) in |a0 a1 s0 s1> order.")
    amplitudes = jnp.reshape(state, (4, 4))
    return jnp.einsum("as,at->st", amplitudes, jnp.conj(amplitudes))


def system_density(theta):
    r"""Return rho_S(theta)=Tr_A(|Psi(theta)><Psi(theta)|), shape (4,4).

    Input: six real paper angles, shape (6,); output: complex128 JAX array.
    No Exact data, Jacobian, McLachlan equation or time step is involved.
    """
    return reduce_system_density(purification_state(theta))

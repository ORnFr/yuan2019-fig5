"""Real-parameter Jacobian of the reduced density; Phase 4 only."""

from .ansatz import system_density, jax, jnp


def density_jacobian(theta):
    r"""Return D_i = partial rho_S / partial theta_i, shape (6,4,4).

    Input: six finite real paper angles (6,), converted to float64.
    Output: JAX complex128 array, parameter axis first; matrix basis
    |s0 s1>=(|00>,|01>,|10>,|11>). Parameters are dimensionless radians.
    The underlying four-qubit basis is |a0 a1 s0 s1>, as in ansatz.py.

    jacfwd on each real/imaginary part initially returns (4,4,6).
    Combining them and moving the last axis preserves the complex
    derivative of a real-parameter, conjugation-containing density map.
    No holomorphic assumption, NumPy conversion, finite difference,
    Hermitian projection or trace correction is used. See the project's
    derivation notes; rho_S is the Figure 4 ansatz, not the full pure-state
    density or an Exact solution. No McLachlan matrix/evolution is computed.
    """
    theta = jnp.asarray(theta)
    if theta.shape != (6,):
        raise ValueError("theta must have shape (6,).")
    if jnp.issubdtype(theta.dtype, jnp.complexfloating):
        raise TypeError("Density derivatives are taken with respect to real parameters.")
    theta = theta.astype(jnp.float64)
    real_jacobian = jax.jacfwd(lambda angles: jnp.real(system_density(angles)))(theta)
    imaginary_jacobian = jax.jacfwd(lambda angles: jnp.imag(system_density(angles)))(theta)
    return jnp.moveaxis(real_jacobian + 1j * imaginary_jacobian, -1, 0)

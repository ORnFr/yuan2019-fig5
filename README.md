# Reproduction of Figure 5 from Yuan et al. (2019)

This repository presents an educational reproduction of the two-qubit
open-system simulation shown in Figure 5 of:

X. Yuan et al., “Theory of variational quantum simulation,”
Quantum 3, 191 (2019).
DOI: https://doi.org/10.22331/q-2019-10-07-191

The variational simulation uses a four-qubit purification ansatz
constructed with TensorCircuit. The system density matrix is obtained
by tracing out the two ancillary qubits, and its parameter derivatives
are computed using JAX automatic differentiation.

The McLachlan equations are solved using a truncated singular value
decomposition (SVD), and the circuit parameters are evolved using the
forward Euler method. An independent, high-accuracy numerical reference
is obtained by solving the Lindblad master equation with QuTiP's
`mesolve` solver.

All simulations are performed on a classical CPU. The repository
includes source code, dependency records, numerical data, and vector
figures.

Author-provided raw data are unavailable, so pointwise agreement with
the published curves has not been established.

Code and documentation were developed with assistance from ChatGPT.

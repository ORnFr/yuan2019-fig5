# Phase 1 environment record

Verified on 2026-10-13 (Asia/Shanghai). Scope: environment configuration and small circuit/AD checks only.

## Host and interpreter choice

- Native architecture: `arm64` (Apple Silicon); macOS 26.6.2.
- Default user interpreter: Python 3.14.6 at `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3`.
- Existing Apple interpreter: `/usr/bin/python3`, Python 3.9.6; not used.
- Existing Codex-bundled interpreter: Python 3.12.14, native arm64; used to create `.venv` with `python -m venv`.
- `.venv/pyvenv.cfg` has `include-system-site-packages = false`. Research packages are installed only in `.venv`; its base standard library remains a read-only dependency outside the project.
- pip 25.0.1 came from venv's ensurepip and was not upgraded. No system Python, global Git setting or unrelated project was changed.

Python 3.12 was selected because the verified NumPy 1.26.4 and jaxlib 0.4.38 releases provide CPython 3.12 macOS arm64 wheels. Their release files do not provide CPython 3.14 wheels. This is a compatibility choice for this pinned stack, not a claim that all JAX or QuTiP releases lack Python 3.14 support. The already present Python 3.12 avoided installing another interpreter.

## Version rationale and official evidence

The [TensorCircuit quick start](https://tensorcircuit.readthedocs.io/en/latest/quickstart.html) documents `set_backend("jax")`; the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html) supports CPU wheels on macOS Apple ARM. The [JAX x64 guide](https://docs.jax.dev/en/latest/101/default_dtypes.html) requires explicit x64 enablement. These settings were applied before circuit/array/JIT creation and verified by actual outputs.

| Selection | Reason and release metadata |
| --- | --- |
| [TensorCircuit 0.12.0](https://pypi.org/project/tensorcircuit/0.12.0/) | The queried stable release of the requested `tensorcircuit` distribution. Its required TensorNetwork dependency is `tensornetwork-ng`, resolved to 0.5.1; no framework substitution was made. |
| [NumPy 1.26.4](https://pypi.org/project/numpy/1.26.4/) | The installed TensorCircuit JAX backend uses `np.ComplexWarning` in `cast` (line 325). This name moved out of the top-level namespace in NumPy 2; see the [NumPy migration guide](https://numpy.org/doc/2.0/numpy_2_0_migration_guide.html). Keep the compatible NumPy 1.x interface. |
| [JAX 0.4.38](https://pypi.org/project/jax/0.4.38/) and [jaxlib 0.4.38](https://pypi.org/project/jaxlib/0.4.38/) | A matched CPU pair with Python 3.12/arm64 wheels and NumPy 1.26 support. Installed TensorCircuit still calls `jax.tree_map` (backend lines 422/760), removed in JAX 0.6.0 according to the [JAX changelog](https://docs.jax.dev/en/latest/changelog.html#jax-0-6-0-april-16-2025). This older pair retains that interface. It passed the selected circuit, AD, JIT and vmap checks; newer JAX stacks were not exhaustively tested. |
| [QuTiP 5.2.3](https://pypi.org/project/qutip/5.2.3/) | Stable 5.x release with a native CPython 3.12 wheel. Installed metadata requires NumPy >=1.22 and SciPy >=1.9, excluding 1.16.0/1.17.0. |
| [SciPy 1.15.3](https://pypi.org/project/scipy/1.15.3/) | Native wheel, NumPy >=1.23.5,<2.5; compatible with this JAX/QuTiP pair. Matrix exponential arithmetic was actually checked. |
| [Matplotlib 3.11.2](https://pypi.org/project/matplotlib/3.11.2/) | Queried stable version supports Python >=3.11 and NumPy >=1.25; headless Agg PNG rendering passed. |
| [pytest 9.1.1](https://pypi.org/project/pytest/9.1.1/) | Queried stable version supports Python >=3.10; four project checks passed. |

The QuTiP installation documentation's generic requirements table is older than its release metadata. Selection used the exact distribution metadata, the pip resolver and runtime evidence rather than that table alone. `results/phase1_dependencies.json` records the exact installed metadata, wheel URL and SHA-256 for all 26 downloaded distributions. pip is the 27th pinned distribution, supplied by ensurepip.

## Installation and cache handling

Commands were executed inside the subproject. pip used the official PyPI index, TLS verification with the existing `/etc/ssl/cert.pem`, `--only-binary=:all:`, a cache under `work/cache/pip`, and `TMPDIR=work/tmp` (absolute values at runtime). A wheel-only dry run succeeded before installation; all selected packages then installed successfully without source builds. A later offline dry run of `requirements.txt` found all pins satisfied.

The default Python 3.14 urllib attempt lacked a usable default CA chain. Retrying with the existing macOS CA bundle succeeded. No certificate store was modified and TLS verification was never disabled. Sandbox network access was used only for official package metadata and authorized package installation; no Git remote operation or upload occurred in Phase 1.

The verification script routes Python bytecode, pip, XDG, Matplotlib, JAX and temporary paths into ignored `work/`. QuTiP 5.2.3's installed `settings.py` initializes `tmproot` by calling `expanduser("~")` and creating `.qutip`, with no environment-variable override for that import-time step. `import_project_qutip()` temporarily redirects that call during import, then restores `os.path.expanduser` immediately and verifies both QuTiP cache roots are inside `work/`. It does not change HOME, installed package files or physical/numerical code. Future project entry points must configure this before their first QuTiP import; plain `import qutip` alone would use its upstream default cache behavior.

## Actual verification

The final report is `results/phase1_environment_20261013_212040_737785.json`. It includes Python/platform details, all versions, the requirements SHA-256, actual dtypes, fixed angles, tolerances and measured errors. No randomness or physical time evolution was used.

| Check | Actual result |
| --- | --- |
| `pip check` | `No broken requirements found.` |
| Pin consistency | All 27 requirements match the installed versions. |
| JAX/TensorCircuit CPU precision | CPU only; parameters/observables/gradients float64; states and state derivatives complex128. |
| Numerical precision probe | A `2^-40 = 9.094947017729282e-13` increment survives near 1 in both real and imaginary parts; float64 epsilon is `2.220446049250313e-16`. |
| RX state vs independent SciPy matrix exponential | Maximum absolute element error `1.1102230246251565e-16`, tolerance `1e-12`. |
| Z expectation / analytic gradient | Maximum absolute errors `5.551115123125783e-17` / `1.1102230246251565e-16`, tolerance `1e-12`. |
| Complex state `jacfwd` vs analytic derivative | Maximum absolute error `0.0` at the five chosen angles, tolerance `1e-12`. |
| Central differences | Steps `1e-4,1e-5,1e-6`: errors `2.083332950597594e-10,4.912126261302774e-12,8.46907544094222e-11`; tolerances `1e-8,1e-9,2e-9`. |
| JIT / TensorCircuit backend AD | Maximum error `1.1102230246251565e-16`, tolerance `1e-12`. |
| JIT + vmap | Maximum error `5.551115123125783e-17`, tolerance `1e-12`. |
| Bell state / normalization | Bell state error `0.0`; RX normalization error `0.0` at the chosen angles. |
| QuTiP and Matplotlib | complex128 matrix arithmetic passed; Agg produced a valid 6438-byte PNG in memory. |
| pytest from `/private/tmp` | Final run: `4 passed in 1.84s`; all caches/logs still in the project. |

The script's absolute invocation is in the README. The actual final pytest invocation additionally used `--junitxml=<project>/work/pytest.xml` and redirected stdout/stderr to `work/pytest.log`. Installation, resolver and exploratory logs remain only in ignored `work/`; only final compact evidence is tracked.

## Limits and phase boundary

These checks establish a usable small CPU circuit/AD environment. They do not establish Figure 5 reproduction, full framework compatibility, solver convergence, correctness of purification/partial trace/McLachlan modules, or GPU support. The full upstream library suites and a second fresh-environment rebuild were not run. The optional TensorCircuit Qiskit/Cirq/cloud/Optax notices concern unused functionality; those packages were not installed. This stack intentionally uses older NumPy/JAX interfaces and must be revalidated before upgrades. The Codex-bundled base interpreter must remain available or the venv must be recreated with an existing compatible Python 3.12. Stop after the local Phase 1 commit and await user acceptance.

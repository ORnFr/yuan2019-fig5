# Yuan et al. (2019) Figure 5 reproduction

Phase 1 (environment configuration and verification) is complete and awaiting user acceptance. No Exact, purification, McLachlan or VQS evolution modules have been implemented.

Target: mixed-state variational quantum simulation of the two-qubit Lindblad model in Figure 5 of [Yuan et al. (2019)](https://quantum-journal.org/papers/q-2019-10-07-191/).

- Exact reference: QuTiP.
- VQS: TensorCircuit with JAX automatic differentiation.
- Environment: project-local `.venv`, Python 3.12.14, pip 25.0.1, CPU only.
- Workflow: review each phase before proceeding; local commits; no push or merge without explicit approval.

See [Master Prompt](docs/Codex_Master_Prompt.md), [Mac/GitHub setup guide](docs/github_setup_zh.md), and [progress](PROGRESS.md).

The Git root is this subproject only. Literature in `/Users/ornfr/Desktop/bookandpaper` remains outside the repository and strictly read-only.

The private GitHub repository is [ORnFr/yuan2019-fig5](https://github.com/ORnFr/yuan2019-fig5). The existing remote baseline was established during Phase 0. Local `main` tracks `origin/main`. Phase 1 uses the local branch `phase1/environment`, based on planning commit `07ea9d7`. No push or merge has been performed.

## Verified dependencies

| Package | Version |
| --- | --- |
| QuTiP | 5.2.3 |
| TensorCircuit | 0.12.0 |
| JAX / jaxlib | 0.4.38 / 0.4.38 |
| NumPy | 1.26.4 |
| SciPy | 1.15.3 |
| Matplotlib | 3.11.2 |
| pytest | 9.1.1 |

`requirements.txt` pins all 27 installed distributions, including transitive dependencies and pip. The version rationale, official sources and limitations are in [environment notes](docs/environment.md). Installation wheel URLs and SHA-256 digests are recorded in [dependency provenance](results/phase1_dependencies.json).

## Run the Phase 1 checks

The environment already exists; no activation or installation is needed to run these commands. Set the project path once:

```bash
FIG5_ROOT='/Users/ornfr/Desktop/ai小垃圾/MSVQD/reproductions/yuan2019_fig5'
mkdir -p "$FIG5_ROOT/work/cache/pycache" "$FIG5_ROOT/work/cache/pip" "$FIG5_ROOT/work/tmp"
export PYTHONPYCACHEPREFIX="$FIG5_ROOT/work/cache/pycache"
export PIP_CACHE_DIR="$FIG5_ROOT/work/cache/pip"
export PIP_CONFIG_FILE=/dev/null
export PIP_DISABLE_PIP_VERSION_CHECK=1
export TMPDIR="$FIG5_ROOT/work/tmp"
```

Run from any working directory:

```bash
"$FIG5_ROOT/.venv/bin/python" -m pip check
"$FIG5_ROOT/.venv/bin/python" "$FIG5_ROOT/scripts/check_environment.py"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$FIG5_ROOT/.venv/bin/python" -m pytest \
  "$FIG5_ROOT/tests" -q \
  -o "cache_dir=$FIG5_ROOT/work/cache/pytest" \
  --basetemp="$FIG5_ROOT/work/tmp/pytest"
```

The script configures CPU, JAX x64, TensorCircuit complex128 and project-local caches before creating circuits. Inputs are five fixed angles in radians and a two-qubit Bell circuit; there is no randomness. It verifies installed pins, imports, `pip check`, actual precision, states, analytic derivatives, a complex-state Jacobian, finite differences, JIT/vmap, QuTiP matrix arithmetic and in-memory Agg rendering. A failed check exits with an error. Successful runs create a unique `results/phase1_environment_<timestamp>.json` without overwriting earlier reports. The pytest suite contains four corresponding numerical checks.

The final Phase 1 evidence is [the environment report](results/phase1_environment_20261013_212040_737785.json). It records CPU execution, float64/complex128 output, the requirements digest and actual numerical errors. Analytic and matrix comparisons use absolute tolerance `1e-12`; central-difference comparisons have separately recorded step-dependent tolerances. See [Chinese learning notes](docs/learning_zh.md) for the formulas and Python concepts.

## Recreate the environment in a new checkout

Use these commands only when that checkout has no `.venv`. Do not recreate or overwrite the existing verified environment. The base interpreter below already existed and was read without modification; a different machine needs an existing native arm64 Python 3.12 interpreter. No Python interpreter was installed for this phase.

```bash
cd "$FIG5_ROOT"
'/Users/ornfr/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3' -m venv .venv
"$FIG5_ROOT/.venv/bin/python" -m pip install \
  --index-url https://pypi.org/simple --cert /etc/ssl/cert.pem \
  --only-binary=:all: -r "$FIG5_ROOT/requirements.txt"
```

Set the local cache variables above before installation. `/etc/ssl/cert.pem` is the existing macOS CA bundle used here; no certificate checks were disabled. Run the Phase 1 checks after installation. This checkout's `.venv` refers to the existing Codex-bundled base interpreter and must be recreated if that interpreter is relocated or removed; virtual environments are not portable directories.

## Files and scope

- `scripts/check_environment.py`: standalone environment verification, not research evolution code.
- `tests/test_environment.py`: four smoke checks using independent analytic/matrix references.
- `requirements.txt`: the single pinned installation specification.
- `docs/`: approved master plan, setup guide, environment rationale and Chinese learning notes.
- `results/`: small tracked Phase 1 verification and dependency records.
- `.venv/` and `work/`: ignored environment, caches and temporary logs; no literature.

TensorCircuit may print notices about absent Qiskit, Cirq, Tencent cloud support and Optax. Those optional features are outside the selected JAX circuit/AD path and were not installed. The small checks passed without them. This phase does not verify the full upstream test suites, GPU execution, the Figure 4 ansatz or Lindblad integration. Phase 2 requires separate user approval.

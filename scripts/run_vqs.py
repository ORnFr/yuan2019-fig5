"""Validate Phase 5, then save the fixed parameter-Euler baseline to t=10."""

import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from importlib import metadata

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from yuan2019_fig5.runtime import configure_project

configure_project()

import numpy as np
from yuan2019_fig5.mclachlan_validation import (
    check_equations, check_zero_solve, check_short_evolution, audit_trajectory,
)
from yuan2019_fig5.vqs import evolve_vqs


def main():
    """Only run the full baseline after independent and short-time checks pass."""
    checks = {"equations": check_equations(), "initial_point": check_zero_solve()}
    diagnostics, checks["short_evolution"] = check_short_evolution()
    print("Independent equations and short evolution passed; running t<=10.", flush=True)
    arrays = evolve_vqs()
    checks["full_baseline"] = audit_trajectory(arrays, .01)
    timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    output_dir = PROJECT / "results" / f"vqs_{timestamp}"
    output_dir.mkdir(exist_ok=False)
    artifacts = {}
    for filename, data in (("vqs.npz", arrays), ("short_diagnostics.npz", diagnostics)):
        path = output_dir / filename
        np.savez_compressed(path, **data)
        with np.load(path, allow_pickle=False) as loaded:
            assert set(loaded.files) == set(data)
            for key, expected in data.items():
                np.testing.assert_array_equal(loaded[key], expected)
        artifacts[filename] = {"path": str(path.relative_to(PROJECT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "shapes": {key: list(value.shape) for key, value in data.items()},
            "dtypes": {key: str(value.dtype) for key, value in data.items()}}
    sources = [Path(__file__), *(PROJECT / "src/yuan2019_fig5" / name for name in
        ("runtime.py", "model.py", "ansatz.py", "ansatz_validation.py", "derivatives.py",
         "derivatives_validation.py", "mclachlan.py", "mclachlan_validation.py", "vqs.py")),
         PROJECT / "tests/test_mclachlan_vqs.py"]
    record = {
        "status": "Phase 5 verified baseline completed; Figure 5 comparison deferred to Phase 6",
        "timestamp": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "command": '"$FIG5_ROOT/.venv/bin/python" "$FIG5_ROOT/scripts/run_vqs.py"',
        "paper": {"title": "Theory of variational quantum simulation", "year": 2019,
            "journal_url": "https://quantum-journal.org/papers/q-2019-10-07-191/",
            "local_pdf_sha256": "e142a56c57c822a45fb22aea7e7b9a47f694a9c37842796a040c9f8d4c2053b7",
            "pages": {"equations_28_29": 11, "figure4_equation62": 20, "figure5": 21},
            "version": "Local 41-page journal-formatted PDF; accepted footer 2019-09-07"},
        "model": {"H": "X tensor I + I tensor X + 0.25 Z tensor Z",
            "collapse": ["|0><1| tensor I", "I tensor |0><1|"], "gamma": [1, 1],
            "initial_state": "|00><00|", "observable": "Z tensor I"},
        "units": "hbar=1; dimensionless time in field coefficient=1; real angles in radians",
        "basis": {"wires": ["a0", "a1", "s0", "s1"], "system": ["00", "01", "10", "11"]},
        "precision": "CPU, JAX float64/complex128, NumPy float64 SVD",
        "equations": "M_ij=Re Tr(D_i^dagger D_j), V_i=Re Tr(D_i^dagger L(rho(theta)))",
        "svd": {"cutoff": "6 * eps(float64) * s_max", "multiplier": 1,
            "retained": "s > cutoff", "method": "truncated minimum-norm least squares",
            "full_condition": "Infinity when numerical rank<6; NPZ infinity is intentional",
            "retained_condition": "s_max / smallest retained singular value"},
        "integrator": {"method": "Forward Euler on parameters", "dt": .01, "stop": 10.,
            "points": 1001, "initial_theta": [0.] * 6, "first_step_special_handling": False},
        "project_choices": "Euler and SVD cutoff are project choices, not asserted paper implementation details",
        "postprocessing": "No normalization, symmetrization, clipping, Tikhonov or initial perturbation",
        "exact_data_read": False, "randomness": "Seed 2019 for fixed validation parameters only; deterministic evolution",
        "python": platform.python_version(), "machine": platform.machine(),
        "packages": dict(sorted((dist.metadata["Name"], dist.version) for dist in metadata.distributions())),
        "requirements_sha256": hashlib.sha256((PROJECT / "requirements.txt").read_bytes()).hexdigest(),
        "source_sha256": {str(path.relative_to(PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "code_parent_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip(),
        "checks": checks, "artifacts": artifacts,
        "limitations": ["No full-time step convergence claim", "No full-time cutoff sensitivity claim",
            "No Exact comparison, error target assessment or Figure 5 plot in Phase 5"],
    }
    with (output_dir / "run_metadata.json").open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output_directory": str(output_dir), "checks": checks}, indent=2))


if __name__ == "__main__":
    main()

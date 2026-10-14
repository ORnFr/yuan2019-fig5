"""Validate and save density Jacobians; no McLachlan or time evolution."""

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
from yuan2019_fig5.derivatives_validation import (
    check_zero_jacobian, check_derivative_samples, check_jvp_and_jit,
)


def main():
    """Run all Phase 4 checks before writing a unique NPZ and metadata record."""
    zero_check = check_zero_jacobian()
    arrays, sample_check = check_derivative_samples()
    checks = {"zero_parameter": zero_check, "samples": sample_check,
              "jvp_and_jit": check_jvp_and_jit()}
    timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    output_dir = PROJECT / "results" / f"derivatives_{timestamp}"
    output_dir.mkdir(exist_ok=False)
    sample_file = output_dir / "derivative_samples.npz"
    np.savez_compressed(sample_file, **arrays)
    with np.load(sample_file, allow_pickle=False) as loaded:
        for key, expected in arrays.items():
            np.testing.assert_array_equal(loaded[key], expected)
    sources = [Path(__file__), PROJECT / "src/yuan2019_fig5/derivatives.py",
               PROJECT / "src/yuan2019_fig5/derivatives_validation.py",
               PROJECT / "src/yuan2019_fig5/ansatz.py", PROJECT / "src/yuan2019_fig5/ansatz_validation.py",
               PROJECT / "src/yuan2019_fig5/runtime.py", PROJECT / "tests/test_derivatives.py"]
    record = {
        "status": "passed", "scope": "Phase 4 reduced-density Jacobian only; no M, V, linear solve or time step",
        "timestamp": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "command": '"$FIG5_ROOT/.venv/bin/python" "$FIG5_ROOT/scripts/check_derivatives.py"',
        "definition": "D[i,s,t]=partial rho_S[s,t] / partial theta_i; real paper parameters",
        "units": "Dimensionless angles in radians; density and derivatives dimensionless; no physical time",
        "basis": {"wires": ["a0", "a1", "s0", "s1"], "system": ["00", "01", "10", "11"]},
        "ad_method": "jax.jacfwd separately on real/imag rho; combine and move final parameter axis to first",
        "holomorphic": False, "parameter_shape": [6], "jacobian_shape": [6, 4, 4],
        "references": ["Independent NumPy/SciPy full-gate product rule and explicit full-density partial trace",
                       "Analytic zero-point commutators", "Central parameter differences at four recorded steps"],
        "postprocessing": "No normalization, symmetrization, trace correction, eigenvalue clipping or initial parameter perturbation",
        "python": platform.python_version(), "machine": platform.machine(),
        "packages": dict(sorted((dist.metadata["Name"], dist.version) for dist in metadata.distributions())),
        "requirements_sha256": hashlib.sha256((PROJECT / "requirements.txt").read_bytes()).hexdigest(),
        "source_sha256": {str(path.relative_to(PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "code_parent_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip(),
        "random_seed": 2019, "randomness": "Only eight validation parameter vectors; no stochastic evolution",
        "checks": checks, "artifacts": {"samples": {"path": str(sample_file.relative_to(PROJECT)),
            "sha256": hashlib.sha256(sample_file.read_bytes()).hexdigest(),
            "shapes": {key: list(array.shape) for key, array in arrays.items()},
            "dtypes": {key: str(array.dtype) for key, array in arrays.items()},
            "undefined_relative_errors": "Use fd_relative_defined mask; stored zeros are placeholders for undefined relative errors"}},
    }
    with (output_dir / "run_metadata.json").open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output_directory": str(output_dir), "checks": checks}, indent=2))


if __name__ == "__main__":
    main()

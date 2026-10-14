"""Validate the Figure 4 ansatz and partial trace; no VQS evolution."""

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
from yuan2019_fig5.ansatz import tc
from yuan2019_fig5.ansatz_validation import (
    sample_parameters, check_gate_conventions, check_partial_trace_examples,
    check_ansatz_states, check_wire_mapping_and_jit,
)


def main():
    """Run independent checks, then save a unique sample/report directory."""
    parameters = sample_parameters()
    states, densities, state_checks = check_ansatz_states(parameters)
    checks = {"gate_conventions": check_gate_conventions(),
              "partial_trace_examples": check_partial_trace_examples(),
              "ansatz_states": state_checks,
              "wire_mapping_and_jit": check_wire_mapping_and_jit()}
    timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    output_dir = PROJECT / "results" / f"ansatz_{timestamp}"
    output_dir.mkdir(exist_ok=False)
    sample_file = output_dir / "ansatz_samples.npz"
    np.savez_compressed(sample_file, theta=parameters, psi=states, rho=densities)
    with np.load(sample_file, allow_pickle=False) as loaded:
        for key, expected in (("theta", parameters), ("psi", states), ("rho", densities)):
            np.testing.assert_array_equal(loaded[key], expected)
    sources = [Path(__file__), PROJECT / "src/yuan2019_fig5/ansatz.py",
               PROJECT / "src/yuan2019_fig5/ansatz_validation.py", PROJECT / "src/yuan2019_fig5/runtime.py",
               PROJECT / "tests/test_ansatz.py"]
    gates_source = Path(tc.__file__).parent / "gates.py"
    record = {
        "status": "passed", "scope": "Phase 3 Figure 4 circuit/partial trace only; no derivatives or evolution",
        "timestamp": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "command": '"$FIG5_ROOT/.venv/bin/python" "$FIG5_ROOT/scripts/check_ansatz.py"',
        "source": {"doi": "10.22331/q-2019-10-07-191", "figure": 4, "pdf_page": 20, "printed_page": 20,
                   "pdf_sha256": "e142a56c57c822a45fb22aea7e7b9a47f694a9c37842796a040c9f8d4c2053b7"},
        "ansatz": {"wires": {"a0": 0, "a1": 1, "s0": 2, "s1": 3},
                   "full_basis": "|a0 a1 s0 s1>, lexicographic binary order, wire0 most significant",
                   "system_basis": ["00", "01", "10", "11"], "initial_full_state": "|0000>",
                   "zero_parameter_full_state": "|++>_A tensor |00>_S", "parameter_shape": [6],
                   "rotation_definition": "RX(theta)=exp(-i theta X); RZZ(theta)=exp(-i theta ZZ); CRX controlled on 1",
                   "library_convention": "RX/CRX angle=2*paper theta; RZZ angle=2*paper theta, half=True explicitly",
                   "gate_order": ["H(a0)", "H(a1)", "CRX(a0,s0,theta0)", "CRX(a1,s1,theta1)",
                                  "RZZ(s0,s1,theta2)", "RX(s0,theta3)", "RX(s1,theta4)", "RZZ(s0,s1,theta5)"],
                   "units": "Dimensionless angles in radians; no physical time or parameter evolution",
                   "reduction": "A[a,s]=reshape(psi,(4,4)); rho[s,t]=sum_a A[a,s]*conj(A[a,t])",
                   "postprocessing": "No normalization, Hermitian projection or positivity clipping"},
        "python": platform.python_version(), "machine": platform.machine(),
        "packages": dict(sorted((dist.metadata["Name"], dist.version) for dist in metadata.distributions())),
        "requirements_sha256": hashlib.sha256((PROJECT / "requirements.txt").read_bytes()).hexdigest(),
        "source_sha256": {str(path.relative_to(PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "installed_gates_source_sha256": hashlib.sha256(gates_source.read_bytes()).hexdigest(),
        "code_parent_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip(),
        "random_seeds": {"parameters": 2019, "partial_trace_states": 2026},
        "checks": checks, "artifacts": {"samples": {"path": str(sample_file.relative_to(PROJECT)),
                    "sha256": hashlib.sha256(sample_file.read_bytes()).hexdigest(),
                    "shapes": {"theta": list(parameters.shape), "psi": list(states.shape), "rho": list(densities.shape)},
                    "dtypes": {"theta": str(parameters.dtype), "psi": str(states.dtype), "rho": str(densities.dtype)}}},
    }
    with (output_dir / "run_metadata.json").open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output_directory": str(output_dir), "checks": checks}, indent=2))


if __name__ == "__main__":
    main()

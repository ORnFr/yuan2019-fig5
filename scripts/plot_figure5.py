"""Compare saved Phase 2/5 baselines and create Figure 5 without rerunning dynamics."""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from yuan2019_fig5.runtime import configure_project

configure_project()

import numpy as np
import matplotlib.pyplot as plt
from yuan2019_fig5.plot import load_trajectory, compare_trajectories, make_figure5
from yuan2019_fig5.vector_output import save_vector_figure


def verify_input_record(path):
    """Require input data to match its saved run metadata digest; return provenance."""
    metadata_path = path.parent / "run_metadata.json"
    record = json.loads(metadata_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    entries = [item for item in record["artifacts"].values()
               if Path(item["path"]).name == path.name]
    if len(entries) != 1 or entries[0]["sha256"] != digest:
        raise ValueError(f"Input file does not match recorded digest: {path}")
    return record, {"path": str(path.relative_to(PROJECT)), "sha256": digest,
        "metadata_path": str(metadata_path.relative_to(PROJECT)),
        "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest()}


def main():
    """Validate input provenance and observables, save comparison and SVG/PDF."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact", default="results/exact_20261013_220047_446893/exact.npz")
    parser.add_argument("--vqs", default="results/vqs_20261014_161232_948193/vqs.npz")
    arguments = parser.parse_args()
    paths = {}
    records = {}
    provenance = {}
    for label, name in (("exact", arguments.exact), ("vqs", arguments.vqs)):
        path = Path(name)
        path = (path if path.is_absolute() else PROJECT / path).resolve()
        if not path.is_relative_to(PROJECT / "results"):
            raise ValueError("Inputs must be saved results within this project.")
        paths[label] = path
        records[label], provenance[label] = verify_input_record(path)
    for key in ("H", "collapse", "gamma", "observable"):
        if records["exact"]["model"][key] != records["vqs"]["model"][key]:
            raise ValueError(f"Input model conventions differ: {key}")
    if records["exact"]["model"]["basis"] != records["vqs"]["basis"]["system"]:
        raise ValueError("Input system bases differ.")
    exact, exact_checks = load_trajectory(paths["exact"])
    vqs, vqs_checks = load_trajectory(paths["vqs"], variational=True)
    arrays, metrics = compare_trajectories(exact, vqs)
    timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    output = PROJECT / "results" / f"figure5_{timestamp}"
    output.mkdir(exist_ok=False)
    comparison_path = output / "comparison.npz"
    np.savez_compressed(comparison_path, **arrays)
    with np.load(comparison_path, allow_pickle=False) as loaded:
        for key, expected in arrays.items():
            np.testing.assert_array_equal(loaded[key], expected)
    figure = make_figure5(arrays)
    figure_paths = save_vector_figure(figure, output / "figure5")
    plt.close(figure)
    if any(name in sys.modules for name in ("qutip", "jax", "tensorcircuit")):
        raise RuntimeError("The plot entry point must not import evolution frameworks.")
    sources = [Path(__file__), PROJECT / "src/yuan2019_fig5/plot.py",
               PROJECT / "src/yuan2019_fig5/runtime.py", PROJECT / "src/yuan2019_fig5/vector_output.py",
               PROJECT / "tests/test_plot.py"]
    record = {
        "status": "Phase 6 comparison completed", "timestamp": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "command": '"$FIG5_ROOT/.venv/bin/python" "$FIG5_ROOT/scripts/plot_figure5.py"',
        "inputs": provenance, "model": records["exact"]["model"],
        "exact_solver_options": records["exact"]["solver_options"],
        "vqs_integrator": records["vqs"]["integrator"], "vqs_svd": records["vqs"]["svd"],
        "source_pdf": records["vqs"]["paper"], "input_checks": {"exact": exact_checks, "vqs": vqs_checks},
        "metrics": metrics, "display": {"marker_stride": 15, "include_first_last_marker": True,
            "main_time_range": [0, 10], "main_z0_range": [-.2, 1],
            "error_inset": "All 1001 signed differences; symmetric range accommodates actual data",
            "formats": ["svg", "pdf"], "time_shift": False, "interpolation": False, "fitting": False},
        "dynamics_rerun": False, "evolution_frameworks_imported": False,
        "python": platform.python_version(), "machine": platform.machine(),
        "packages": dict(sorted((dist.metadata["Name"], dist.version) for dist in metadata.distributions())),
        "requirements_sha256": hashlib.sha256((PROJECT / "requirements.txt").read_bytes()).hexdigest(),
        "source_sha256": {str(path.relative_to(PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "code_parent_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip(),
        "artifacts": {path.name: {"path": str(path.relative_to(PROJECT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in [comparison_path, *figure_paths]},
        "limitations": ["Error target verified on the 1001 saved samples, not all continuous times",
            "No full-time step convergence or full-state error claim",
            "Author raw VQS arrays and explicit inset sign convention unavailable; no pointwise paper-error-curve equivalence claim",
            "Published Exact line time-coordinate discrepancy remains documented separately; no correction applied"],
    }
    with (output / "run_metadata.json").open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output_directory": str(output), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()

"""Compute, validate and plot only Figure 5's Exact evolution curve."""

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
import matplotlib.pyplot as plt
from yuan2019_fig5.exact import evolve_exact, SOLVER_OPTIONS, solve_lindblad, observable_values
from yuan2019_fig5.model import build_model
from yuan2019_fig5.validation import check_model_and_generator, check_analytic_damping, check_full_evolution
from yuan2019_fig5.plot import make_exact_figure
from yuan2019_fig5.vector_output import save_vector_figure


def compare_published_line():
    """Compare independent solves to PDF vertices without changing saved times.

    The first published vertex has (t=0.01,Z0=1). In addition to direct
    coordinates, inspect the possible mapping for 1000 samples across
    [0,10] displayed at j*0.01, j=1..1000. This is an inferred plotting
    interpretation, not a claim about the authors' undisclosed source code.
    No fitting of H, damping rates, initial state or solver is performed.
    """
    source = PROJECT / "papers" / "figure5_exact_reference.json"
    reference = json.loads(source.read_text())
    plotted_times = np.asarray(reference["t"], dtype=np.float64)
    reference_values = np.asarray(reference["z0"], dtype=np.float64)
    hamiltonian, initial_density, collapse_operators, z0 = build_model()
    comparisons = {}
    for name, comparison_times in {
        "direct_published_coordinates": plotted_times,
        "one_output_interval_earlier": np.round(plotted_times - 0.01, 8),
        "hypothesis_1000_samples_across_0_to_10": np.round((plotted_times - 0.01) * 1000 / 999, 12),
    }.items():
        time_grid = np.unique(np.concatenate(([0], comparison_times)))
        densities = solve_lindblad(hamiltonian, initial_density, collapse_operators, time_grid)
        values = observable_values(densities, z0)[np.searchsorted(time_grid, comparison_times)]
        differences = np.abs(values - reference_values)
        comparisons[name] = {"max_absolute_difference": float(np.max(differences)),
                             "rms_difference": float(np.sqrt(np.mean(differences ** 2)))}
    return {"reference_file": str(source.relative_to(PROJECT)),
            "reference_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "vertices": len(plotted_times), "first_vertex": [float(plotted_times[0]), float(reference_values[0])],
            "comparisons": comparisons,
            "interpretation": "Time mapping hypothesis explains the plotted line, but author raw data/code are unavailable. Saved Exact data remain on t_n=0.01*n including t=0."}


def plot_exact(times, values, output):
    """Save the Exact-only figure as SVG/PDF using a suffix-free output path."""
    figure = make_exact_figure(times, values)
    paths = save_vector_figure(figure, output)
    plt.close(figure)
    return paths


def main():
    """Save a unique validated run with t/rho/z0, metadata and Exact-only vectors."""
    timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    output_dir = PROJECT / "results" / f"exact_{timestamp}"
    output_dir.mkdir(exist_ok=False)
    times, densities, values = evolve_exact()
    checks = {"model_and_generator": check_model_and_generator(),
              "analytic_damping": check_analytic_damping(),
              "full_evolution": check_full_evolution(times, densities, values)}
    published_comparison = compare_published_line()
    data_path = output_dir / "exact.npz"
    np.savez_compressed(data_path, t=times, rho=densities, z0=values)
    figure_paths = plot_exact(times, values, output_dir / "exact_evolution")
    loaded = np.load(data_path, allow_pickle=False)
    for key, expected in (("t", times), ("rho", densities), ("z0", values)):
        np.testing.assert_array_equal(loaded[key], expected)
    loaded.close()
    record = {
        "status": "passed", "scope": "Phase 2: Exact only; no VQS or error inset",
        "timestamp": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "command": '"$FIG5_ROOT/.venv/bin/python" "$FIG5_ROOT/scripts/run_exact.py"',
        "source": {"doi": "10.22331/q-2019-10-07-191", "model_location": "Section 6.3, Eq. (62), printed/PDF page 20", "figure_location": "Figure 5, printed/PDF page 21"},
        "model": {"H": "X tensor I + I tensor X + 0.25 Z tensor Z", "collapse": ["|0><1| tensor I", "I tensor |0><1|"],
                  "gamma": [1, 1], "initial_density": "|00><00|", "observable": "Z tensor I", "basis": ["00", "01", "10", "11"], "hbar": 1, "units": "Dimensionless time; transverse-field coefficient=1"},
        "grid": {"dt_output": 0.01, "points": 1001, "t_start": 0, "t_stop": 10, "internal_steps": "Adaptive DOP853, not Euler"},
        "solver_options": SOLVER_OPTIONS, "dtypes": {"t": str(times.dtype), "rho": str(densities.dtype), "z0": str(values.dtype)},
        "python": platform.python_version(), "machine": platform.machine(),
        "packages": dict(sorted((dist.metadata["Name"], dist.version) for dist in metadata.distributions())),
        "requirements_sha256": hashlib.sha256((PROJECT / "requirements.txt").read_bytes()).hexdigest(),
        "source_sha256": {str(path.relative_to(PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in [Path(__file__), *sorted((PROJECT / "src" / "yuan2019_fig5").glob("*.py"))]},
        "code_parent_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip(),
        "random_seed": 2019, "randomness": "Only the generator's test density uses RNG; evolution is deterministic.",
        "checks": checks, "published_line_comparison": published_comparison,
        "summary": {"z0_initial": float(values[0]), "z0_minimum": float(np.min(values)), "t_at_grid_minimum": float(times[np.argmin(values)]), "z0_at_t10": float(values[-1])},
        "artifacts": {name: {"path": str(path.relative_to(PROJECT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                      for name, path in [("data", data_path), *[(path.name, path) for path in figure_paths]]},
    }
    with (output_dir / "run_metadata.json").open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output_directory": str(output_dir), "checks": checks, "summary": record["summary"], "published_line_comparison": published_comparison}, indent=2))


if __name__ == "__main__":
    main()

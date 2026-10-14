"""Run dt=.005 through t=10, save its figure and overlay the preserved dt=.01 run."""

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
from scipy.sparse.linalg import expm_multiply
from yuan2019_fig5.vqs import evolve_vqs
from yuan2019_fig5.mclachlan_validation import audit_trajectory
from yuan2019_fig5.model import build_model
from yuan2019_fig5.exact import solve_lindblad, observable_values, density_diagnostics, SOLVER_OPTIONS
from yuan2019_fig5.validation import reference_liouvillian
from yuan2019_fig5.plot import load_trajectory, compare_trajectories, make_figure5
from yuan2019_fig5.step_comparison import compare_step_sizes, make_step_comparison


def main():
    """Keep old artifacts byte-for-byte; change only the VQS Euler step to .005."""
    old_exact_dir = PROJECT / "results/exact_20261013_220047_446893"
    old_vqs_dir = PROJECT / "results/vqs_20261014_161232_948193"
    old_figure_dir = PROJECT / "results/figure5_20261014_162632_671057"
    preserved = {str(path.relative_to(PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for directory in (old_exact_dir, old_vqs_dir, old_figure_dir)
        for path in directory.iterdir() if path.is_file()}
    exact_record = json.loads((old_exact_dir / "run_metadata.json").read_text())
    vqs_record = json.loads((old_vqs_dir / "run_metadata.json").read_text())
    assert preserved[str((old_exact_dir / "exact.npz").relative_to(PROJECT))] == exact_record["artifacts"]["data"]["sha256"]
    assert preserved[str((old_vqs_dir / "vqs.npz").relative_to(PROJECT))] == vqs_record["artifacts"]["vqs.npz"]["sha256"]
    exact_coarse, _ = load_trajectory(old_exact_dir / "exact.npz")
    coarse, _ = load_trajectory(old_vqs_dir / "vqs.npz", variational=True)

    print("Running parameter Forward Euler dt=.005, 2000 updates through t=10.", flush=True)
    fine = evolve_vqs(dt=.005, stop=10, cutoff_multiplier=1)
    vqs_checks = audit_trajectory(fine, .005)
    # This Exact solve has no variational input: only its output sampling is
    # denser. The adaptive DOP853 algorithm and tolerances are unchanged.
    times = .005 * np.arange(2001, dtype=np.float64)
    hamiltonian, rho0, collapse, observable = build_model()
    densities = solve_lindblad(hamiltonian, rho0, collapse, times)
    exact_fine = {"t": times, "rho": densities, "z0": observable_values(densities, observable)}
    generator = reference_liouvillian()
    initial_vector = np.zeros(16, dtype=np.complex128)
    initial_vector[0] = 1
    vectors = expm_multiply(generator, initial_vector, start=0, stop=10, num=2001,
                            endpoint=True, traceA=np.trace(generator))
    independent = vectors.reshape(2001, 4, 4).swapaxes(1, 2)
    reference_error = float(np.max(abs(independent - densities)))
    common_rho_error = float(np.max(abs(densities[::2] - exact_coarse["rho"])))
    assert reference_error < 1e-9 and common_rho_error < 1e-9
    physical = density_diagnostics(densities)
    assert physical["all_finite"] and physical["max_trace_error"] < 1e-10
    assert physical["max_hermiticity_frobenius_error"] < 1e-10 and physical["minimum_eigenvalue"] >= -1e-10
    comparisons, metrics = compare_step_sizes(exact_coarse, coarse, exact_fine, fine)

    timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    output = PROJECT / "results" / f"step_comparison_{timestamp}"
    output.mkdir(exist_ok=False)
    artifacts = {}
    for name, data in (("vqs_dt005.npz", fine), ("exact_dt005.npz", exact_fine),
                       ("comparison.npz", comparisons)):
        path = output / name
        np.savez_compressed(path, **data)
        with np.load(path, allow_pickle=False) as loaded:
            assert set(loaded.files) == set(data)
            for key, value in data.items():
                np.testing.assert_array_equal(loaded[key], value)
        artifacts[name] = {"path": str(path.relative_to(PROJECT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "shapes": {key: list(value.shape) for key, value in data.items()},
            "dtypes": {key: str(value.dtype) for key, value in data.items()}}
    _, fine_file_checks = load_trajectory(output / "vqs_dt005.npz", variational=True, dt=.005)
    _, exact_file_checks = load_trajectory(output / "exact_dt005.npz", dt=.005)
    fine_comparison, _ = compare_trajectories(exact_fine, fine)
    new_figure = make_figure5(fine_comparison, marker_stride=30)
    new_figure.subplots_adjust(top=.91)
    new_figure.axes[0].set_title(r"VQS parameter Euler: $dt=0.005$", fontsize=11)
    merged_figure = make_step_comparison(comparisons)
    for name, figure in (("figure5_dt005.png", new_figure), ("figure5_step_comparison.png", merged_figure)):
        path = output / name
        figure.savefig(path, dpi=240)
        plt.close(figure)
        artifacts[name] = {"path": str(path.relative_to(PROJECT)),
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    for name, digest in preserved.items():
        assert hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() == digest, name
    sources = [Path(__file__), *(PROJECT / "src/yuan2019_fig5" / name for name in
        ("runtime.py", "vqs.py", "mclachlan.py", "model.py", "ansatz.py", "derivatives.py",
         "mclachlan_validation.py", "ansatz_validation.py", "derivatives_validation.py",
         "exact.py", "validation.py", "plot.py", "step_comparison.py")), PROJECT / "tests/test_step_comparison.py"]
    record = {
        "status": "Authorized dt=.005 experiment complete; dt=.01 baseline preserved",
        "timestamp": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "command": '"$FIG5_ROOT/.venv/bin/python" "$FIG5_ROOT/scripts/run_step_comparison.py"',
        "model": vqs_record["model"], "basis": vqs_record["basis"], "units": vqs_record["units"],
        "paper": vqs_record["paper"], "svd": vqs_record["svd"],
        "vqs_integrator": {**vqs_record["integrator"], "dt": .005, "points": 2001},
        "change": "VQS parameter Euler dt .01 -> .005; no change to model, ansatz, initial parameters, SVD or integrator method",
        "exact_reference": {"solver_options": SOLVER_OPTIONS, "output_spacing": .005,
            "interpretation": "Denser output grid only; Exact uses adaptive internal steps",
            "exponential_max_rho_error": reference_error, "old_common_grid_max_rho_difference": common_rho_error,
            "reference_tolerance": 1e-9, "density_checks": physical},
        "vqs_exact_data_read": False, "vqs_trajectory_checks": vqs_checks,
        "saved_file_checks": {"vqs": fine_file_checks, "exact": exact_file_checks},
        "metrics": metrics, "preserved_file_sha256": preserved,
        "display": {"coarse_stride": 15, "fine_stride": 30, "physical_marker_spacing": .15,
            "include_endpoints": True, "error_definition": "VQS - Exact", "error_thinning": False,
            "png_dpi": 240, "interpolation": False, "time_shift": False},
        "python": platform.python_version(), "machine": platform.machine(),
        "packages": dict(sorted((dist.metadata["Name"], dist.version) for dist in metadata.distributions())),
        "requirements_sha256": hashlib.sha256((PROJECT / "requirements.txt").read_bytes()).hexdigest(),
        "source_sha256": {str(path.relative_to(PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "code_parent_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip(),
        "artifacts": artifacts,
        "limitations": ["Two step sizes do not establish convergence order or the dt->0 limit",
            "Total Z0 error includes representation and discretization effects; these are not separated",
            "No comparison to author raw data or original error inset"]}
    with (output / "run_metadata.json").open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output_directory": str(output), "metrics": metrics, "vqs_checks": vqs_checks}, indent=2))


if __name__ == "__main__":
    main()

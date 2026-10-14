"""Redraw preserved scientific figures as SVG/PDF; no dynamics or PNG conversion."""

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
from yuan2019_fig5.plot import load_trajectory, compare_trajectories, make_figure5, make_exact_figure
from yuan2019_fig5.step_comparison import compare_step_sizes, make_step_comparison
from yuan2019_fig5.vector_output import save_vector_figure


def main():
    """Keep all previous artifacts; recreate four figures from validated saved arrays."""
    directories = [PROJECT / "results" / name for name in
        ("exact_20261013_220047_446893", "vqs_20261014_161232_948193",
         "figure5_20261014_162632_671057", "step_comparison_20261014_165015_516599")]
    preserved = {str(path.relative_to(PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest()
                 for directory in directories for path in directory.iterdir() if path.is_file()}
    for directory in directories:
        record = json.loads((directory / "run_metadata.json").read_text())
        for artifact in record["artifacts"].values():
            assert hashlib.sha256((PROJECT / artifact["path"]).read_bytes()).hexdigest() == artifact["sha256"]
    exact, _ = load_trajectory(directories[0] / "exact.npz")
    coarse, _ = load_trajectory(directories[1] / "vqs.npz", variational=True)
    exact_fine, _ = load_trajectory(directories[3] / "exact_dt005.npz", dt=.005)
    fine, _ = load_trajectory(directories[3] / "vqs_dt005.npz", dt=.005, variational=True)
    baseline, baseline_metrics = compare_trajectories(exact, coarse)
    new, new_metrics = compare_trajectories(exact_fine, fine)
    combined, metrics = compare_step_sizes(exact, coarse, exact_fine, fine)
    exact_figure = make_exact_figure(exact["t"], exact["z0"])
    coarse_figure = make_figure5(baseline, marker_stride=15)
    fine_figure = make_figure5(new, marker_stride=30)
    fine_figure.subplots_adjust(top=.91)
    fine_figure.axes[0].set_title(r"VQS parameter Euler: $dt=0.005$", fontsize=11)
    merged_figure = make_step_comparison(combined)
    axis = merged_figure.axes[0]
    assert len(axis.lines[1].get_xdata()) == len(axis.lines[2].get_xdata()) == 68
    np.testing.assert_array_equal(axis.lines[1].get_xdata(), axis.lines[2].get_xdata())
    inset = axis.child_axes[0]
    np.testing.assert_array_equal(inset.lines[0].get_ydata(), combined["coarse_error"])
    np.testing.assert_array_equal(inset.lines[2].get_ydata(), combined["fine_error"])
    timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    output = PROJECT / "results" / f"vectors_{timestamp}"
    output.mkdir(exist_ok=False)
    artifacts = {}
    for name, figure in (("exact_evolution", exact_figure), ("figure5_dt001", coarse_figure),
                         ("figure5_dt005", fine_figure), ("figure5_step_comparison", merged_figure)):
        for path in save_vector_figure(figure, output / name):
            artifacts[path.name] = {"path": str(path.relative_to(PROJECT)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        plt.close(figure)
    for name, digest in preserved.items():
        assert hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() == digest, name
    sources = [Path(__file__), *(PROJECT / "src/yuan2019_fig5" / name for name in
        ("runtime.py", "plot.py", "step_comparison.py", "vector_output.py"))]
    record = {"scope": "Vector output and restored thinned layout only; no dynamics/data changes",
        "command": '"$FIG5_ROOT/.venv/bin/python" "$FIG5_ROOT/scripts/export_vector_figures.py"',
        "formats": ["svg", "pdf"], "embedded_bitmap_conversion": False,
        "raster_images_allowed": False, "preserved_file_sha256": preserved,
        "metrics": metrics, "coarse_metrics": baseline_metrics, "fine_metrics": new_metrics,
        "display": {"coarse_stride": 15, "fine_stride": 30, "markers_each": 68,
            "physical_spacing": .15, "actual_computed_points": [1001, 2001],
            "error_thinning": False, "error_definition": "VQS - Exact"},
        "python": platform.python_version(), "packages": dict(sorted(
            (dist.metadata["Name"], dist.version) for dist in metadata.distributions())),
        "source_sha256": {str(path.relative_to(PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "code_parent_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip(),
        "artifacts": artifacts}
    with (output / "run_metadata.json").open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(str(output))


if __name__ == "__main__":
    main()

"""Compare the preserved dt=.01 baseline with an explicitly authorized dt=.005 run."""

import numpy as np
from .plot import compare_trajectories, make_figure5


def compare_step_sizes(exact_coarse, coarse, exact_fine, fine):
    r"""Return native-grid errors and differences on the common coarse grid.

    Each input has float64 t,z0 (N,); fine grids have 2*N-1 samples.
    e_dt(t)=z0_VQS,dt(t)-z0_Exact(t), hbar=1, dimensionless time.
    Fine sample 2*n must equal coarse time n; no interpolation is used.
    Old errors use the preserved Exact data, fine errors use a new adaptive
    Exact solve at fine times. This two-step comparison is not an order proof.
    """
    old, old_metrics = compare_trajectories(exact_coarse, coarse)
    new, new_metrics = compare_trajectories(exact_fine, fine)
    if len(new["t"]) != 2 * len(old["t"]) - 1 or not np.array_equal(new["t"][::2], old["t"]):
        raise ValueError("Fine and coarse grids must nest exactly with factor two.")
    difference = new["z0_vqs"][::2] - old["z0_vqs"]
    index = int(np.argmax(abs(difference)))
    arrays = {f"{prefix}_{key}": value for prefix, data in (("coarse", old), ("fine", new))
              for key, value in data.items()}
    arrays["fine_minus_coarse_common"] = difference
    metrics = {"coarse": old_metrics, "fine": new_metrics,
        "common_points": len(difference),
        "fine_common_max_absolute_error": float(np.max(abs(new["error"][::2]))),
        "max_step_difference": float(abs(difference[index])),
        "time_at_max_step_difference": float(old["t"][index]),
        "rms_step_difference": float(np.sqrt(np.mean(difference**2))),
        "max_exact_common_difference": float(np.max(abs(new["z0_exact"][::2] - old["z0_exact"]))),
        "max_error_reduction_fraction": float(1 - new_metrics["max_absolute_error"] / old_metrics["max_absolute_error"])
            if old_metrics["max_absolute_error"] > 0 else None}
    return arrays, metrics


def make_step_comparison(arrays):
    """Overlay both VQS runs and their signed errors, preserving physical time.

    Main Exact line and fine error use every fine point; old error uses all
    coarse points. Displayed markers have the same .15 time spacing:
    stride15 for dt=.01 and stride30 for dt=.005, including both endpoints.
    """
    coarse = {key: arrays[f"coarse_{key}"] for key in ("t", "z0_exact", "z0_vqs", "error")}
    figure = make_figure5(coarse, marker_stride=15)
    axis = figure.axes[0]
    inset = axis.child_axes[0]
    axis.lines[0].set_data(arrays["fine_t"], arrays["fine_z0_exact"])
    axis.lines[1].set_label(r"VQS, $dt=0.01$")
    indices = np.unique(np.r_[np.arange(0, len(arrays["fine_t"]), 30), len(arrays["fine_t"]) - 1])
    axis.plot(arrays["fine_t"][indices], arrays["fine_z0_vqs"][indices], linestyle="none",
              marker="^", markersize=4, markerfacecolor="none", color="#7851a9",
              markeredgewidth=.9, label=r"VQS, $dt=0.005$")
    axis.legend(loc="lower right", framealpha=1, edgecolor="black", fancybox=False, fontsize=9)
    inset.lines[0].set(color="#f4ad00", linestyle="-.", label=r"$dt=0.01$")
    inset.plot(arrays["fine_t"], arrays["fine_error"], color="#7851a9", linestyle="--",
               linewidth=1.2, label=r"$dt=0.005$")
    bound = max(.001, float(np.ceil(max(np.max(abs(arrays["coarse_error"])),
        np.max(abs(arrays["fine_error"]))) * 1.15 / .001) * .001))
    inset.set_ylim(-bound, bound)
    inset.legend(loc="upper right", fontsize=8, framealpha=1)
    return figure

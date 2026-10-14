"""Read saved trajectories and compare Z0; no dynamics or parameter fitting."""

from pathlib import Path
import numpy as np
from .runtime import configure_project

configure_project()

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


def load_trajectory(path, *, variational=False):
    r"""Read a saved 1001-point baseline and verify its physical Z0 data.

    Required arrays: t/z0 float64 (1001,), rho complex128 (1001,4,4);
    VQS also requires theta float64 (1001,6). Basis |00>,|01>,|10>,|11>,
    hbar=1, dimensionless t_n=.01*n. Z0=Z tensor I, expectation
    z0_n=Re Tr(Z0 rho_n). No interpolation, shift or state repair.
    """
    required = ["t", "rho", "z0"] + (["theta"] if variational else [])
    with np.load(Path(path), allow_pickle=False) as loaded:
        if not set(required).issubset(loaded.files):
            raise ValueError(f"Missing trajectory arrays in {path}.")
        arrays = {name: loaded[name].copy() for name in required}
    shapes = {"t": (1001,), "rho": (1001, 4, 4), "z0": (1001,), "theta": (1001, 6)}
    for name, array in arrays.items():
        dtype = np.complex128 if name == "rho" else np.float64
        if array.shape != shapes[name] or array.dtype != dtype:
            raise ValueError(f"Invalid shape/dtype for {name}: {array.shape}, {array.dtype}.")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"Nonfinite {name} in {path}.")
    if not np.array_equal(arrays["t"], .01 * np.arange(1001)):
        raise ValueError("Baseline grid must be exactly t_n=.01*n, n=0,...,1000.")
    density = arrays["rho"]
    trace_error = float(np.max(abs(np.trace(density, axis1=1, axis2=2) - 1)))
    hermiticity_error = float(np.max(np.linalg.norm(density - density.conj().swapaxes(1, 2), axis=(1, 2))))
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(density)))
    if trace_error > 1e-10 or hermiticity_error > 1e-10 or minimum_eigenvalue < -1e-10:
        raise ValueError("Saved trajectory violates density-matrix validity tolerances.")
    # The diagonal expression is independent of the evolution's trace product.
    diagonals = np.diagonal(density, axis1=1, axis2=2)
    expectation = diagonals[:, 0] + diagonals[:, 1] - diagonals[:, 2] - diagonals[:, 3]
    observable_error = float(np.max(abs(arrays["z0"] - expectation.real)))
    if np.max(abs(expectation.imag)) > 1e-12 or observable_error > 1e-12:
        raise ValueError("Saved Z0 is inconsistent with rho or has an imaginary expectation.")
    if np.max(abs(arrays["z0"])) > 1 + 1e-10:
        raise ValueError("Z0 expectation is outside its physical range.")
    if np.max(abs(density[0] - np.diag([1, 0, 0, 0]))) > 1e-12:
        raise ValueError("Saved initial state is not |00><00|.")
    if variational and not np.array_equal(arrays["theta"][0], np.zeros(6)):
        raise ValueError("VQS baseline must start at six zero parameters.")
    checks = {"trace_max_absolute_error": trace_error,
              "hermiticity_max_frobenius_error": hermiticity_error,
              "minimum_density_eigenvalue": minimum_eigenvalue,
              "saved_z0_max_absolute_error": observable_error,
              "physical_tolerance": 1e-10, "observable_tolerance": 1e-12}
    return arrays, checks


def compare_trajectories(exact, variational):
    r"""Return full-grid signed error e_n=z0_VQS,n-z0_Exact,n and metrics.

    Inputs contain matching t,z0 float64 (N,); outputs contain t,z0_exact,
    z0_vqs,error of shape (N,). max|e_n| is over every sample, independent
    of marker thinning. RMS=sqrt(sum_n e_n^2/N); target is strictly <.01.
    This tests a sampled observable, not full-state or continuous-time error.
    """
    times = np.asarray(exact["t"])
    if times.ndim != 1 or len(times) == 0 or not np.array_equal(times, variational["t"]):
        raise ValueError("Exact and VQS must have identical nonempty time grids.")
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError("Times must be finite and strictly increasing.")
    for values in (exact["z0"], variational["z0"]):
        if np.shape(values) != times.shape or np.iscomplexobj(values) or not np.all(np.isfinite(values)):
            raise ValueError("Observables must be finite real vectors on the same grid.")
    arrays = {"t": times.copy(), "z0_exact": np.asarray(exact["z0"]).copy(),
              "z0_vqs": np.asarray(variational["z0"]).copy()}
    arrays["error"] = arrays["z0_vqs"] - arrays["z0_exact"]
    error = arrays["error"]
    index = int(np.argmax(abs(error)))
    metrics = {"points": len(times), "error_definition": "z0_vqs - z0_exact",
        "max_absolute_error": float(abs(error[index])), "max_error_index": index,
        "time_at_max_absolute_error": float(times[index]),
        "signed_error_at_max_absolute_error": float(error[index]),
        "minimum_signed_error": float(np.min(error)), "maximum_signed_error": float(np.max(error)),
        "rms_error": float(np.sqrt(np.mean(error**2))), "final_signed_error": float(error[-1]),
        "comparison_target": .01, "strict_target_met": bool(np.max(abs(error)) < .01)}
    return arrays, metrics


def make_figure5(arrays, *, marker_stride=15):
    r"""Return a Matplotlib figure with Exact line, VQS dots and signed-error inset.

    Inputs: t,z0_exact,z0_vqs,error, float64 (1001,). Marker stride affects
    only display; the Exact line and error inset use all saved points.
    Main limits follow Figure 5 (PDF/printed p21): t=[0,10], Z0=[-.2,1].
    The inset range adapts to the actual signed error, preserving every point.
    """
    if not isinstance(marker_stride, (int, np.integer)) or marker_stride < 1:
        raise ValueError("marker_stride must be a positive integer.")
    indices = np.unique(np.r_[np.arange(0, len(arrays["t"]), marker_stride), len(arrays["t"]) - 1])
    with plt.rc_context({"font.family": "DejaVu Serif", "font.size": 12,
                         "mathtext.fontset": "dejavuserif", "axes.linewidth": .8}):
        figure, axis = plt.subplots(figsize=(7.0, 5.4))
        figure.subplots_adjust(left=.13, right=.97, bottom=.14, top=.96)
        axis.plot(arrays["t"], arrays["z0_exact"], color=(.302, .745, .933),
                  linewidth=1.8, label="Exact evolution")
        axis.plot(arrays["t"][indices], arrays["z0_vqs"][indices], linestyle="none",
                  marker="o", markersize=3.4, color="#f4ad00", label="Variational simulation")
        axis.set(xlim=(0, 10), ylim=(-.2, 1), xlabel="Time", ylabel=r"$\langle Z_0\rangle$")
        axis.set_xticks(np.arange(0, 11, 2))
        axis.set_yticks(np.arange(-.2, 1.01, .2))
        axis.tick_params(direction="in", top=True, right=True)
        axis.legend(loc="lower right", framealpha=1, edgecolor="black", fancybox=False, fontsize=10)
        inset = axis.inset_axes([.25, .49, .64, .42])
        inset.plot(arrays["t"], arrays["error"], color="#70a300", linewidth=1.1, linestyle="-.")
        inset.axhline(0, color=".65", linewidth=.5)
        bound = max(.001, float(np.ceil(np.max(abs(arrays["error"])) * 1.15 / .001) * .001))
        inset.set(xlim=(0, 10), ylim=(-bound, bound), xlabel="Time", ylabel=r"$\Delta Z_0$ (VQS $-$ Exact)")
        inset.set_xticks(np.arange(0, 11, 2))
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-3, -3))
        inset.yaxis.set_major_formatter(formatter)
        inset.tick_params(direction="in", top=True, right=True, labelsize=8)
        inset.xaxis.label.set_size(9)
        inset.yaxis.label.set_size(9)
        inset.yaxis.get_offset_text().set_size(8)
    return figure

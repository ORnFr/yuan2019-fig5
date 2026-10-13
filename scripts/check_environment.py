"""Phase 1 CPU/precision smoke checks; no Figure 5 dynamics are implemented."""

import io
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from importlib import metadata
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
WORK = PROJECT / "work"
if Path(sys.prefix).resolve() != PROJECT / ".venv":
    raise RuntimeError("Run this script with this project's .venv/bin/python.")

# Set process-local paths and precision before scientific imports or arrays.
for name, path in {
    "TMPDIR": WORK / "tmp",
    "XDG_CACHE_HOME": WORK / "cache",
    "MPLCONFIGDIR": WORK / "matplotlib",
    "JAX_COMPILATION_CACHE_DIR": WORK / "cache" / "jax",
    "PYTHONPYCACHEPREFIX": WORK / "cache" / "pycache",
    "PIP_CACHE_DIR": WORK / "cache" / "pip",
}.items():
    path.mkdir(parents=True, exist_ok=True)
    os.environ[name] = str(path)
sys.pycache_prefix = os.environ["PYTHONPYCACHEPREFIX"]
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"
os.environ["MPLBACKEND"] = "Agg"
os.environ["PIP_CONFIG_FILE"] = os.devnull
os.environ["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")

import jax.numpy as jnp
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import tensorcircuit as tc

tc.set_backend("jax")
tc.set_dtype("complex128")


def import_project_qutip():
    """Import QuTiP with its import-time cache confined to this project.

    QuTiP 5.2.3 constructs ~/.qutip before exposing its path settings.
    Temporarily redirect only expanduser('~') during that import; restore
    the standard function immediately afterward. No HOME or library file
    is modified, and no physical/numerical behavior is patched.
    """
    original_expanduser = os.path.expanduser
    cache_home = WORK / "cache" / "qutip-home"
    cache_home.mkdir(parents=True, exist_ok=True)
    with patch(
        "os.path.expanduser",
        side_effect=lambda path: str(cache_home) if path == "~" else original_expanduser(path),
    ):
        import qutip
    if os.path.expanduser is not original_expanduser:
        raise RuntimeError("The temporary path redirect was not restored.")
    for path in (qutip.settings.tmproot, qutip.settings.coeffroot):
        if not Path(path).resolve().is_relative_to(WORK):
            raise RuntimeError("QuTiP was imported before project cache configuration.")
    return qutip


qutip = import_project_qutip()


def rx_state(angle):
    r"""Return R_x(a)|0> = cos(a/2)|0> - i sin(a/2)|1>.

    Input: real float64 scalar angle a in radians. Output: complex128
    vector of shape (2,) in basis (|0>, |1>). This uses the library's
    exp(-i a X/2) convention, not the paper's parameter convention.
    """
    circuit = tc.Circuit(1)
    circuit.rx(0, theta=angle)
    return circuit.state()


def z_expectation(angle):
    r"""Return <0|R_x(a)^dagger Z R_x(a)|0> = cos(a).

    Input and output are float64 scalars; a is in radians. The real
    projection is valid for this Hermitian observable and remains in JAX.
    Its analytic derivative is -sin(a). No NumPy conversion occurs here.
    """
    circuit = tc.Circuit(1)
    circuit.rx(0, theta=angle)
    return jnp.real(circuit.expectation_ps(z=[0]))


def check_precision():
    """Check CPU devices, actual dtypes and a sub-float32 increment."""
    devices = jax.devices()
    assert jax.config.x64_enabled and all(device.platform == "cpu" for device in devices)
    assert tc.backend.name == "jax"
    increment = 2.0 ** -40
    real_value = jnp.asarray(1.0, dtype=jnp.float64) + increment
    complex_value = jnp.asarray(1.0 + 1.0j, dtype=jnp.complex128) + 1j * increment
    assert real_value.dtype == jnp.float64 and complex_value.dtype == jnp.complex128
    assert float(real_value - 1.0) == increment
    assert float(jnp.imag(complex_value) - 1.0) == increment
    state = rx_state(jnp.asarray(0.37, dtype=jnp.float64))
    gradient = jax.grad(z_expectation)(jnp.asarray(0.37, dtype=jnp.float64))
    assert state.dtype == jnp.complex128 and gradient.dtype == jnp.float64
    return {
        "devices": [str(device) for device in devices],
        "jax_x64_enabled": bool(jax.config.x64_enabled),
        "tensorcircuit_backend": tc.backend.name,
        "parameter_dtype": str(real_value.dtype),
        "state_dtype": str(state.dtype),
        "gradient_dtype": str(gradient.dtype),
        "float64_epsilon": float(np.finfo(np.float64).eps),
        "preserved_increment": increment,
    }


def check_rotation_and_ad():
    """Compare a small circuit and AD with analytic and matrix references.

    All errors are maximum absolute element errors over five fixed angles.
    The complex-state derivative uses real inputs with jacfwd, without
    assuming holomorphicity. Central differences are validation only.
    """
    angles = np.array([0.0, 0.37, -0.83, 1.2, np.pi], dtype=np.float64)
    errors = {name: 0.0 for name in (
        "state", "normalization", "expectation", "gradient", "state_jacobian", "jit", "backend_gradient"
    )}
    finite_difference_errors = {str(step): 0.0 for step in (1e-4, 1e-5, 1e-6)}
    pauli_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    compiled_value_and_grad = jax.jit(jax.value_and_grad(z_expectation))
    for angle in angles:
        parameter = jnp.asarray(angle, dtype=jnp.float64)
        state = rx_state(parameter)
        reference = scipy.linalg.expm(-0.5j * angle * pauli_x)[:, 0]
        analytic_derivative = np.array([-0.5 * np.sin(angle / 2), -0.5j * np.cos(angle / 2)])
        value, gradient = jax.value_and_grad(z_expectation)(parameter)
        derivative = jax.jacfwd(rx_state)(parameter)
        backend_value, backend_gradient = tc.backend.value_and_grad(z_expectation)(parameter)
        jit_value, jit_gradient = compiled_value_and_grad(parameter)
        assert state.shape == (2,) and derivative.shape == (2,)
        assert state.dtype == derivative.dtype == jnp.complex128
        assert value.dtype == gradient.dtype == jnp.float64
        measurements = {
            "state": np.max(np.abs(np.asarray(state) - reference)),
            "normalization": abs(np.vdot(np.asarray(state), np.asarray(state)) - 1),
            "expectation": abs(float(value) - np.cos(angle)),
            "gradient": abs(float(gradient) + np.sin(angle)),
            "state_jacobian": np.max(np.abs(np.asarray(derivative) - analytic_derivative)),
            "jit": max(abs(float(jit_value) - np.cos(angle)), abs(float(jit_gradient) + np.sin(angle))),
            "backend_gradient": max(abs(float(backend_value) - np.cos(angle)), abs(float(backend_gradient) + np.sin(angle))),
        }
        for name, error in measurements.items():
            errors[name] = max(errors[name], float(error))
        for step in (1e-4, 1e-5, 1e-6):
            difference = (rx_state(parameter + step) - rx_state(parameter - step)) / (2 * step)
            error = float(np.max(np.abs(np.asarray(difference - derivative))))
            finite_difference_errors[str(step)] = max(finite_difference_errors[str(step)], error)
    for error in errors.values():
        assert error < 1e-12, errors
    difference_tolerances = {"0.0001": 1e-8, "1e-05": 1e-9, "1e-06": 2e-9}
    for step, error in finite_difference_errors.items():
        assert error < difference_tolerances[step], finite_difference_errors
    batched_values = jax.jit(jax.vmap(z_expectation))(jnp.asarray(angles))
    np.testing.assert_allclose(batched_values, np.cos(angles), atol=1e-12, rtol=0)
    return {
        "angles_radians": angles.tolist(), "absolute_tolerance": 1e-12,
        "max_absolute_errors": errors,
        "finite_difference_max_absolute_errors": finite_difference_errors,
        "finite_difference_absolute_tolerances": difference_tolerances,
        "jit_vmap_max_absolute_error": float(np.max(np.abs(np.asarray(batched_values) - np.cos(angles)))),
    }


def check_bell_circuit():
    r"""Check H(0), CNOT(0,1)|00> = (|00>+|11>)/sqrt(2).

    The state has shape (4,) in the order |00>, |01>, |10>, |11>.
    This is a two-qubit smoke check, not the Figure 4 ansatz.
    """
    circuit = tc.Circuit(2)
    circuit.h(0)
    circuit.cnot(0, 1)
    state = circuit.state()
    expected = np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2)
    assert state.dtype == jnp.complex128 and state.shape == (4,)
    error = float(np.max(np.abs(np.asarray(state) - expected)))
    assert error < 1e-12
    return {"state_dtype": str(state.dtype), "max_absolute_error": error, "absolute_tolerance": 1e-12}


def check_qutip_and_plotting():
    """Check QuTiP matrix arithmetic and an in-memory Agg PNG render."""
    state = qutip.basis(2, 0)
    matrix = state.full()
    assert matrix.shape == (2, 1) and matrix.dtype == np.complex128
    assert qutip.expect(qutip.sigmaz(), state) == 1.0
    assert state.norm() == 1.0
    figure, axes = plt.subplots(figsize=(2, 2))
    axes.plot([0, 1], [0, 1])
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png")
    plt.close(figure)
    assert buffer.getvalue().startswith(b"\x89PNG\r\n\x1a\n")
    return {"qutip_matrix_dtype": str(matrix.dtype), "qutip_expectation": 1.0,
            "matplotlib_backend": str(plt.get_backend()), "png_bytes": len(buffer.getvalue()),
            "qutip_cache": str(Path(qutip.settings.tmproot).relative_to(PROJECT))}


def main():
    """Run smoke checks and save a uniquely named reproducibility record."""
    pip_check = subprocess.run([sys.executable, "-m", "pip", "check"], cwd=PROJECT,
                               text=True, capture_output=True, check=True)
    requirements = PROJECT / "requirements.txt"
    pins = {}
    for line in requirements.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            name, version = line.split("==")
            assert metadata.version(name) == version, f"Installed version differs from pin: {name}"
            pins[name] = version
    report = {
        "status": "passed", "scope": "Phase 1 environment only; no dynamics or reproduction",
        "timestamp": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "python": platform.python_version(), "machine": platform.machine(),
        "macos": platform.mac_ver()[0], "base_python": sys.base_prefix,
        "venv": str(Path(sys.prefix).relative_to(PROJECT)),
        "packages": dict(sorted((dist.metadata["Name"], dist.version) for dist in metadata.distributions())),
        "pip_check": pip_check.stdout.strip(),
        "verified_pins": pins,
        "requirements_sha256": hashlib.sha256(requirements.read_bytes()).hexdigest(),
        "precision": check_precision(), "rotation_and_ad": check_rotation_and_ad(),
        "bell_circuit": check_bell_circuit(), "qutip_and_plotting": check_qutip_and_plotting(),
        "random_seed": None, "randomness": "None; all inputs are fixed.",
    }
    output_dir = PROJECT / "results"
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    output = output_dir / f"phase1_environment_{timestamp}.json"
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(report, indent=2))
    print(f"Report: {output}")


if __name__ == "__main__":
    main()

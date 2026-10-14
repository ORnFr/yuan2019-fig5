"""Keep runtime caches local without importing the scientific frameworks."""

import os
from pathlib import Path
import sys
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[2]
WORK = PROJECT / "work"


def configure_project():
    """Require the project venv and configure process-local CPU/cache paths."""
    if Path(sys.prefix).resolve() != PROJECT / ".venv":
        raise RuntimeError("Use this project's .venv/bin/python.")
    paths = {
        "TMPDIR": WORK / "tmp", "XDG_CACHE_HOME": WORK / "cache",
        "MPLCONFIGDIR": WORK / "matplotlib",
        "JAX_COMPILATION_CACHE_DIR": WORK / "cache" / "jax",
        "PYTHONPYCACHEPREFIX": WORK / "cache" / "pycache",
        "PIP_CACHE_DIR": WORK / "cache" / "pip",
    }
    for name, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    sys.pycache_prefix = os.environ["PYTHONPYCACHEPREFIX"]
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_ENABLE_X64"] = "True"
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["PIP_CONFIG_FILE"] = os.devnull
    os.environ["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"


def import_project_qutip():
    """Import QuTiP with its import-time cache confined to this project.

    QuTiP 5.2.3 creates ~/.qutip before exposing path settings. Redirect
    only expanduser('~') during import and immediately restore it. No HOME,
    library file or scientific behavior is changed.
    """
    configure_project()
    original_expanduser = os.path.expanduser
    cache_home = WORK / "cache" / "qutip-home"
    cache_home.mkdir(parents=True, exist_ok=True)
    with patch("os.path.expanduser", side_effect=lambda path: (
        str(cache_home) if path == "~" else original_expanduser(path)
    )):
        import qutip
    if os.path.expanduser is not original_expanduser:
        raise RuntimeError("The temporary path redirect was not restored.")
    for path in (qutip.settings.tmproot, qutip.settings.coeffroot):
        if not Path(path).resolve().is_relative_to(WORK):
            raise RuntimeError("QuTiP was imported before project cache configuration.")
    return qutip

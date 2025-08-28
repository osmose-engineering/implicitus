
"""Python bindings for Rust primitive sampling functions.

This module loads the ``core_engine`` Rust extension on demand without
triggering heavy design API imports.  It mirrors the loader used by the
``design_api`` package but lives here to avoid a circular import that caused
startup failures when the extension wasn't yet available.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import pathlib
import subprocess
import sys
from types import ModuleType


def _load_core_engine() -> ModuleType:
    """Import the compiled ``core_engine`` module, building it if needed."""

    spec = importlib.util.find_spec("core_engine.core_engine")
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    crate_dir = pathlib.Path(__file__).resolve().parents[1] / "core_engine"
    env = os.environ.copy()
    env.setdefault("PYO3_USE_ABI3_FORWARD_COMPATIBILITY", "1")
    env.setdefault("PYO3_PYTHON", sys.executable)

    subprocess.run(
        ["cargo", "build", "--features", "extension-module"],
        cwd=crate_dir,
        env=env,
        check=True,
    )


    if sys.platform.startswith("win"):
        lib_name = "core_engine.dll"
    elif sys.platform == "darwin":
        lib_name = "libcore_engine.dylib"
    else:
        lib_name = "libcore_engine.so"

    lib_path = crate_dir / "target" / "debug" / lib_name
    loader = importlib.machinery.ExtensionFileLoader(
        "core_engine.core_engine", str(lib_path)
    )
    spec = importlib.util.spec_from_loader("core_engine.core_engine", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


_core = _load_core_engine()
_sample_inside_rust = _core.sample_inside


def sample_inside(shape_spec, spacing):
    """Return seed points inside the given primitive at the specified spacing."""

    return _sample_inside_rust(shape_spec, spacing)

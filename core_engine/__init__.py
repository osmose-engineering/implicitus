"""Python package wrapper for the Rust `core_engine` extension.

This module loads the compiled extension from the sibling Rust crate,
building it on demand if necessary, and re-exports its public symbols.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import os
import pathlib
import subprocess
import sys


def _load() -> object:
    try:
        return importlib.import_module("core_engine.core_engine")
    except ModuleNotFoundError:
        crate_dir = pathlib.Path(__file__).resolve().parent
        env = os.environ.copy()
        env.setdefault("PYO3_USE_ABI3_FORWARD_COMPATIBILITY", "1")
        env.setdefault("PYO3_PYTHON", sys.executable)
        subprocess.run(
            ["cargo", "build", "--lib", "--features", "extension-module"],
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


_core = _load()

def __getattr__(name: str):
    return getattr(_core, name)

__all__ = [name for name in dir(_core) if not name.startswith("_")]

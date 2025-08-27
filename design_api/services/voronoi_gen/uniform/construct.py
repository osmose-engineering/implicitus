import numpy as np
import logging
import os
import json
from typing import Any, Dict
from pathlib import Path

from .sampler import compute_medial_axis, trace_hexagon
from core_engine import compute_uniform_cells as _compute_uniform_cells


def dump_uniform_cell_map(dump_data: Dict[str, Any]) -> None:
    """Persist uniform generation diagnostics to ``UNIFORM_CELL_DUMP.json``.

    The dump is written relative to the repository root if possible; otherwise
    the current working directory is used.  A best effort is made to ensure the
    destination directory exists and is writable.  Any failure is logged at
    ``WARNING`` level while successful writes are logged at ``INFO``.
    """

    # Determine a suitable repository root. When the package is installed the
    # source may live under ``site-packages`` where writing is disallowed. Walk
    # up the path looking for a ``logs`` directory or a ``.git`` folder; if
    # neither is found, fall back to the current working directory.
    root_candidate = Path(__file__).resolve()
    repo_root = None
    for parent in root_candidate.parents:
        if (parent / "logs").exists() or (parent / ".git").exists():
            repo_root = parent
            break
    if repo_root is None:
        repo_root = Path.cwd()

    logging.debug("REPO ROOT: %s", repo_root)

    dump_path = repo_root / "logs" / "UNIFORM_CELL_DUMP.json"
    try:
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        if not os.access(dump_path.parent, os.W_OK):
            logging.warning("Uniform cell dump path %s is not writable", dump_path)
            return
        with dump_path.open("w", encoding="utf-8") as f:
            json.dump(dump_data, f)
        logging.info("Uniform cell dump written to %s", dump_path)
    except Exception as exc:  # pragma: no cover - best effort
        logging.warning("Failed to write uniform cell dump to %s: %s", dump_path, exc)


# Expose the Rust implementation while keeping helpers available for monkeypatching
compute_uniform_cells = _compute_uniform_cells

__all__ = ["compute_uniform_cells", "dump_uniform_cell_map", "compute_medial_axis", "trace_hexagon"]

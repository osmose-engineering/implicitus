import json
import os
from pathlib import Path
import design_api.services.infill_service as infill


def _fake_generate_hex_lattice(spec, *_, **__):
    seeds = [(0.0, 0.0, 0.0)]
    vertices = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
    ]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    cells = [{"vertices": vertices, "out_of_bounds": False}]

    if os.getenv("UNIFORM_CELL_DEBUG"):
        dump_data = {
            "bbox_min": spec.get("bbox_min"),
            "bbox_max": spec.get("bbox_max"),
            "cells": cells,
            "edges": edges,
        }
        dump_path = Path("logs/UNIFORM_CELL_DUMP.json")
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        with dump_path.open("w", encoding="utf-8") as fh:
            json.dump(dump_data, fh)
    return {
        "seed_points": seeds,
        "cell_vertices": vertices,
        "edge_list": edges,
        "cells": cells,
        "bbox_min": spec.get("bbox_min"),
        "bbox_max": spec.get("bbox_max"),
        "debug": {},
    }


def test_uniform_cell_dump_logging(monkeypatch):
    monkeypatch.setenv("UNIFORM_CELL_DEBUG", "1")
    monkeypatch.setattr(infill, "generate_hex_lattice", _fake_generate_hex_lattice)

    dump_path = Path("logs/UNIFORM_CELL_DUMP.json")
    if dump_path.exists():
        dump_path.unlink()

    spec = {"bbox_min": [0, 0, 0], "bbox_max": [1, 1, 0], "spacing": 1.0}
    infill.generate_hex_lattice(spec)

    data = json.loads(dump_path.read_text())

    assert "bbox_min" in data and "bbox_max" in data
    assert all(not cell.get("out_of_bounds") for cell in data["cells"])
    vertex_count = len(data["cells"][0]["vertices"])
    assert all(0 <= i < vertex_count and 0 <= j < vertex_count for i, j in data["edges"])

    dump_path.unlink()

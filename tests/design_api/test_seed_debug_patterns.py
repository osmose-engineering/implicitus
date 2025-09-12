import json
from pathlib import Path

from fastapi.testclient import TestClient


def test_seed_debug_log_records_seeds(monkeypatch):
    monkeypatch.setenv("IMPLICITUS_DEBUG_SEEDS", "1")
    import design_api.main as main
    monkeypatch.setattr(main, "DEBUG_SEEDS", True)

    log_file = Path(main.SEED_DEBUG_LOG)
    if log_file.exists():
        log_file.unlink()

    def _fake_voronoi(spec):
        seeds = spec.get("seed_points", [])
        return {"seed_points": seeds, "edge_list": [], "vertices": seeds}

    def _fake_hex(spec):
        seeds = spec.get("seed_points", [])
        return {"seed_points": seeds, "edge_list": [], "cell_vertices": [], "cells": []}

    monkeypatch.setattr(main, "generate_voronoi", _fake_voronoi)
    monkeypatch.setattr(main, "generate_hex_lattice", _fake_hex)
    monkeypatch.setattr(main, "validate_proto", lambda x: x)

    client = TestClient(main.app)

    patterns = [
        ("voronoi", [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        ("hex", [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
    ]

    for pattern, seeds in patterns:
        node = {
            "primitive": {"box": {"size": {"x": 1, "y": 1, "z": 1}}},
            "modifiers": {
                "infill": {
                    "pattern": pattern,
                    "seed_points": seeds,
                    "bbox_min": [0, 0, 0],
                    "bbox_max": [1, 1, 1],
                    "mode": "organic",
                }
            },
        }

        def _fake_review_request(_req, _node=node):
            return ([_node], "ok")

        monkeypatch.setattr(main, "review_request", _fake_review_request)

        resp = client.post("/design/review", json={"raw": ""})
        assert resp.status_code == 200
        sid = resp.json()["sid"]
        resp2 = client.post("/design/submit", params={"sid": sid}, json={})
        assert resp2.status_code == 200
        del main.design_states[sid]

    with open(log_file) as f:
        entries = [json.loads(line) for line in f]

    assert entries[0]["spec"]["spec"][0]["modifiers"]["infill"]["pattern"] == "voronoi"
    assert entries[0]["spec"]["spec"][0]["modifiers"]["infill"]["seed_points"] == patterns[0][1]
    assert entries[1]["spec"]["spec"][0]["modifiers"]["infill"]["pattern"] == "hex"
    assert entries[1]["spec"]["spec"][0]["modifiers"]["infill"]["seed_points"] == patterns[1][1]

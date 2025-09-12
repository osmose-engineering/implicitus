import design_api.main as main


def test_review_returns_edge_list_for_voronoi_infill(client, monkeypatch):
    """Ensure `/design/review` surfaces `edge_list` results."""

    seeds = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
    edges = [[0, 1]]

    # Stub the lattice generator to return a deterministic edge list.
    def _fake_hex(spec):
        return {
            "seed_points": seeds,
            "cell_vertices": [],
            "edge_list": edges,
            "cells": [],
            "bbox_min": spec.get("bbox_min"),
            "bbox_max": spec.get("bbox_max"),
        }

    monkeypatch.setattr(main, "generate_hex_lattice", _fake_hex)

    node = {
        "primitive": {"box": {"size": {"x": 1, "y": 1, "z": 1}}},
        "modifiers": {
            "infill": {
                "pattern": "voronoi",
                "seed_points": seeds,
                "bbox_min": [0, 0, 0],
                "bbox_max": [1, 1, 1],
                "mode": "uniform",
            }
        },
    }

    def _fake_review_request(_req):
        return ([node], "ok")

    monkeypatch.setattr(main, "review_request", _fake_review_request)

    resp = client.post("/design/review", json={"raw": ""})
    assert resp.status_code == 200
    infill = resp.json()["spec"][0]["modifiers"]["infill"]
    assert infill["edge_list"] == edges


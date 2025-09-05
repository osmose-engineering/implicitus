from fastapi.testclient import TestClient

from design_api.main import app, design_states, DesignState


def test_submit_infill_preserves_seed_points(monkeypatch):
    # Stub lattice generator to return seeds verbatim
    def _fake_voronoi(spec):
        seeds = spec.get("seed_points", [])
        return {"seed_points": seeds, "vertices": seeds, "edge_list": []}

    monkeypatch.setattr("design_api.main.generate_voronoi", _fake_voronoi)
    monkeypatch.setattr("design_api.main.validate_proto", lambda x: x)

    client = TestClient(app)

    seeds = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
    node = {
        "primitive": {"box": {"size": {"x": 1, "y": 1, "z": 1}}},
        "modifiers": {
            "infill": {
                "pattern": "voronoi",
                "seed_points": seeds,
                "bbox_min": [0, 0, 0],
                "bbox_max": [1, 1, 1],
                "mode": "organic",
            }
        },
    }

    sid = "test-session"
    design_states[sid] = DesignState(draft_spec=[node], seed_cache={0: seeds})

    resp = client.post("/design/submit", params={"sid": sid}, json={})
    assert resp.status_code == 200
    locked = resp.json()["locked_model"]
    lattice = locked["root"]["children"][0]["root"]["children"][1]["primitive"]["lattice"]
    assert lattice["seed_points"] == seeds

    del design_states[sid]

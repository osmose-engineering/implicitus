import design_api.main as main
from design_api.main import design_states, DesignState


def test_update_infill_nested_under_modifiers(client, monkeypatch):
    # Prepare initial state with simple primitive
    sid = "test-session"
    node = {"primitive": {"box": {"size": {"x": 1, "y": 1, "z": 1}}}}
    design_states[sid] = DesignState(draft_spec=[node])

    # Adapter returns infill at the top level
    def _fake_update_request(_sid, _spec, _raw):
        return [
            {"primitive": node["primitive"], "infill": {"pattern": "gyroid"}}
        ], "summary"

    monkeypatch.setattr(main, "update_request", _fake_update_request)

    resp = client.post(
        "/design/update",
        json={"sid": sid, "spec": [node], "raw": "add gyroid infill"},
    )
    assert resp.status_code == 200
    spec = resp.json()["spec"]
    assert spec[0]["modifiers"]["infill"]["pattern"] == "gyroid"

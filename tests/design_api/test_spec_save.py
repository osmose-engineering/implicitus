import json
from pathlib import Path

import design_api.main as main
from design_api.main import DesignState, design_states


def test_update_saves_spec_file(client, monkeypatch):
    """Calling the update endpoint should persist the rendered spec to disk."""

    sid = "spec-save-session"
    design_states[sid] = DesignState(draft_spec=[])

    # Stub the adapter to return a simple spec
    def _fake_update_request(_sid, _spec, _raw):
        return ([{"primitive": {"cube": {"size": 1}}}], "summary")

    monkeypatch.setattr(main, "update_request", _fake_update_request)

    resp = client.post(
        "/design/update", params={"sid": sid}, json={"raw": "add cube", "spec": []}
    )
    assert resp.status_code == 200

    saved_path = main.RENDERED_SPEC_DIR / f"{sid}.json"
    assert saved_path.exists(), "Expected spec file to be written"

    with open(saved_path) as f:
        data = json.load(f)
    assert data["spec"][0]["primitive"]["cube"]["size"] == 1

    # Clean up
    saved_path.unlink()

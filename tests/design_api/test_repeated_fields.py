import httpx
from design_api.services.mapping import map_to_proto_dict

# Reuse simple HTTP client stubs similar to test_slice

class DummyResponse:
    def __init__(self, data, status_code=200):
        self._data = data
        self.status_code = status_code

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=self)


class DummyClient:
    def __init__(self, capture, status_code=200, data=None):
        self.capture = capture
        self.status_code = status_code
        self.data = {"ok": True} if data is None else data

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def post(self, url, json):
        self.capture["url"] = url
        self.capture["json"] = json
        return DummyResponse(self.data, self.status_code)


def test_mapping_inserts_repeated_fields():
    spec = {
        "primitive": {"sphere": {"radius": 1.0}},
        "modifiers": {
            "infill": {
                "pattern": "voronoi",
                "cells": [{"vertices": [[0, 0, 0]], "faces": [{}]}],
            },
            "boolean_op": {"op": "union", "shape_node": {"shape": "sphere", "size_mm": 1}},
        },
    }
    mapped = map_to_proto_dict(spec)
    lattice = mapped["root"]["children"][0]["children"][1]["primitive"]["lattice"]
    assert lattice["seed_points"] == []
    assert lattice["edges"] == []
    assert lattice["cells"][0]["faces"][0]["vertex_indices"] == []
    assert mapped["root"]["booleanOp"]["union"]["nodes"] == []


def test_slice_preserves_repeated_fields(client, monkeypatch):
    capture = {}
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: DummyClient(capture))
    model = {
        "id": "abc",
        "version": 1,
        "root": {
            "primitive": {"sphere": {"radius": 1.0}},
            "modifiers": {"infill": {"pattern": "voronoi"}},
        },
    }
    client.post("/models", json=model)
    resp = client.get("/models/abc/slices?layer=0")
    assert resp.status_code == 200
    infill = capture["json"]["model"]["root"]["modifiers"][0]["infill"]
    assert infill["seed_points"] == []
    assert infill["edges"] == []
    assert infill["cells"] == []

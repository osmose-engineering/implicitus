import httpx

from design_api.main import SPEC_VERSION, models


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


def test_slice_defaults_version(client, monkeypatch):
    capture = {}
    monkeypatch.setattr(httpx, "AsyncClient", lambda *args, **kwargs: DummyClient(capture))
    client.post("/models", json={"id": "abc"})
    resp = client.get("/models/abc/slices?layer=0")
    assert resp.status_code == 200
    assert models["abc"]["version"] == SPEC_VERSION


def test_slice_incomplete_lattice_returns_400(client, monkeypatch):
    capture = {}
    monkeypatch.setattr(httpx, "AsyncClient", lambda *args, **kwargs: DummyClient(capture))
    # Bypass heavy validation
    monkeypatch.setattr("design_api.main.validate_proto", lambda m, **k: m)
    model = {
        "id": "abc",
        "version": SPEC_VERSION,
        "root": {
            "children": [
                {
                    "primitive": {"box": {"size": {"x": 1, "y": 1, "z": 1}}},
                    "modifiers": {"infill": {"pattern": "hex"}},
                }
            ]
        },
    }
    client.post("/models", json=model)
    resp = client.get("/models/abc/slices?layer=0")
    assert resp.status_code == 400
    assert capture == {}


def test_slice_generates_lattice_when_missing(client, monkeypatch):
    capture = {}
    monkeypatch.setattr(httpx, "AsyncClient", lambda *args, **kwargs: DummyClient(capture))
    monkeypatch.setattr("design_api.main.validate_proto", lambda m, **k: m)
    monkeypatch.setattr("design_api.main.MessageToDict", lambda m, **k: m)

    def _fake_hex(spec):
        return {
            "cell_vertices": [[0, 0, 0]],
            "edge_list": [[0, 0]],
            "cells": [],
        }

    monkeypatch.setattr("design_api.main.generate_hex_lattice", _fake_hex)

    model = {
        "id": "abc",
        "version": SPEC_VERSION,
        "root": {
            "children": [
                {
                    "primitive": {"box": {"size": {"x": 1, "y": 1, "z": 1}}},
                    "modifiers": {
                        "infill": {"pattern": "hex", "seed_points": [[0, 0, 0]]}
                    },
                }
            ]
        },
    }

    client.post("/models", json=model)
    resp = client.get("/models/abc/slices?layer=0")
    assert resp.status_code == 200
    assert capture["json"]["cell_vertices"] == [[0, 0, 0]]
    assert capture["json"]["edge_list"] == [[0, 0]]


def test_slice_empty_children_round_trip(client, monkeypatch):
    capture = {}
    monkeypatch.setattr(httpx, "AsyncClient", lambda *args, **kwargs: DummyClient(capture))
    model = {
        "id": "abc",
        "version": SPEC_VERSION,
        "root": {"children": []},
    }
    client.post("/models", json=model)

    resp = client.get("/models/abc")
    assert resp.status_code == 200
    assert resp.json()["root"]["children"] == []

    resp = client.get("/models/abc/slices?layer=0")
    assert resp.status_code == 200
    assert capture["json"]["model"]["root"]["children"] == []


def test_slice_sets_empty_modifiers(client, monkeypatch):
    capture = {}
    monkeypatch.setattr(httpx, "AsyncClient", lambda *args, **kwargs: DummyClient(capture))
    model = {
        "id": "abc",
        "version": SPEC_VERSION,
        "root": {
            "children": [
                {"primitive": {"box": {"size": {"x": 1, "y": 1, "z": 1}}}}
            ]
        },
    }
    client.post("/models", json=model)

    resp = client.get("/models/abc/slices?layer=0")
    assert resp.status_code == 200
    root = capture["json"]["model"]["root"]
    assert root["modifiers"] == []
    assert root["children"][0]["modifiers"] == []


def test_slice_sets_empty_constraints(client, monkeypatch):
    capture = {}
    monkeypatch.setattr(httpx, "AsyncClient", lambda *args, **kwargs: DummyClient(capture))
    model = {
        "id": "abc",
        "version": SPEC_VERSION,
        "root": {
            "children": [
                {"primitive": {"box": {"size": {"x": 1, "y": 1, "z": 1}}}}
            ]
        },
    }
    client.post("/models", json=model)

    resp = client.get("/models/abc/slices?layer=0")
    assert resp.status_code == 200
    root = capture["json"]["model"]["root"]
    assert root["constraints"] == []
    assert root["children"][0]["constraints"] == []


def test_slice_top_level_empty_constraints(client, monkeypatch):
    capture = {}
    monkeypatch.setattr(httpx, "AsyncClient", lambda *args, **kwargs: DummyClient(capture))
    model = {
        "id": "abc",
        "version": SPEC_VERSION,
        "root": {"children": []},
        "constraints": [],
    }
    client.post("/models", json=model)

    resp = client.get("/models/abc/slices?layer=0")
    assert resp.status_code == 200
    assert capture["json"]["model"]["constraints"] == []


def test_slice_sets_modifier_constraints(client, monkeypatch):
    capture = {}
    monkeypatch.setattr(httpx, "AsyncClient", lambda *args, **kwargs: DummyClient(capture))
    monkeypatch.setattr("design_api.main.validate_proto", lambda m, **k: m)
    def _msg_to_dict(m, **k):
        import copy

        def _convert(o):
            if isinstance(o, dict) and isinstance(o.get("modifiers"), dict):
                o["modifiers"] = [{k: v} for k, v in o["modifiers"].items()]
                for mod in o["modifiers"]:
                    _convert(mod)
            elif isinstance(o, dict):
                for v in o.values():
                    _convert(v)
            elif isinstance(o, list):
                for item in o:
                    _convert(item)

        res = copy.deepcopy(m)
        _convert(res)
        return res

    monkeypatch.setattr("design_api.main.MessageToDict", _msg_to_dict)
    monkeypatch.setattr(
        "design_api.main.generate_hex_lattice",
        lambda spec: {"cell_vertices": [[0, 0, 0]], "edge_list": [], "cells": []},
    )
    model = {
        "id": "abc",
        "version": SPEC_VERSION,
        "root": {
            "modifiers": {
                "infill": {
                    "pattern": "hex",
                    "modifiers": {"shell": {"thickness": 1.0}},
                }
            }
        },
    }
    client.post("/models", json=model)

    resp = client.get("/models/abc/slices?layer=0")
    assert resp.status_code == 200
    mod = capture["json"]["model"]["root"]["modifiers"][0]
    assert mod["constraints"] == []
    nested = mod["infill"]["modifiers"][0]
    assert nested["constraints"] == []


def test_slice_error_cors_headers(client):
    resp = client.get(
        "/models/missing/slices?layer=0",
        headers={"Origin": "http://localhost:3000"},
    )
    assert resp.status_code == 404
    assert resp.headers["access-control-allow-origin"] == "http://localhost:3000"

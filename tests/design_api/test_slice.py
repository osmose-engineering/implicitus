import httpx
from fastapi import HTTPException


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


def test_slice_forwards_params(client, monkeypatch):
    capture = {}
    monkeypatch.setattr(httpx, "AsyncClient", lambda *args, **kwargs: DummyClient(capture))
    client.post("/models", json={"id": "abc", "version": 1})
    resp = client.get(
        "/models/abc/slices",
        params={
            "layer": "0.5",
            "x_min": "-2",
            "x_max": "2",
            "y_min": "-3",
            "y_max": "3",
            "nx": "10",
            "ny": "20",
        },
    )
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    assert capture["url"] == "http://127.0.0.1:4000/slice"
    assert capture["json"] == {
        "model": {"id": "abc"},
        "layer": 0.5,
        "x_min": -2.0,
        "x_max": 2.0,
        "y_min": -3.0,
        "y_max": 3.0,
        "nx": 10,
        "ny": 20,
    }


def test_slice_forwards_bbox(client, monkeypatch):
    capture = {}
    monkeypatch.setattr(httpx, "AsyncClient", lambda *args, **kwargs: DummyClient(capture))
    client.post(
        "/models",
        json={
            "id": "abc",
            "version": 1,
            "bbox_min": [0, 1, 2],
            "bbox_max": [3, 4, 5],
        },
    )
    resp = client.get("/models/abc/slices?layer=1.0")
    assert resp.status_code == 200
    assert capture["json"]["bbox_min"] == [0, 1, 2]
    assert capture["json"]["bbox_max"] == [3, 4, 5]


def test_slice_uses_defaults(client, monkeypatch):
    capture = {}
    monkeypatch.setattr(httpx, "AsyncClient", lambda *args, **kwargs: DummyClient(capture))
    client.post("/models", json={"id": "abc", "version": 1})
    resp = client.get("/models/abc/slices?layer=2.0")
    assert resp.status_code == 200
    assert capture["json"] == {
        "model": {"id": "abc"},
        "layer": 2.0,
        "x_min": -1.0,
        "x_max": 1.0,
        "y_min": -1.0,
        "y_max": 1.0,
        "nx": 50,
        "ny": 50,
    }


def test_slice_surfaces_errors(client, monkeypatch):
    class ErrorClient(DummyClient):
        async def post(self, url, json):
            raise httpx.HTTPError("boom")

    monkeypatch.setattr(httpx, "AsyncClient", lambda *args, **kwargs: ErrorClient({}))
    client.post("/models", json={"id": "abc", "version": 1})
    resp = client.get("/models/abc/slices?layer=1.0")
    assert resp.status_code == 500
    assert "Slicing service failure" in resp.json()["detail"]


def test_slice_allows_extra_infill_fields(client, monkeypatch):
    capture = {}
    monkeypatch.setattr(httpx, "AsyncClient", lambda *args, **kwargs: DummyClient(capture))
    client.post(
        "/models",
        json={
            "id": "abc",
            "version": 1,
            "root": {
                "children": [
                    {
                        "primitive": {"sphere": {"radius": 1.0}},

                          "modifiers": {
                              "infill": {
                                  "pattern": "hex",
                                  "cell_vertices": [],
                                  "edge_list": [],
                                  "seed_points": [[0, 0, 0]],
                                  "num_points": 5,
                                  "cells": [],
                              }
                          },
                      }
                  ]
              },
          },
      )


    resp = client.get("/models/abc/slices?layer=0")
    assert resp.status_code == 200


def test_slice_with_full_spec_returns_expected_fields(client, monkeypatch):
    capture = {}
    slice_resp = {
        "contours": [[[0, 0], [1, 0], [1, 1], [0, 1]]],
        "segments": [],
        "debug": {"seed_count": 0},
    }
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda *args, **kwargs: DummyClient(capture, data=slice_resp),
    )
    full_spec = {
        "id": "abc",
        "version": 1,
        "root": {
            "children": [
                {
                    "primitive": {"sphere": {"radius": 1.0}},
                    "modifiers": {
                        "infill": {
                            "pattern": "hex",
                            "cell_vertices": [],
                            "edge_list": [],
                            "seed_points": [[0, 0, 0]],
                            "num_points": 5,
                            "cells": [],
                        }
                    },
                }
            ]
        },
    }
    client.post("/models", json=full_spec)
    resp = client.get("/models/abc/slices?layer=0")
    assert resp.status_code == 200
    body = resp.json()
    assert body["contours"] == slice_resp["contours"]
    assert body["segments"] == []
    assert body["debug"]["seed_count"] == 0


def test_slice_forwards_cells(client, monkeypatch):
    capture = {}
    cells_raw = [
        {
            "vertices": [[0.1, 0.2, 0.3], [1.0, 1.1, 1.2], [2.0, 2.1, 2.2]],
            "faces": [[0, 1, 2]],
        }
    ]

    from design_api.services.validator import validate_model_spec as real_validate

    def check_validate(model, ignore_unknown_fields=True):
        import json as _json
        assert "cells" not in _json.dumps(model)
        return real_validate(model, ignore_unknown_fields=ignore_unknown_fields)

    monkeypatch.setattr("design_api.main.validate_proto", check_validate)
    monkeypatch.setattr(httpx, "AsyncClient", lambda *args, **kwargs: DummyClient(capture))

    client.post(
        "/models",
        json={
            "id": "abc",
            "version": 1,
            "root": {
                "children": [
                    {
                        "primitive": {"sphere": {"radius": 1.0}},
                        "modifiers": {"infill": {"pattern": "hex", "cells": cells_raw}},
                    }
                ]
            },
        },
    )

    resp = client.get("/models/abc/slices?layer=0")
    assert resp.status_code == 200
    expected = [
        {
            "vertices": [
                {"x": 0.1, "y": 0.2, "z": 0.3},
                {"x": 1.0, "y": 1.1, "z": 1.2},
                {"x": 2.0, "y": 2.1, "z": 2.2},
            ],
            "faces": [{"vertex_indices": [0, 1, 2]}],
        }
    ]
    assert capture["json"]["cells"] == expected


def test_slice_accepts_dict_cells(client, monkeypatch):
    capture = {}
    cells_raw = {
        "0": {
            "vertices": [[0.1, 0.2, 0.3], [1.0, 1.1, 1.2], [2.0, 2.1, 2.2]],
            "faces": [[0, 1, 2]],
        }
    }

    from design_api.services.validator import validate_model_spec as real_validate

    def check_validate(model, ignore_unknown_fields=True):
        import json as _json
        assert "cells" not in _json.dumps(model)
        return real_validate(model, ignore_unknown_fields=ignore_unknown_fields)

    monkeypatch.setattr("design_api.main.validate_proto", check_validate)
    monkeypatch.setattr(httpx, "AsyncClient", lambda *args, **kwargs: DummyClient(capture))

    client.post(
        "/models",
        json={
            "id": "abc",
            "version": 1,
            "root": {
                "children": [
                    {
                        "primitive": {"sphere": {"radius": 1.0}},
                        "modifiers": {"infill": {"pattern": "hex", "cells": cells_raw}},
                    }
                ]
            },
        },
    )

    resp = client.get("/models/abc/slices?layer=0")
    assert resp.status_code == 200
    expected = [
        {
            "vertices": [
                {"x": 0.1, "y": 0.2, "z": 0.3},
                {"x": 1.0, "y": 1.1, "z": 1.2},
                {"x": 2.0, "y": 2.1, "z": 2.2},
            ],
            "faces": [{"vertex_indices": [0, 1, 2]}],
        }
    ]
    assert capture["json"]["cells"] == expected


def test_slice_missing_model_returns_404(client):
    resp = client.get("/models/missing/slices?layer=1.0")
    assert resp.status_code == 404


def test_slice_missing_layer_returns_422(client):
    client.post("/models", json={"id": "abc", "version": 1})
    resp = client.get("/models/abc/slices")
    assert resp.status_code == 422


def test_slice_validation_error_includes_fields(client, monkeypatch):
    def bad_validate(model, ignore_unknown_fields=True):
        raise HTTPException(
            status_code=400,
            detail={
                "detail": "bad model",
                "offending_fields": ["foo", "bar"],
            },
        )

    monkeypatch.setattr("design_api.main.validate_proto", bad_validate)
    client.post("/models", json={"id": "abc", "version": 1, "root": {}})
    resp = client.get("/models/abc/slices?layer=0")
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "bad model" in detail
    assert "foo" in detail and "bar" in detail


def test_slice_validation_error_includes_tip(client, monkeypatch):
    def bad_validate(model, ignore_unknown_fields=True):
        raise HTTPException(
            status_code=400,
            detail={"detail": "bad model", "tip": "use snake_case"},
        )

    monkeypatch.setattr("design_api.main.validate_proto", bad_validate)
    client.post("/models", json={"id": "abc", "version": 1, "root": {}})
    resp = client.get("/models/abc/slices?layer=0")
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "bad model" in detail
    assert "use snake_case" in detail

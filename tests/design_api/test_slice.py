import httpx
import pytest
from fastapi.testclient import TestClient
from design_api.main import app, models


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
    def __init__(self, capture, status_code=200):
        self.capture = capture
        self.status_code = status_code

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def post(self, url, json):
        self.capture["url"] = url
        self.capture["json"] = json
        return DummyResponse({"ok": True}, self.status_code)


def test_slice_forwards_params(monkeypatch):
    capture = {}
    monkeypatch.setattr(httpx, "AsyncClient", lambda *args, **kwargs: DummyClient(capture))
    models.clear()
    models["abc"] = {"id": "abc"}
    client = TestClient(app)
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


def test_slice_uses_defaults(monkeypatch):
    capture = {}
    monkeypatch.setattr(httpx, "AsyncClient", lambda *args, **kwargs: DummyClient(capture))
    models.clear()
    models["abc"] = {"id": "abc"}
    client = TestClient(app)
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


def test_slice_surfaces_errors(monkeypatch):
    class ErrorClient(DummyClient):
        async def post(self, url, json):
            raise httpx.HTTPError("boom")

    monkeypatch.setattr(httpx, "AsyncClient", lambda *args, **kwargs: ErrorClient({}))
    models.clear()
    models["abc"] = {"id": "abc"}
    client = TestClient(app)
    resp = client.get("/models/abc/slices?layer=1.0")
    assert resp.status_code == 500
    assert "Slicing service failure" in resp.json()["detail"]

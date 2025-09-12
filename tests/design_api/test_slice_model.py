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

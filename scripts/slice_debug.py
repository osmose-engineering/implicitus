import httpx

BASE_URL = "http://127.0.0.1:8000"


def main() -> None:
    model = {"id": "debug-model", "name": "debug"}
    try:
        print(f"POST {BASE_URL}/models -> {model}")
        resp = httpx.post(f"{BASE_URL}/models", json=model)
        print(f"Status: {resp.status_code}")
        print(f"Response: {resp.text}")
        resp.raise_for_status()
    except Exception as exc:
        print(f"Error posting model: {exc}")
        return

    model_id = resp.json().get("id", model["id"])
    slice_url = f"{BASE_URL}/models/{model_id}/slices"
    params = {"layer": 0}
    try:
        print(f"GET {slice_url} params={params}")
        resp = httpx.get(slice_url, params=params)
        print(f"Status: {resp.status_code}")
        print(f"Response: {resp.text}")
        resp.raise_for_status()
    except Exception as exc:
        print(f"Error fetching slice: {exc}")


if __name__ == "__main__":
    main()

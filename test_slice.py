import requests, json

# 1) register
model = {
  "id":"test_sphere",
  "root":{
    "primitive":{
      "sphere":{"radius":1.0}
    }
  }
}
r = requests.post("http://localhost:8000/models", json=model)
assert r.ok and r.json().get("id") == "test_sphere"

# 2) slice
params = {"layer":"0.0", "nx":"5", "ny":"5"}
r2 = requests.get("http://localhost:8000/models/test_sphere/slices", params=params)
data = r2.json()
assert "contours" in data and isinstance(data["contours"], list)
assert len(data["contours"]) > 0, "Expected at least one contour"
print(json.dumps(data, indent=2))
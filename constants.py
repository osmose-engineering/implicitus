import json
from pathlib import Path

_constants_path = Path(__file__).with_name("constants.json")
with _constants_path.open(encoding="utf-8") as f:
    _cfg = json.load(f)

MAX_VORONOI_SEEDS: int = int(_cfg["MAX_VORONOI_SEEDS"])

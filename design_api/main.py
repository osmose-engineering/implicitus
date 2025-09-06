import os
import sys
import logging
from pathlib import Path

# Ensure repository root and its parent are on PYTHONPATH so sibling packages import cleanly
ROOT = Path(__file__).resolve().parent.parent
PARENT = ROOT.parent
for p in (ROOT, PARENT):
    if str(p) not in sys.path:
        sys.path.append(str(p))

LOG_LEVEL_NAME = os.getenv("IMPLICITUS_LOG_LEVEL", "DEBUG").upper()
LOG_LEVEL = getattr(logging, LOG_LEVEL_NAME, logging.DEBUG)
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)

from pydantic import Field
import time
import json
import copy
import traceback
import uuid
from json.decoder import JSONDecodeError
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Any, Dict, List, Literal
import httpx
from dataclasses import dataclass, field
from design_api.services.json_cleaner import clean_llm_output
from design_api.services.llm_service import generate_design_spec
from design_api.services.mapping import map_primitive as map_to_proto_dict
from design_api.services.validator import validate_model_spec as validate_proto
from ai_adapter.csg_adapter import review_request, generate_summary, update_request
from design_api.services.infill_service import (
    generate_hex_lattice,
    generate_voronoi,
)
from design_api.services.seed_utils import resolve_seed_spec

DEBUG_SEEDS = bool(os.getenv("IMPLICITUS_DEBUG_SEEDS"))
LOG_DIR = Path(__file__).parent
CONVERSATION_LOG = LOG_DIR / "conversation_log.jsonl"
SEED_DEBUG_LOG = LOG_DIR / "seed_debug.log"

@dataclass
class DesignState:
    draft_spec: list
    locked_model: Optional[dict] = None
    seed_cache: Dict[int, List[List[float]]] = field(default_factory=dict)

# session store: session_id -> DesignState
design_states: dict[str, DesignState] = {}

# in-memory model storage: model_id -> model
models: dict[str, dict] = {}

def log_turn(session_id: str, turn_type: str, raw: str, spec: list, summary: Optional[str] = None, question: Optional[str] = None):
    entry = {
        "session": session_id,
        "timestamp": time.time(),
        "type": turn_type,
        "raw": raw,
        "spec": spec,
    }

    debug_entry = None
    if DEBUG_SEEDS:
        debug_entry = copy.deepcopy(entry)

    # remove large fields from infill modifiers to keep logs small
    scrubbed_spec = []
    for node in entry["spec"]:
        node_copy = copy.deepcopy(node)
        mods = node_copy.get("modifiers")
        if mods and isinstance(mods.get("infill"), dict):
            mods["infill"].pop("seed_points", None)
            mods["infill"].pop("cell_vertices", None)
            mods["infill"].pop("edge_list", None)
            mods["infill"].pop("vertices", None)
        scrubbed_spec.append(node_copy)
    entry["spec"] = scrubbed_spec
    if summary is not None:
        entry["summary"] = summary
    if question is not None:
        entry["question"] = question

    with open(CONVERSATION_LOG, "a") as f:
        f.write(
            json.dumps(
                entry,
                default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o),
            )
            + "\n"
        )

    if debug_entry is not None:
        with open(SEED_DEBUG_LOG, "a") as f:
            f.write(
                json.dumps(
                    debug_entry,
                    default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o),
                )
                + "\n"
            )

app = FastAPI(title="Implicitus Design API", debug=True)

# Allow local front-end to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/models", response_model=dict)
async def store_model(model: Dict[str, Any]):
    model_id = model.get("id")
    if not model_id:
        raise HTTPException(status_code=400, detail="Missing model.id")
    models[model_id] = model
    return {"id": model_id}


@app.get("/models/{model_id}", response_model=dict)
async def get_model(model_id: str):
    model = models.get(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


@app.get("/models/{model_id}/slices", response_model=dict)
async def slice_model(
    model_id: str,
    layer: float = Query(...),
    x_min: float = -1.0,
    x_max: float = 1.0,
    y_min: float = -1.0,
    y_max: float = 1.0,
    nx: int = 50,
    ny: int = 50,
):
    """Proxy slice requests to the slicer service."""
    model = models.get(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    def _extract_lattice_data(obj: Any) -> tuple[Optional[list], Optional[list]]:
        """Recursively search for lattice fields in ``obj``.

        Returns the first encountered ``cell_vertices`` and ``edge_list`` values
        if present anywhere within the nested structure.  Either value may be
        ``None`` if not found.
        """

        cell_verts: Optional[list] = None
        edges: Optional[list] = None

        def walk(o: Any) -> None:
            nonlocal cell_verts, edges
            if isinstance(o, dict):
                if cell_verts is None and isinstance(o.get("cell_vertices"), list):
                    cell_verts = o["cell_vertices"]
                if edges is None and isinstance(o.get("edge_list"), list):
                    edges = o["edge_list"]
                for v in o.values():
                    if cell_verts is not None and edges is not None:
                        break
                    walk(v)
            elif isinstance(o, list):
                for item in o:
                    if cell_verts is not None and edges is not None:
                        break
                    walk(item)

        walk(obj)
        return cell_verts, edges

    cell_vertices, edge_list = _extract_lattice_data(model)
    logging.debug(
        "slice_model: cell_vertices[:3]=%s, edge_list[:3]=%s",
        cell_vertices[:3] if cell_vertices else None,
        edge_list[:3] if edge_list else None,
    )

    payload = {
        "model": model,
        "layer": layer,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "nx": nx,
        "ny": ny,
    }
    if cell_vertices is not None:
        payload["cell_vertices"] = cell_vertices
    if edge_list is not None:
        payload["edge_list"] = edge_list

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post("http://127.0.0.1:4000/slice", json=payload)
            resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=500, detail=f"Slicing service failure: {exc}")

    return resp.json()

class DesignRequest(BaseModel):
    prompt: str


class MeshRequest(BaseModel):
    seed_points: list[list[float]] = Field(default_factory=list)
    spacing: Optional[float] = None
    num_points: Optional[int] = None
    mode: Optional[Literal["uniform", "organic"]] = None


@app.post("/design/mesh", response_model=dict)
async def voronoi_mesh_endpoint(req: MeshRequest):
    """Generate a Voronoi mesh from seed points."""
    seed_cfg = resolve_seed_spec(
        {},
        None,
        None,
        seed_points=req.seed_points or None,
        num_points=req.num_points,
        spacing=req.spacing,
        mode=req.mode,
    )
    pts = [tuple(p) for p in seed_cfg.get("seed_points") or []]
    spacing = seed_cfg["spacing"]
    try:
        import core_engine as _core  # type: ignore

        verts, edge_list = _core.voronoi_mesh_py(pts)
        if not verts:
            raise ValueError("empty mesh")
        vertices = [list(v) for v in verts]
        edges = [list(e) for e in edge_list]
    except Exception:
        from design_api.services.voronoi_gen.voronoi_gen import (
            compute_voronoi_adjacency,
        )

        adjacency = compute_voronoi_adjacency(pts, spacing=spacing)
        seen = set()
        edges = []
        for i, j in adjacency:
            if i > j:
                i, j = j, i
            if (i, j) not in seen:
                seen.add((i, j))
                edges.append([i, j])
        vertices = [list(p) for p in pts]

    return {"vertices": vertices, "edges": edges}

@app.post("/design", response_model=dict)
async def design(req: DesignRequest):
    try:
        # 0. Invoke LLM
        raw_output = generate_design_spec(req.prompt)
        # 1. Clean and parse JSON
        cleaned = clean_llm_output(raw_output)
        logging.debug(f"Cleaned output repr: {cleaned}")
        try:
            spec_dict = json.loads(cleaned)
        except JSONDecodeError as je:
            raise HTTPException(status_code=502, detail=f"Failed to parse JSON from LLM. Cleaned: {cleaned}")
        # 2. Map and normalize into protobuf dict
        spec_dict = map_to_proto_dict(spec_dict)
        # 3. Validate against the Protobuf schema
        proto_spec = validate_proto(spec_dict)

        # 4. Return full spec dict
        return spec_dict

    except HTTPException:
        # pass through validation errors
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/design/review", response_model=dict)
async def review(req: dict, sid: Optional[str] = None):
    try:
        if sid is None or sid not in design_states:
            sid = str(uuid.uuid4())
            design_states[sid] = DesignState(draft_spec=[])
        # interpret and summarize the user's spec (or ask clarification)
        result = review_request(req)
        # If the adapter returned a clarification question, surface it
        if isinstance(result, dict) and "question" in result:
            question = result["question"]
            log_turn(sid, "clarify", req.get("raw", ""), [], question=question)
            return {"sid": sid, "question": question}
        # Otherwise we have a spec and summary tuple
        spec, summary = result

        # Compute adjacency and edges for any infill
        for idx, node in enumerate(spec):
            inf = node.get("modifiers", {}).get("infill", {})
            pts = inf.get("seed_points")
            num_pts = inf.get("num_points")
            bbox_min = inf.get("bbox_min")
            bbox_max = inf.get("bbox_max")
            logging.debug(
                f"pts {pts} num_pts {num_pts} bbox_min {bbox_min} bbox_max {bbox_max}"
            )

            stored = design_states[sid].seed_cache.get(idx)

            if (pts or num_pts or stored) and bbox_min and bbox_max:
                pattern = inf.get("pattern")
                logging.debug(f"PATTERN {pattern}")
                seed_pts = None
                if stored and (num_pts is None or num_pts == len(stored)):
                    seed_pts = stored
                elif pts is not None:
                    seed_pts = pts
                seed_cfg = resolve_seed_spec(
                    node.get("primitive", {}),
                    bbox_min,
                    bbox_max,
                    seed_points=seed_pts,
                    num_points=None if seed_pts is not None else num_pts,
                    spacing=inf.get("spacing") or inf.get("min_dist"),
                    mode=inf.get("mode") or req.get("mode"),
                    uniform=inf.get("uniform") or req.get("uniform"),
                )
                spec_kwargs = {
                    **inf,
                    **seed_cfg,
                    "primitive": node.get("primitive", {}),
                    "imds_mesh": inf.get("imds_mesh") or req.get("imds_mesh"),
                    "plane_normal": inf.get("plane_normal") or req.get("plane_normal"),
                    "max_distance": inf.get("max_distance") or req.get("max_distance"),
                    "use_voronoi_edges": inf.get("use_voronoi_edges", False),
                }
                if pattern == "hex":
                    res = generate_hex_lattice(spec_kwargs)
                else:
                    if seed_cfg["mode"] == "uniform":
                        res = generate_hex_lattice(spec_kwargs)
                    else:
                        res = generate_voronoi({**inf, **seed_cfg})
                inf.update(res)
                inf["pattern"] = pattern
                if seed_cfg.get("seed_points") is not None:
                    design_states[sid].seed_cache[idx] = seed_cfg["seed_points"]
                logging.debug(
                    f"[DEBUG review] produced {len(res.get('edge_list', res.get('edges', [])))} edges"
                )


        # sanitize spec to convert numpy arrays into lists for JSON serialization
        def _sanitize(o):
            import numpy as np

            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.floating, float)):
                return float(o)
            if isinstance(o, (np.integer, int)):
                return int(o)
            if isinstance(o, tuple):
                return [_sanitize(i) for i in o]
            if isinstance(o, list):
                return [_sanitize(i) for i in o]
            if isinstance(o, dict):
                return {k: _sanitize(v) for k, v in o.items()}
            return o

        spec = _sanitize(spec)
        design_states[sid].draft_spec = spec

        log_turn(sid, "review", req.get("raw", ""), spec, summary=summary)
        return {"sid": sid, "spec": spec, "summary": summary}
    except HTTPException:
        # propagate intentional HTTP errors without remapping to 500s
        raise
    except Exception as e:
        logging.exception("Error in review endpoint")
        raise HTTPException(status_code=500, detail=str(e))



# New update endpoint: expects session and spec in body, returns summary
class UpdateRequest(BaseModel):
    sid: str
    raw: str
    spec: list
    imds_mesh: Optional[Any] = None
    plane_normal: Optional[Any] = None
    max_distance: Optional[float] = None

@app.post("/design/update", response_model=dict)
async def update(req: UpdateRequest):
    """
    Update the design spec for a session, returning a summary.
    """
    logging.debug(f"Received /design/update request for session {req.sid}: spec={req.spec}, raw={req.raw}")
    sid = req.sid
    if sid not in design_states:
        raise HTTPException(status_code=400, detail=f"Unknown session id {sid}")
    # Call the adapter to update the spec
    result = update_request(req.sid, req.spec, req.raw)
    # If adapter returned a clarification question, forward it
    if isinstance(result, dict) and "question" in result:
        question = result["question"]
        log_turn(req.sid, "clarify", req.raw, design_states[req.sid].draft_spec, question=question)
        return {"sid": req.sid, "question": question}
    # Unpack updated spec and summary
    new_spec, new_summary = result
    # Normalize any top-level infill into node['modifiers']['infill']
    for node in new_spec:
        if "infill" in node:
            node.setdefault("modifiers", {})["infill"] = node.pop("infill")

    # Compute adjacency and edges for any infill
    for idx, node in enumerate(new_spec):
        inf = node.get("modifiers", {}).get("infill", {})
        pts = inf.get("seed_points")
        num_pts = inf.get("num_points")
        bbox_min = inf.get("bbox_min")
        bbox_max = inf.get("bbox_max")
        stored = design_states[sid].seed_cache.get(idx)

        if (pts or num_pts or stored) and bbox_min and bbox_max:
            pattern = inf.get("pattern")
            seed_pts = None
            if stored and (num_pts is None or num_pts == len(stored)):
                seed_pts = stored
            elif pts is not None:
                seed_pts = pts
            seed_cfg = resolve_seed_spec(
                node.get("primitive", {}),
                bbox_min,
                bbox_max,
                seed_points=seed_pts,
                num_points=None if seed_pts is not None else num_pts,
                spacing=inf.get("spacing") or inf.get("min_dist"),
                mode=inf.get("mode"),
                uniform=inf.get("uniform"),
            )
            spec_kwargs = {
                **inf,
                **seed_cfg,
                "primitive": node.get("primitive", {}),
                "imds_mesh": inf.get("imds_mesh") or req.imds_mesh,
                "plane_normal": inf.get("plane_normal") or req.plane_normal,
                "max_distance": inf.get("max_distance") or req.max_distance,
                "use_voronoi_edges": inf.get("use_voronoi_edges", False),
            }
            if pattern == "hex":
                res = generate_hex_lattice(spec_kwargs)
            else:
                if seed_cfg["mode"] == "uniform":
                    res = generate_hex_lattice(spec_kwargs)
                else:
                    res = generate_voronoi({**inf, **seed_cfg})
            inf.update(res)
            inf["pattern"] = pattern
            if seed_cfg.get("seed_points") is not None:
                design_states[sid].seed_cache[idx] = seed_cfg["seed_points"]
            logging.debug(
                f"[DEBUG update] produced {len(res.get('edge_list', res.get('edges', [])))} edges",
            )


    # sanitize new_spec to convert numpy arrays into lists for JSON serialization
    def _sanitize(o):
        import numpy as np

        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, float)):
            return float(o)
        if isinstance(o, (np.integer, int)):
            return int(o)
        if isinstance(o, tuple):
            return [_sanitize(i) for i in o]
        if isinstance(o, list):
            return [_sanitize(i) for i in o]
        if isinstance(o, dict):
            return {k: _sanitize(v) for k, v in o.items()}
        return o

    new_spec = _sanitize(new_spec)
    design_states[req.sid].draft_spec = new_spec
    log_turn(req.sid, "update", req.raw, new_spec)
    return {"sid": req.sid, "spec": new_spec, "summary": new_summary}


# New endpoint: submit final model
@app.post("/design/submit", response_model=dict)
async def submit(req: dict, sid: str):
    """
    Finalize and lock in the current draft spec for the given session.
    """
    logging.debug(f"Received /design/submit request for session {sid}: body={req}")
    if sid not in design_states:
        raise HTTPException(status_code=400, detail=f"Unknown session id {sid}")
    # Retrieve the draft spec
    spec_dict = design_states[sid].draft_spec
    logging.debug(f"Draft spec for submission: {spec_dict}")
    # Recompute infill using stored seeds so final output matches previews
    nodes = spec_dict if isinstance(spec_dict, list) else spec_dict.get("primitives", [])
    for idx, node in enumerate(nodes):
        inf = node.get("modifiers", {}).get("infill", {})
        pts = inf.get("seed_points")
        num_pts = inf.get("num_points")
        bbox_min = inf.get("bbox_min")
        bbox_max = inf.get("bbox_max")
        stored = design_states[sid].seed_cache.get(idx)
        if (pts or num_pts or stored) and bbox_min and bbox_max:
            pattern = inf.get("pattern")
            seed_pts = None
            if stored and (num_pts is None or num_pts == len(stored)):
                seed_pts = stored
            elif pts is not None:
                seed_pts = pts
            seed_cfg = resolve_seed_spec(
                node.get("primitive", {}),
                bbox_min,
                bbox_max,
                seed_points=seed_pts,
                num_points=None if seed_pts is not None else num_pts,
                spacing=inf.get("spacing") or inf.get("min_dist"),
                mode=inf.get("mode"),
                uniform=inf.get("uniform"),
            )
            spec_kwargs = {
                **inf,
                **seed_cfg,
                "primitive": node.get("primitive", {}),
                "imds_mesh": inf.get("imds_mesh"),
                "plane_normal": inf.get("plane_normal"),
                "max_distance": inf.get("max_distance"),
                "use_voronoi_edges": inf.get("use_voronoi_edges", False),
            }
            if pattern == "hex":
                res = generate_hex_lattice(spec_kwargs)
            else:
                if seed_cfg["mode"] == "uniform":
                    res = generate_hex_lattice(spec_kwargs)
                else:
                    res = generate_voronoi({**inf, **seed_cfg})
            inf.update(res)
            inf["pattern"] = pattern
            if seed_cfg.get("seed_points") is not None:
                design_states[sid].seed_cache[idx] = seed_cfg["seed_points"]
    entries = nodes

    def is_proto_node(e):
        if not isinstance(e, dict) or 'modifiers' in e:
            return False
        if 'booleanOp' in e or 'children' in e:
            return True
        prim = e.get('primitive')
        if isinstance(prim, dict):
            return any(k in prim for k in ('lattice', 'shell', 'shellFill'))
        return False

    children = [e if is_proto_node(e) else map_to_proto_dict(e) for e in entries]
    # Build the root model dict
    proto_dict = {
        'id': str(uuid.uuid4()),
        'root': {
            'children': children
        }
    }
    # Validate against protobuf
    validated = validate_proto(proto_dict)
    # Store locked model
    design_states[sid].locked_model = proto_dict
    return {"sid": sid, "locked_model": proto_dict}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

import logging
from pydantic import Field
import time
import numpy as np
import os
from pathlib import Path
import requests

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)

import json
import copy
import traceback
import uuid
from json.decoder import JSONDecodeError
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from dataclasses import dataclass
from design_api.services.json_cleaner import clean_llm_output
from design_api.services.llm_service import generate_design_spec
from design_api.services.mapping import map_primitive as map_to_proto_dict
from design_api.services.validator import validate_model_spec as validate_proto
from ai_adapter.csg_adapter import review_request, generate_summary, update_request
from design_api.services.voronoi_gen.organic.construct import construct_voronoi_cells
from design_api.services.voronoi_gen.uniform.construct import compute_uniform_cells

@dataclass
class DesignState:
    draft_spec: list
    locked_model: Optional[dict] = None

# session store: session_id -> DesignState
design_states: dict[str, DesignState] = {}

# persistent model storage directory
MODEL_DIR = Path(os.environ.get("MODEL_STORE", "models"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# slicer service URL
SLICER_URL = os.environ.get("SLICER_URL", "http://127.0.0.1:4000/slice")

def log_turn(session_id: str, turn_type: str, raw: str, spec: list, summary: Optional[str] = None, question: Optional[str] = None):
    entry = {
        "session": session_id,
        "timestamp": time.time(),
        "type": turn_type,
        "raw": raw,
        "spec": spec,
    }
    # remove seed_points from infill modifiers to keep logs small
    scrubbed_spec = []
    for node in entry["spec"]:
        node_copy = copy.deepcopy(node)
        mods = node_copy.get("modifiers")
        if mods and isinstance(mods.get("infill"), dict):
            mods["infill"].pop("seed_points", None)
        scrubbed_spec.append(node_copy)
    entry["spec"] = scrubbed_spec
    if summary is not None:
        entry["summary"] = summary
    if question is not None:
        entry["question"] = question
    with open("conversation_log.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")

app = FastAPI(title="Implicitus Design API", debug=True)

# Allow local front-end to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DesignRequest(BaseModel):
    prompt: str

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
        # Optionally generate Voronoi cell previews for any infill modifiers
        for node in spec:
            inf = node.get("modifiers", {}).get("infill")
            if isinstance(inf, dict) and inf.get("pattern") == "voronoi":
                seed_points = inf.get("seed_points")
                bbox_min = inf.get("bbox_min")
                bbox_max = inf.get("bbox_max")
                if seed_points and bbox_min and bbox_max:
                    pts_arr = np.asarray(seed_points)
                    try:
                        uniform_flag = inf.get("uniform") or (
                            isinstance(inf.get("mode"), str) and inf.get("mode").lower() == "uniform"
                        )
                        if uniform_flag:
                            inf["uniform"] = True  # ensure downstream components see flag
                            plane = np.array([0.0, 0.0, 1.0])
                            extent = np.linalg.norm(np.array(bbox_max) - np.array(bbox_min)) * 0.5
                            cells = compute_uniform_cells(
                                pts_arr, None, plane, max_distance=extent
                            )
                            inf["cells"] = [poly.tolist() for poly in cells.values()]
                        else:
                            inf["cells"] = construct_voronoi_cells(
                                pts_arr, tuple(bbox_min), tuple(bbox_max)
                            )
                    except Exception:
                        logging.exception("Failed to generate Voronoi cells")
        design_states[sid].draft_spec = spec
        log_turn(sid, "review", req.get("raw", ""), spec, summary=summary)
        return {"sid": sid, "spec": spec, "summary": summary}
    except Exception as e:
        logging.exception("Error in review endpoint")
        raise HTTPException(status_code=500, detail=str(e))



# New update endpoint: expects session and spec in body, returns summary
class UpdateRequest(BaseModel):
    sid: str
    raw: str
    spec: list

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
    # Normalize entries: if they look like proto Nodes, use as-is; else map primitives
    entries = spec_dict if isinstance(spec_dict, list) else spec_dict.get('primitives', [])
    def is_proto_node(e):
        return isinstance(e, dict) and any(k in e for k in ['primitive','booleanOp','infill','shell','shellFill','children'])
    if all(is_proto_node(e) for e in entries):
        children = entries
    else:
        children = [map_to_proto_dict(e) for e in entries]
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


# Model storage and slicing endpoints
@app.post("/models", response_model=dict)
async def store_model(model: dict):
    """Persist a model on disk."""
    model_id = model.get("id")
    if not model_id:
        raise HTTPException(status_code=400, detail="Missing model.id")
    path = MODEL_DIR / f"{model_id}.json"
    with open(path, "w") as f:
        json.dump(model, f)
    return {"id": model_id}


@app.get("/models/{model_id}", response_model=dict)
async def fetch_model(model_id: str):
    """Retrieve a stored model by ID."""
    path = MODEL_DIR / f"{model_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    with open(path) as f:
        return json.load(f)


@app.get("/models/{model_id}/slices", response_model=dict)
async def slice_model(
    model_id: str,
    layer: float,
    x_min: float = -1.0,
    x_max: float = 1.0,
    y_min: float = -1.0,
    y_max: float = 1.0,
    nx: int = 50,
    ny: int = 50,
):
    """Slice a stored model at a given layer by proxying to the slicer service."""
    path = MODEL_DIR / f"{model_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    with open(path) as f:
        model = json.load(f)
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
    try:
        resp = requests.post(SLICER_URL, json=payload)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logging.exception("Slicer service error")
        raise HTTPException(status_code=500, detail=f"Slicing service failure: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
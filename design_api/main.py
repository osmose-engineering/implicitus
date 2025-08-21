import logging
from pydantic import Field
import time

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)

import json
import copy
import traceback
import uuid
import numpy as np
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
from design_api.services.voronoi_gen.voronoi_gen import compute_voronoi_adjacency

@dataclass
class DesignState:
    draft_spec: list
    locked_model: Optional[dict] = None

# session store: session_id -> DesignState
design_states: dict[str, DesignState] = {}

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
        f.write(json.dumps(entry, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")

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

        # Compute adjacency and edges for any infill
        for node in spec:
            inf = node.get("modifiers", {}).get("infill", {})
            pts = inf.get("seed_points")
            bbox_min = inf.get("bbox_min") or inf.get("bboxMin")
            bbox_max = inf.get("bbox_max") or inf.get("bboxMax")
            logging.debug(f"pts {pts} bbox_min {bbox_min} bbox_max {bbox_max}")

            if pts and bbox_min and bbox_max:
                pattern = inf.get("pattern")
                logging.debug(f"PATTERN {pattern}")
                if pattern == "voronoi":
                    # prefer explicit spacing, otherwise fall back to min_dist or a sane default
                    spacing = (
                        inf.get("spacing")
                        or inf.get("min_dist")
                        or 2.0
                    )
                    # compute adjacency of seed points using spatial pruning
                    adjacency = compute_voronoi_adjacency(pts, spacing=spacing)
                    # convert adjacency map -> unique edge list (i<j)
                    edge_list = []
                    for i, nbrs in adjacency.items():
                        for j in nbrs:
                            if j > i:
                                edge_list.append([i, j])
                    # expose both raw edges and combined cell data for frontend
                    inf["edges"] = edge_list
                    inf["cells"] = {"points": pts, "edges": edge_list}
                    logging.debug(
                        f"[DEBUG review] spacing={spacing} produced {len(edge_list)} edges; sample: {edge_list[:10]}"
                    )

        # sanitize spec to convert numpy arrays into lists for JSON serialization
        def _sanitize(o):
            import numpy as np
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, list):
                return [_sanitize(i) for i in o]
            if isinstance(o, dict):
                return {k: _sanitize(v) for k, v in o.items()}
            return o

        spec = _sanitize(spec)
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

    # Compute adjacency and edges for any infill
    for node in new_spec:
        inf = node.get("modifiers", {}).get("infill", {})
        pts = inf.get("seed_points")
        bbox_min = inf.get("bbox_min") or inf.get("bboxMin")
        bbox_max = inf.get("bbox_max") or inf.get("bboxMax")
        if pts and bbox_min and bbox_max:
            pattern = inf.get("pattern")
            if pattern == "voronoi":
                spacing = (
                    inf.get("spacing")
                    or inf.get("min_dist")
                    or 2.0
                )
                adjacency = compute_voronoi_adjacency(pts, spacing=spacing)
                edge_list = []
                for i, nbrs in adjacency.items():
                    for j in nbrs:
                        if j > i:
                            edge_list.append([i, j])
                inf["edges"] = edge_list
                inf["cells"] = {"points": pts, "edges": edge_list}
            logging.debug(
                f"[DEBUG update] spacing={spacing} produced {len(inf.get('edges', []))} edges; sample: {inf.get('edges', [])[:10]}"
            )

    # sanitize new_spec to convert numpy arrays into lists for JSON serialization
    def _sanitize(o):
        import numpy as np
        if isinstance(o, np.ndarray):
            return o.tolist()
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
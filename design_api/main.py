import logging
from pydantic import Field

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)

import json
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

@dataclass
class DesignState:
    draft_spec: list
    locked_model: Optional[dict] = None

# session store: session_id -> DesignState
design_states: dict[str, DesignState] = {}

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
        # interpret and summarize the user's spec
        spec, summary = review_request(req)
        design_states[sid].draft_spec = spec
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
    # Apply the user's raw update instruction to the existing spec
    new_spec, new_summary = update_request(req.sid, req.spec, req.raw)
    design_states[req.sid].draft_spec = new_spec
    return {"sid": req.sid, "spec": new_spec, "summary": new_summary}


# New endpoint: /design/submit
@app.post("/design/submit", response_model=dict)
async def submit(req: dict, sid: str):
    logging.debug(f"Received /design/submit request for session {sid}: raw request body={req}")
    try:
        if sid not in design_states:
            raise HTTPException(status_code=400, detail=f"Unknown session id {sid}")
        # 0. Accept spec directly from stored draft_spec
        spec_dict = design_states[sid].draft_spec
        logging.debug(f"Received spec for submission from sid {sid}: {spec_dict}")
        # Determine children nodes: accept either raw primitive specs or already-mapped Node dicts
        entries = spec_dict if isinstance(spec_dict, list) else spec_dict.get('primitives', [])
        # If entries look like protobuf Node dicts (contain 'primitive', 'booleanOp', etc.), use directly;
        # otherwise map raw user specs with shape keys.
        def is_proto_node(e):
            return isinstance(e, dict) and any(k in e for k in ['primitive','booleanOp','infill','shell','shellFill','children'])
        if all(is_proto_node(e) for e in entries):
            children = entries
        else:
            # raw user specs with 'shape' -> map through map_to_proto_dict
            children = [map_to_proto_dict(e) for e in entries]

        # Assemble Model JSON
        proto_dict = {
            'id': str(uuid.uuid4()),
            'root': {
                'children': children
            }
        }
        # 2. Validate against the Protobuf schema
        proto_spec = validate_proto(proto_dict)
        # store the final locked model
        design_states[sid].locked_model = proto_dict
        # 3. Return validated spec dict
        return {"sid": sid, "locked_model": proto_dict}
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Error in submit endpoint")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
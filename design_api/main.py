# design_api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ai_adapter.inference_pipeline import generate  # your LLM pipeline
from ai_adapter.schema.implicitus_pb2 import Model as ImplicitSpec
from google.protobuf.json_format import ParseDict, ParseError
import json
import traceback
from json.decoder import JSONDecodeError
import uuid
import re

app = FastAPI(title="Implicitus Design API", debug=True)

class DesignRequest(BaseModel):
    prompt: str

class DesignModel(BaseModel):
    id: str
    shape: str
    size_mm: float
    # ... extend with other fields as needed

@app.post("/design", response_model=DesignModel)
async def design(req: DesignRequest):
    try:
        # 0. Invoke LLM to get JSON output
        json_output = generate(req.prompt)
        # (Optional) debug print of raw LLM output
        # print(f"LLM raw output: {json_output}")
        # 1. Clean up LLM output: strip whitespace and headers
        raw_lines = json_output.strip().splitlines()
        # drop a leading "Something:" header
        if raw_lines and raw_lines[0].strip().lower().endswith(':'):
            raw_lines = raw_lines[1:]
        # drop any code fence lines (``` or ```lang)
        raw_lines = [line for line in raw_lines if not line.strip().startswith('```')]
        # drop a standalone "json" or "json:" line
        if raw_lines and raw_lines[0].strip().lower().rstrip(':') == 'json':
            raw_lines = raw_lines[1:]
        # rebuild cleaned JSON string
        cleaned = '\n'.join(raw_lines).strip()

        # 2. Parse into a dict
        try:
            spec_dict = json.loads(cleaned)
        except JSONDecodeError as je:
            print(f"JSONDecodeError: raw json_output: {json_output}")
            print(f"JSONDecodeError: cleaned string: {cleaned}")
            raise HTTPException(status_code=502, detail=f"Failed to parse JSON from LLM. Cleaned: {cleaned}. Raw output: {json_output}")

        # 2.5. Ensure an 'id' field is present
        if 'id' not in spec_dict or not spec_dict['id']:
            spec_dict['id'] = str(uuid.uuid4())

        # Normalize keys to match Protobuf JSON field names
        if 'size_mm' in spec_dict:
            spec_dict['sizeMm'] = spec_dict.pop('size_mm')

        # 2.6. Map simple shape/size to Protobuf CSG root node
        shape = spec_dict.pop('shape', None)
        size_mm = spec_dict.pop('sizeMm', None)
        if shape:
            if shape.lower() == 'sphere' and size_mm is not None:
                # Protobuf expects Node.primitive.sphere.radius
                spec_dict['root'] = {
                    'primitive': {
                        'sphere': {'radius': size_mm / 2}
                    }
                }
            else:
                # TODO: extend for other primitives (cube, cylinder, etc.)
                spec_dict['root'] = {
                    'primitive': {}
                }

        # 3. Validate against the Protobuf schema
        proto_spec = ImplicitSpec()
        try:
            ParseDict(spec_dict, proto_spec)
        except ParseError as pe:
            detail_msg = f"Schema validation error: {pe}. Spec dict was: {spec_dict}. Raw output: {json_output}. Cleaned string: {cleaned}"
            print(f"ParseError: {detail_msg}")
            raise HTTPException(status_code=400, detail=detail_msg)

        # 4. Build and return the API model
        # Derive shape and size_mm for response
        response_shape = shape or ""
        response_size_mm = size_mm or 0
        return {
            "id": proto_spec.id,
            "shape": response_shape,
            "size_mm": response_size_mm,
        }

    except HTTPException:
        # pass through validation errors
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
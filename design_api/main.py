# design_api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ai_adapter.transformers_temp import chat  # your pipeline builder
from ai_adapter.csg_adapter import parse_and_build_model

app = FastAPI(title="Implicitus Design API")

class DesignRequest(BaseModel):
    prompt: str

class DesignModel(BaseModel):
    id: str
    # ... other fields your implicitus.Model needs

@app.post("/design", response_model=DesignModel)
async def design(req: DesignRequest):
    # 1. Ask the LLM to generate a protobuf CSG spec
    try:
        answer = chat([
            {"role": "user", "content": req.prompt}
        ])
    except Exception as e:
        raise HTTPException(500, f"LLM error: {e}")

    # 2. Parse the LLM’s JSON or protobuf into your internal Model class
    #    (you’ll call into ai_adapter → CSG builder here)
    model = parse_and_build_model(answer)  

    # 3. Persist or assign an ID (if you want durability here, you could forward
    #    to your canvas_api /models or slice immediately)
    return model
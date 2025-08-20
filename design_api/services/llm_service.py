import logging
# Imports for LLM call and JSON parsing
from ai_adapter import inference_pipeline
import json
from design_api.services.json_cleaner import clean_llm_output

def generate_design_spec(prompt: str) -> dict:
    """
    Call the LLM with the given prompt and return the parsed JSON spec.
    If the LLM call or parsing fails, return a minimal placeholder spec
    so downstream code can proceed.
    """
    try:
        raw_output = inference_pipeline.generate(prompt)
    except Exception:
        logging.exception("LLM generation failed")
        return {"shape": "box", "size_mm": 1}

    raw_str = raw_output if isinstance(raw_output, str) else repr(raw_output)
    logging.debug(f"LLM raw_output repr: {raw_str}")
    cleaned_output = clean_llm_output(raw_str)
    if not cleaned_output or not cleaned_output.strip():
        logging.warning("LLM returned empty output")
        return {"shape": "box", "size_mm": 1}

    try:
        spec = json.loads(cleaned_output)
    except json.JSONDecodeError:
        logging.exception("Failed to parse LLM output as JSON")
        return {"shape": "box", "size_mm": 1}

    if not isinstance(spec, dict) or "shape" not in spec or "size_mm" not in spec:
        logging.warning("LLM output missing required fields")
        return {"shape": "box", "size_mm": 1}

    return spec

__all__ = ["generate_design_spec"]
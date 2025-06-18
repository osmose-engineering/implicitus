import logging
# Imports for LLM call and JSON parsing
from ai_adapter.inference_pipeline import generate
import json
from design_api.services.json_cleaner import clean_llm_output

def generate_design_spec(prompt: str) -> dict:
    """
    Call the LLM with the given prompt and return the parsed JSON spec.
    """
    # Call the LLM and get raw output
    raw_output = generate(prompt)
    raw_str = raw_output if isinstance(raw_output, str) else repr(raw_output)
    logging.debug(f"LLM raw_output repr: {raw_str}")
    # Clean any extraneous characters (code fences, labels, etc.)
    cleaned_output = clean_llm_output(raw_str)
    if not cleaned_output or not cleaned_output.strip():
        raise ValueError(f"LLM returned empty output: {raw_str!r}")
    try:
        spec = json.loads(cleaned_output)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse LLM output as JSON:\nRaw output: {raw_str!r}\nCleaned output: {cleaned_output!r}\nError: {e}"
        ) from e
    return spec

__all__ = ["generate_design_spec"]
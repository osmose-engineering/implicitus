import re
from typing import Any
import json
import ast

def clean_llm_output(raw: Any) -> str:
    """
    Clean raw LLM output by stripping markdown fences, labels, and extraneous characters,
    returning only the raw JSON string.
    """
    text = raw
    if isinstance(text, (list, dict)):
        text = str(text)
    # Strip markdown code fences
    text = re.sub(r'```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```', '', text)
    # Remove "JSON:" or "Response:" labels (case-insensitive)
    text = re.sub(r'(?i)(?:json|response):\s*', '', text)
    text = text.strip()
    # Extract the JSON object
    match = re.search(r'\{.*\}', text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Unable to find JSON object in LLM output: {repr(text)}")
    json_text = match.group(0)
    # Parse into a Python object
    try:
        obj = json.loads(json_text)
    except json.JSONDecodeError:
        try:
            obj = ast.literal_eval(json_text)
        except Exception as e:
            raise ValueError(f"Failed to parse JSON or literal dict: {e}. Raw: {repr(json_text)}")
    # Return a clean JSON string
    return json.dumps(obj)
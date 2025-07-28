import re
import json

def generate(raw_request: str) -> str:
    # Parse simple "<size>mm <shape>" requests into a JSON spec
    match = re.search(r"(\d+(?:\.\d+)?)\s*mm\s*(\w+)", raw_request)
    if match:
        size_val = float(match.group(1)) if "." in match.group(1) else int(match.group(1))
        shape_val = match.group(2)
        spec_dict = {"shape": shape_val, "size_mm": size_val}
        user_content = json.dumps(spec_dict)
    else:
        # Fallback: send the raw request directly
        user_content = raw_request

    # Build messages for the model
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]
    # Record last prompt for inspection
    inference_pipeline.last_prompt = "\n".join([msg["content"] for msg in messages])

    # Invoke the text-generation pipeline
    generator = pipeline("text2text-generation")
    outputs = generator(messages)
    # Return the generated text (or echoed content under test)
    return outputs[0].get("generated_text", "")
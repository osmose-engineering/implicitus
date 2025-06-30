from transformers import AutoTokenizer

# 1. Copy your SYSTEM_PROMPT exactly as defined in inference_pipeline.py
SYSTEM_PROMPT = (
    "You are an implicit-design API. When given a request like 'Make me a 10 mm sphere', "
    "respond with exactly one JSON object (no surrounding text) with keys: "
    "'shape' (e.g., 'sphere'), 'size_mm' (a number), and any other minimal parameters. "
    "Do not include any explanations or steps. "
    "Do not wrap your output in markdown code fences, include JSON labels like 'JSON:', "
    "or add any extra characters—emit only raw JSON."
)

# 2. Load the same tokenizer your pipeline uses
tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3", use_fast=True
)

def count_tokens(prompt: str):
    sys_ids    = tokenizer(SYSTEM_PROMPT).input_ids
    prompt_ids = tokenizer(prompt).input_ids
    return {
        "system_tokens": len(sys_ids),
        "prompt_tokens": len(prompt_ids),
        "total_tokens":  len(sys_ids) + len(prompt_ids)
    }

# Example
result = count_tokens("Can you make me a 10mm sphere?")
print(result)
from transformers import pipeline, AutoTokenizer

# system-level guardrail: only emit raw JSON with shape parameters
SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are an implicit-design API. "
        "When given a request like 'Make me a 10 mm sphere', "
        "respond with exactly one JSON object (no surrounding text) with keys: "
        "'shape' (e.g., 'sphere'), 'size_mm' (a number), and any other minimal parameters. "
        "Do not include any explanations or steps. "
        "Do not wrap your output in markdown code fences, include JSON labels like 'JSON:', "
        "or add any extra characters—emit only raw JSON."
    )
}

_chat = None

def _init_chat():
    global _chat
    if _chat is None:
        tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.3",
            use_fast=True
        )
        _chat = pipeline(
            "text-generation",
            model="mistralai/Mistral-7B-Instruct-v0.3",
            tokenizer=tokenizer,
            device_map="auto",
            torch_dtype="auto",
            return_full_text=False,
        )
    return _chat

def generate(prompt: str) -> str:
    """
    Generate a structured JSON response from the model for the given prompt.
    The pipeline expects a list of messages as input.
    Returns the raw JSON string.
    """
    pipe = _init_chat()
    messages = [
        SYSTEM_PROMPT,
        {"role": "user", "content": prompt}
    ]

    # try with max_new_tokens, but fall back if not supported
    try:
        resp = pipe(messages, max_new_tokens=128)
    except TypeError:
        resp = pipe(messages)

    # normalize pipeline output
    if isinstance(resp, list) and resp and isinstance(resp[0], dict) and "generated_text" in resp[0]:
        raw = resp[0]["generated_text"]
    elif isinstance(resp, str):
        raw = resp
    else:
        raise RuntimeError(f"Unexpected pipeline return type: {resp!r}")

    return raw.strip()
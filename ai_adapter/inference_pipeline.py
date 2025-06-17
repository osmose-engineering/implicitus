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

# Ensure the Rust-backed tokenizers library is installed (pip install tokenizers)
tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    use_fast=True
)

chat = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.3",
    tokenizer=tokenizer,
    device_map="auto",
    torch_dtype="auto",
    return_full_text=False,
)

def generate(prompt: str) -> str:
    """
    Generate a structured JSON response from the model for the given prompt.
    Returns the raw JSON string.
    """
    prompt_text = SYSTEM_PROMPT["content"] + "\nUser: " + prompt
    print(f"DEBUG prompt_text: {prompt_text}")
    # call text-generation pipeline with the combined prompt text
    response = chat(prompt_text, max_new_tokens=128)
    # extract generated text from the first result entry
    json_str = response[0]["generated_text"].strip()
    return json_str
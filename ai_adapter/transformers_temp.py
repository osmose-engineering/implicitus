from transformers import pipeline, AutoTokenizer

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
    torch_dtype="auto"
)
result = chat([{"role":"user","content":"Explain quantum entanglement in two sentences."}])
# Extract and print only the assistant’s reply
assistant_reply = result[0]['generated_text'][-1]['content']
print(assistant_reply)
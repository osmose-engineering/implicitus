# ai_adapter/adapter.py
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from ai_adapter.schema import implicitus_pb2
from typing import Optional
from google.protobuf.json_format import Parse

MODEL_NAME = "mistralai/Mistral-7B-Instruct"

_tokenizer: Optional[AutoTokenizer] = None
_model: Optional[AutoModelForCausalLM] = None

def _load_llm():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
    return _tokenizer, _model

def generate_model(prompt: str, use_llm: bool = True) -> implicitus_pb2.Model:
    """
    Uses Mistral 7B-Instruct to generate a JSON-formatted implicit model
    matching the implicitus.Model protobuf schema.

    Args:
        prompt: The input prompt string to convert.
        use_llm: If False, returns a stub model instead of running LLM inference.

    Returns:
        An implicitus_pb2.Model instance representing the generated model.
    """
    if not use_llm:
        # Stub mode: return a unit sphere model with the prompt as ID
        model_pb = implicitus_pb2.Model()
        model_pb.id = prompt
        root = implicitus_pb2.Node()
        primitive = implicitus_pb2.Primitive()
        sphere = implicitus_pb2.Sphere(radius=1.0)
        primitive.sphere.CopyFrom(sphere)
        root.primitive.CopyFrom(primitive)
        model_pb.root.CopyFrom(root)
        return model_pb

    tokenizer, model = _load_llm()
    # System and user messages to enforce JSON output
    system_msg = (
        "You are Implicitus, a JSON-generating model. "
        "Respond with valid JSON matching the implicitus.Model schema."
    )
    user_msg = f"Convert this request into Implicitus JSON:\n\n{prompt}"

    # Tokenize inputs
    inputs = tokenizer([system_msg, user_msg], return_tensors="pt", padding=True)
    # Move tensors to the model device if available
    try:
        inputs = inputs.to(model.device)
    except Exception:
        pass

    # Generation configuration
    gen_config = GenerationConfig(
        max_new_tokens=512,
        temperature=0.2,
        top_p=0.9,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id
    )

    # Generate and decode
    outputs = model.generate(**inputs, generation_config=gen_config)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Parse JSON
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM output is not valid JSON: {e}\n{text}")

    # Convert JSON dict into protobuf Model
    model_pb = implicitus_pb2.Model()
    # Parse JSON string into the protobuf model
    Parse(json.dumps(data), model_pb)
    return model_pb
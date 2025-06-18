# ai_adapter/csg_adapter.py
import json
import base64
from google.protobuf.json_format import ParseDict
from ai_adapter.schema.implicitus_pb2 import Model
import uuid
from google.protobuf import json_format

# mapping from LLM 'shape' string to (proto field name, field-builder function)
_SHAPE_FIELD_MAP = {
    'sphere': ('sphere', lambda size: {'radius': size / 2}),
    'cube':   ('cube',   lambda size: {'size': size}),
    'box':    ('box',    lambda size: {'size': {'x': size, 'y': size, 'z': size}}),
    # extend as more primitives are supported...
}

def parse_and_build_model(llm_output) -> Model:
    # normalize raw JSON string or dict
    if isinstance(llm_output, list) and llm_output:
        llm_output = llm_output[0]
    if isinstance(llm_output, str):
        llm_output = json.loads(llm_output)
    elif not isinstance(llm_output, dict):
        raise TypeError(f"Unsupported llm_output type: {type(llm_output)}")

    # extract and normalize shape
    raw_shape = llm_output.get('shape', '').strip().lower()
    if raw_shape not in _SHAPE_FIELD_MAP:
        raise ValueError(f"Unknown shape '{raw_shape}' in LLM output")
    field_name, build_fields = _SHAPE_FIELD_MAP[raw_shape]

    # extract size (accepting snake_case or camelCase)
    size = llm_output.get('size_mm') or llm_output.get('sizeMm')
    if size is None:
        raise ValueError("No size_mm or sizeMm field found in LLM output")

    # assemble proto-compatible dict
    model_dict = {
        'id': str(uuid.uuid4()),
        'root': {
            'primitive': {
                field_name: build_fields(float(size))
            }
        }
    }

    # parse and return a validated Model
    proto = Model()
    json_format.ParseDict(model_dict, proto)
    return proto
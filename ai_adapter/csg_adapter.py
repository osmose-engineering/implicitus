def _build_modifier_dict(raw_spec: dict) -> dict:
    """
    Normalize any infill specification, whether it comes as a nested dict
    under 'infill' or as top‐level keys like 'infill_pattern' and 'infill_thickness_mm'.
    Returns either {'pattern': <str>, 'density': <float>} or {} if none found.
    """
    infill_data = {}
    # nested infill
    if isinstance(raw_spec.get('infill'), dict):
        infill_data.update(raw_spec['infill'])
    # flattened fields
    if 'infill_pattern' in raw_spec:
        infill_data['pattern'] = raw_spec.pop('infill_pattern')
    if 'infill_shape' in raw_spec:
        infill_data['pattern'] = raw_spec.pop('infill_shape')
    if 'infill_density' in raw_spec:
        infill_data['density'] = raw_spec.pop('infill_density')
    if 'infill_thickness_mm' in raw_spec:
        infill_data['density'] = raw_spec.pop('infill_thickness_mm')
    if 'infill_mm' in raw_spec:
        infill_data['density'] = raw_spec.pop('infill_mm')

    # Normalize 'type' to 'pattern' and 'thickness_mm' to 'density'
    if 'type' in infill_data:
        infill_data['pattern'] = infill_data.pop('type')
    if 'thickness_mm' in infill_data:
        infill_data['density'] = infill_data.pop('thickness_mm')

    pattern = infill_data.get('pattern')
    density = infill_data.get('density')
    if pattern is not None and density is not None:
        return {'pattern': pattern, 'density': density}
    return {}
# ai_adapter/csg_adapter.py
import json
import base64
from google.protobuf.json_format import ParseDict
from ai_adapter.schema.implicitus_pb2 import Model
import uuid
from google.protobuf import json_format

import logging
logging.basicConfig(level=logging.DEBUG)
from ai_adapter.inference_pipeline import generate as llm_generate

# Ensure generate_summary is available for import elsewhere if needed

# mapping from LLM 'shape' string to (proto field name, field-builder function)
_SHAPE_FIELD_MAP = {
    'sphere': ('sphere', lambda size: {'radius': size / 2}),
    'cube':   ('cube',   lambda size: {'size': size}),
    'box':    ('box',    lambda size: {'size': {'x': size, 'y': size, 'z': size}}),
    'cylinder': ('cylinder', lambda size: {'height': size, 'radius': size / 2}),
    # extend as more primitives are supported...
}

def _build_primitive_dict(spec):
    raw_shape = spec.get('shape', '').strip().lower()
    if raw_shape not in _SHAPE_FIELD_MAP:
        raise ValueError(f"Unknown shape '{raw_shape}' in spec")
    field_name, build_fields = _SHAPE_FIELD_MAP[raw_shape]
    params = build_fields(float(spec.get('size_mm') or spec.get('sizeMm')))
    primitive = {field_name: params}
    return primitive

def parse_raw_spec(llm_output):
    logging.debug(f"parse_raw_spec received input: {repr(llm_output)}")
    # If input is a string, attempt to parse JSON
    if isinstance(llm_output, str):
        try:
            llm_output = json.loads(llm_output)
        except json.JSONDecodeError as e:
            raise TypeError(f"Failed to parse JSON from string input: {e}")

    # Expect a dict with optional 'primitives' list, or a list of spec-dicts
    if isinstance(llm_output, dict) and 'primitives' in llm_output:
        specs = llm_output['primitives']
    elif isinstance(llm_output, dict) and 'shapes' in llm_output:
        specs = llm_output['shapes']
    elif isinstance(llm_output, dict):
        specs = [llm_output]
    elif isinstance(llm_output, list):
        specs = llm_output
    else:
        raise TypeError(f"Unsupported spec type: {type(llm_output)}")
    nodes = []
    for spec in specs:
        modifier = _build_modifier_dict(spec)
        primitive_params = _build_primitive_dict(spec)
        primitive_node = {'primitive': primitive_params}
        if modifier:
            infill_node = {
                'infill': modifier,
                'children': [primitive_node]
            }
            nodes.append(infill_node)
        else:
            nodes.append(primitive_node)
    return nodes

def interpret_llm_request(llm_output):
    logging.debug(f"interpret_llm_request received input: {repr(llm_output)}")
    """
    Interpret LLM output into a reviewable spec.
    Returns a dict:
      {
        "primitives": [ ... ],      # list of parsed primitive nodes
        "boolean": "union" | ...    # optional boolean operation
      }
    """
    # Normalize raw JSON
    if isinstance(llm_output, str):
        # first try to parse as JSON
        try:
            raw = json.loads(llm_output)
            logging.debug("interpret_llm_request parsed input as JSON: %r", raw)
        except json.JSONDecodeError:
            # not JSON, call the LLM to turn natural language into JSON
            logging.debug("interpret_llm_request calling LLM for JSON generation on: %r", llm_output)
            generated = llm_generate(llm_output)
            logging.debug("interpret_llm_request received generated JSON: %r", generated)
            raw = json.loads(generated)
            logging.debug("interpret_llm_request parsed generated JSON into dict: %r", raw)
            # Normalize flattened infill fields into a single nested dict
            if isinstance(raw, dict):
                pattern = raw.pop('infill_pattern', None) or raw.pop('infill_shape', None)
                density = raw.pop('infill_density', None)
                if density is None:
                    density = raw.pop('infill_thickness_mm', None) or raw.pop('infill_mm', None)
                if pattern is not None and density is not None:
                    raw['infill'] = {'pattern': pattern, 'density': density}
                    logging.debug("interpret_llm_request normalized infill: %r", raw['infill'])
    else:
        raw = llm_output
        # Normalize flattened infill fields into a single nested dict
        if isinstance(raw, dict):
            pattern = raw.pop('infill_pattern', None) or raw.pop('infill_shape', None)
            density = raw.pop('infill_density', None)
            if density is None:
                density = raw.pop('infill_thickness_mm', None) or raw.pop('infill_mm', None)
            if pattern is not None and density is not None:
                raw['infill'] = {'pattern': pattern, 'density': density}
                logging.debug("interpret_llm_request normalized infill: %r", raw['infill'])

    nodes = parse_raw_spec(raw)
    spec = {"primitives": nodes}
    boolean_op = raw.get("boolean")
    if boolean_op:
        spec["boolean"] = boolean_op
    return spec

def build_model_from_spec(nodes, boolean_op=None):
    # Wrap nodes into root based on boolean_op or count
    if boolean_op:
        root = {
            'booleanOp': {boolean_op: {}},
            'children': nodes
        }
    elif len(nodes) > 1:
        root = {'children': nodes}
    else:
        root = nodes[0]

    model_dict = {
        'id': str(uuid.uuid4()),
        'root': root
    }
    proto = Model()
    json_format.ParseDict(model_dict, proto)
    return proto

def parse_and_build_model(llm_output):
    """
    Parses the LLM output and returns a reviewable spec dict.
    Later, after user confirmation, pass this spec to build_model_from_spec.
    """
    return interpret_llm_request(llm_output)

def generate_summary(spec):
    segments = []
    for action in spec:
        # handle direct primitives
        if 'primitive' in action:
            prim = action['primitive']
            for shape, details in prim.items():
                if shape == 'sphere':
                    segments.append(f"sphere radius {details['radius']}")
                elif shape == 'box':
                    size = details['size']
                    segments.append(f"box size {size['x']}x{size['y']}x{size['z']}")
                elif shape == 'cylinder':
                    segments.append(f"cylinder radius {details['radius']} height {details['height']}")
                else:
                    segments.append(shape)
        # handle compound actions with children
        if 'children' in action:
            for child in action['children']:
                if 'primitive' in child:
                    prim = child['primitive']
                    for shape, details in prim.items():
                        if shape == 'sphere':
                            segments.append(f"sphere radius {details['radius']}")
                        elif shape == 'box':
                            size = details['size']
                            segments.append(f"box size {size['x']}x{size['y']}x{size['z']}")
                        elif shape == 'cylinder':
                            segments.append(f"cylinder radius {details['radius']} height {details['height']}")
                        else:
                            segments.append(shape)
        # handle modifiers like infill
        if 'infill' in action:
            infill = action['infill']
            pattern = infill.get('pattern')
            if 'density' in infill:
                segments.append(f"infill pattern {pattern} density {infill['density']}")
            elif 'thickness_mm' in infill:
                segments.append(f"infill pattern {pattern} thickness {infill['thickness_mm']}")
            else:
                segments.append(f"infill pattern {pattern}")
    return "; ".join(segments)

def review_request(request_data):
    """
    Takes the user's original request (either as raw JSON or wrapped in a single-key dict),
    normalizes it, runs it through our existing interpret_llm_request pipeline,
    and returns the primitive-spec list plus a human-readable summary.
    """
    # If the UI sends back an already-built spec, use it directly
    if isinstance(request_data, dict) and "spec" in request_data:
        spec = request_data["spec"]
        summary = generate_summary(spec)
        return spec, summary
    else:
        # If user passed a direct spec dict or list, skip LLM and parse directly
        if isinstance(request_data, dict) and ("shape" in request_data or "primitives" in request_data):
            spec = parse_raw_spec(request_data)
            summary = generate_summary(spec)
            return spec, summary

        # Unwrap single-key string/dict/list payloads
        raw = request_data
        if isinstance(request_data, dict) and len(request_data) == 1:
            first_val = next(iter(request_data.values()))
            if isinstance(first_val, str):
                try:
                    raw = json.loads(first_val)
                except (ValueError, json.JSONDecodeError):
                    raw = first_val
            elif isinstance(first_val, (dict, list)):
                raw = first_val

        # Interpret through our standard pipeline
        interpreted = interpret_llm_request(raw)
        spec = interpreted.get("primitives", [])
        summary = generate_summary(spec)
        return spec, summary
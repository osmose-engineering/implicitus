# ai_adapter/csg_adapter.py
import json
import base64
from google.protobuf.json_format import ParseDict
from ai_adapter.schema.implicitus_pb2 import Model
import uuid
from google.protobuf import json_format


import logging
logging.basicConfig(level=logging.DEBUG)

# Helper for bounding box calculation for primitives
def _compute_bbox_from_primitive(prim: dict):
    """
    Given a primitive dict, return (bbox_min, bbox_max) in object space.
    """
    if 'box' in prim:
        sz = prim['box']['size']
        return (0.0, 0.0, 0.0), (sz['x'], sz['y'], sz['z'])
    if 'cube' in prim:
        s = prim['cube']['size']
        return (0.0, 0.0, 0.0), (s, s, s)
    if 'sphere' in prim:
        r = prim['sphere']['radius']
        return (-r, -r, -r), (r, r, r)
    if 'cylinder' in prim:
        h = prim['cylinder']['height']
        r = prim['cylinder']['radius']
        return (-r, -r, 0.0), (r, r, h)
    # default fallback
    return (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
from ai_adapter.inference_pipeline import generate as llm_generate

import re

# declarative requirements for primitives and modifiers
PRIMITIVE_REQUIREMENTS = {
    "sphere": {"oneOf": [["size_mm", "radius_mm"]]},
    "cube":   {"required": ["size_mm"]},
    "box":    {"required": ["size_mm"]},
    "cylinder": {"oneOf": [["size_mm"], ["radius_mm", "height_mm"]]},
    # extend with more primitives as needed...
    "voronoi": {
        "required": ["seed_points", "bbox_min", "bbox_max"],
        "oneOf": [["min_dist"], ["density_field"], ["scale_field"]]
    },
}

MODIFIER_REQUIREMENTS = {
    # Infill must include a pattern; density is optional for voronoi
    "infill": {
        "required": ["pattern"],
        "oneOf": [["density"], []]
    },
    # extend with more modifiers (shell, boolean, etc.) as needed...
}

def _check_requirements(raw_spec: dict, shape: str) -> list[str]:
    """
    Return a list of missing parameter descriptions for the given primitive shape.
    """
    req = PRIMITIVE_REQUIREMENTS.get(shape.lower())
    if not req:
        return [f"unknown primitive '{shape}'"]
    missing = []
    for key in req.get("required", []):
        if key not in raw_spec or raw_spec.get(key) is None:
            missing.append(key)
    for group in req.get("oneOf", []):
        if not any(k in raw_spec and raw_spec.get(k) is not None for k in group):
            missing.append(f"one of {group}")
    return missing

def _find_missing_fields(raw_spec):
    """
    Inspect one or more raw spec dict(s) and return a list of missing key descriptions.
    """
    missing = []
    specs = raw_spec if isinstance(raw_spec, list) else [raw_spec]
    for spec in specs:
        shape = spec.get("shape")
        if not shape:
            missing.append("shape")
            continue
        # primitive‐level checks
        missing += _check_requirements(spec, shape)
        # modifier‐level checks
        for mod, mod_req in MODIFIER_REQUIREMENTS.items():
            if mod in spec and isinstance(spec[mod], dict):
                for key in mod_req.get("required", []):
                    if key not in spec[mod]:
                        missing.append(f"{shape} {mod}: missing {key}")
    return missing

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
    # Support flattened infill_type field
    if 'infill_type' in raw_spec:
        infill_data['pattern'] = raw_spec.pop('infill_type')

    # Support flattened infill_voronoi boolean flag
    if raw_spec.get('infill_voronoi'):
        infill_data['pattern'] = 'voronoi'
        # leave density to downstream defaults if missing

    # Normalize 'type' to 'pattern' and 'thickness_mm' to 'density'
    if 'type' in infill_data:
        infill_data['pattern'] = infill_data.pop('type')
    if 'thickness_mm' in infill_data:
        infill_data['density'] = infill_data.pop('thickness_mm')

    pattern = infill_data.get('pattern')
    # Mark Voronoi infill specially
    if pattern == 'voronoi':
        infill_data['_is_voronoi'] = True
    density = infill_data.get('density')
    if pattern is not None and density is not None:
        return {'pattern': pattern, 'density': density, **({k: v for k, v in infill_data.items() if k not in ('pattern', 'density')})}
    # Allow just pattern (for voronoi, density may be omitted)
    if pattern is not None:
        return {'pattern': pattern, **({k: v for k, v in infill_data.items() if k != 'pattern'})}
    return {}

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
    if raw_shape == 'voronoi':
        allowed = {
            'seed_points', 'bbox_min', 'bbox_max',
            'min_dist', 'density_field', 'scale_field',
            'adaptive', 'max_depth', 'threshold', 'error_metric',
            'resolution', 'wall_thickness', 'shell_offset',
            'blend_curve', 'csg_ops', 'auto_cap', 'cap_blend'
        }
        params = {k: spec[k] for k in allowed if k in spec}
        return {'voronoi': params}
    # Special handling for cylinder when size_mm is a dict with radius and height
    if raw_shape == 'cylinder' and isinstance(spec.get('size_mm'), dict):
        size_info = spec['size_mm']
        radius = float(size_info.get('radius', size_info.get('radius_mm', 0)))
        height = float(size_info.get('height', size_info.get('height_mm', 0)))
        return {'cylinder': {'radius': radius, 'height': height}}
    # Special handling for cylinder with explicit radius_mm and/or height_mm
    if raw_shape == 'cylinder' and ('radius_mm' in spec or 'height_mm' in spec):
        radius = float(spec.get('radius_mm', spec.get('radiusMm', 0)))
        height = float(spec.get('height_mm', spec.get('heightMm', 0)))
        return {'cylinder': {'radius': radius, 'height': height}}
    # Special handling for sphere: use radius_mm directly if given, otherwise treat size_mm as diameter
    if raw_shape == 'sphere':
        # prefer explicit radius
        if 'radius_mm' in spec or 'radiusMm' in spec:
            radius = float(spec.get('radius_mm') or spec.get('radiusMm'))
        else:
            diameter = float(spec.get('size_mm') or spec.get('sizeMm'))
            radius = diameter / 2.0
        return {'sphere': {'radius': radius}}
    if raw_shape not in _SHAPE_FIELD_MAP:
        raise ValueError(f"Unknown shape '{raw_shape}' in spec")
    field_name, build_fields = _SHAPE_FIELD_MAP[raw_shape]
    params = build_fields(float(spec.get('size_mm') or spec.get('sizeMm')))
    primitive = {field_name: params}
    return primitive

def parse_raw_spec(llm_output):
    logging.debug(f"parse_raw_spec received input: {repr(llm_output)}")
    # If the input is already a list of nodes in our internal format, return as is.
    def _is_internal_node_dict(d):
        # Only a dict with exactly key "primitive", or with exactly keys "infill" and "children"
        if not isinstance(d, dict):
            return False
        keys = set(d.keys())
        if keys == {"primitive"}:
            return True
        if keys == {"infill", "children"}:
            return True
        return False

    if isinstance(llm_output, list) and all(_is_internal_node_dict(item) for item in llm_output):
        return llm_output

    # If the input is already a single node dict in internal format, wrap it in a list.
    if _is_internal_node_dict(llm_output):
        return [llm_output]

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

    # Handle update calls: dict with existing 'spec' (list) and a new 'raw' instruction
    if isinstance(llm_output, dict) and "spec" in llm_output and "raw" in llm_output:
        logging.debug("interpret_llm_request handling update: spec + raw")
        old_spec = llm_output["spec"]
        raw_text = llm_output["raw"]
        # Craft a prompt giving the LLM the existing spec and the new instruction
        prompt = (
            f"Here is the current primitive spec list:\n{json.dumps(old_spec, indent=2)}\n"
            f"Please apply this modification: {raw_text}\n"
            "Return only the updated list of primitive spec dicts as JSON."
        )
        generated = llm_generate(prompt)
        logging.debug("interpret_llm_request received update-generated JSON: %r", generated)
        # Attempt to parse the last JSON block in case LLM returned multiple arrays
        text = generated.strip()
        parts = re.split(r'\n\s*\n', text)
        last_block = parts[-1]
        try:
            update_list = json.loads(last_block)
        except json.JSONDecodeError:
            try:
                update_list = json.loads(parts[0])
            except json.JSONDecodeError:
                raise RuntimeError(f"Failed to parse LLM update JSON: {generated}")
        logging.debug("interpret_llm_request parsed update-generated JSON into dict: %r", update_list)
        # Immediately return the updated primitives list, skipping missing-field checks
        nodes = parse_raw_spec(update_list)
        return {"primitives": nodes}
    else:
        raw = llm_output

    # Normalize raw JSON
    if isinstance(raw, str):
        # first try to parse as JSON
        try:
            raw = json.loads(raw)
            logging.debug("interpret_llm_request parsed input as JSON: %r", raw)
        except json.JSONDecodeError:
            # not JSON, call the LLM to turn natural language into JSON
            logging.debug("interpret_llm_request calling LLM for JSON generation on: %r", raw)
            generated = llm_generate(raw)
            logging.debug("interpret_llm_request received generated JSON: %r", generated)
            raw = json.loads(generated)
            logging.debug("interpret_llm_request parsed generated JSON into dict: %r", raw)

    # Check for any required fields that are still missing
    missing = _find_missing_fields(raw)
    if missing:
        # Ask the LLM to generate exactly one clarification question
        prompt = (
            f"You have this incomplete CSG spec:\n{json.dumps(raw, indent=2)}\n"
            f"The following data is missing: {', '.join(missing)}.\n"
            "Please ask the user exactly one clear, concise question to obtain the missing information."
        )
        question = llm_generate(prompt)
        logging.debug("interpret_llm_request clarification question: %r", question)
        # Parse the LLM's JSON response for the question
        try:
            q_dict = json.loads(question)
            return {"needsClarification": True, **q_dict}
        except json.JSONDecodeError:
            return {"needsClarification": True, "question": question}

    # Apply our unified infill normalization
    if isinstance(raw, dict):
        modifier = _build_modifier_dict(raw)
        if modifier:
            raw['infill'] = modifier
            logging.debug("interpret_llm_request applied _build_modifier_dict: %r", modifier)

    nodes = parse_raw_spec(raw)
    # Transform voronoi-marked infill into a standalone primitive
    new_nodes = []
    for node in nodes:
        if 'infill' in node and node['infill'].get('_is_voronoi'):
            prim = node['children'][0]['primitive']
            bbox_min, bbox_max = _compute_bbox_from_primitive(prim)
            size = [bbox_max[i] - bbox_min[i] for i in range(3)]
            default_min_dist = max(size) / 20.0
            default_wall = size[2] / 50.0
            vor_spec = {
                'shape': 'voronoi',
                'seed_points': [],
                'bbox_min': list(bbox_min),
                'bbox_max': list(bbox_max),
                'min_dist': default_min_dist,
                'wall_thickness': default_wall,
                'adaptive': True,
                'max_depth': 3,
                'threshold': 0.1,
                'resolution': [32, 32, 32],
                'shell_offset': 0.0,
                'auto_cap': False
            }
            new_nodes.append({'primitive': {'voronoi': vor_spec}})
        else:
            new_nodes.append(node)
    nodes = new_nodes
    spec = {"primitives": nodes}
    boolean_op = raw.get("boolean") if isinstance(raw, dict) else None
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
            density = infill.get('density')
            # determine primary shape name
            if 'children' in action and action['children']:
                child = action['children'][0]
                prim = child.get('primitive', {})
                shape = next(iter(prim.keys()), '')
            elif 'primitive' in action:
                shape = next(iter(action['primitive'].keys()), '')
            else:
                shape = ''
            segments.append(f"{shape} infill pattern {pattern} density {density}")
    return "; ".join(segments)


# Helper: confirm_change
def confirm_change(old_nodes, new_nodes, instruction):
    """
    Compare the old and new primitive lists and
    ask the LLM to produce a one-line confirmation of what changed.
    """
    prompt = (
        f"I had this CSG spec:\n{json.dumps(old_nodes, indent=2)}\n"
        f"Then you applied: {instruction}\n"
        f"New spec is:\n{json.dumps(new_nodes, indent=2)}\n"
        "Please confirm back to me in one sentence what changed."
    )
    confirmation = llm_generate(prompt)
    return confirmation.strip()

def review_request(request_data):
    """
    Takes the user's original request (either as raw JSON or wrapped in a single-key dict),
    normalizes it, runs it through our existing interpret_llm_request pipeline,
    and returns the primitive-spec list plus a human-readable summary.
    """
    # If our adapter asked a clarification question, propagate it back to the UI
    if isinstance(request_data, dict) and request_data.get("needsClarification"):
        return {"needsClarification": True, "question": request_data["question"]}
    # Follow-up update: apply a new raw instruction to an existing spec
    if isinstance(request_data, dict) and "spec" in request_data and "raw" in request_data:
        interpreted = interpret_llm_request(request_data)
        spec = interpreted.get("primitives", [])
        summary = generate_summary(spec)
        return spec, summary

    # Manual JSON-only spec from UI: just summarize and return
    if isinstance(request_data, dict) and "spec" in request_data:
        spec = request_data["spec"]
        summary = generate_summary(spec)
        return spec, summary

    # Manual JSON-only spec from UI: parse and summarize directly
    if isinstance(request_data, dict) and ("shape" in request_data or "primitives" in request_data):
        # parse the raw spec without calling the LLM
        nodes = parse_raw_spec(request_data)
        # Transform voronoi-marked infill into standalone primitive
        new_nodes = []
        for node in nodes:
            if 'infill' in node and node['infill'].get('_is_voronoi'):
                prim = node['children'][0]['primitive']
                bbox_min, bbox_max = _compute_bbox_from_primitive(prim)
                size = [bbox_max[i] - bbox_min[i] for i in range(3)]
                default_min_dist = max(size) / 20.0
                default_wall = size[2] / 50.0
                vor_spec = {
                    'shape': 'voronoi',
                    'seed_points': [],
                    'bbox_min': list(bbox_min),
                    'bbox_max': list(bbox_max),
                    'min_dist': default_min_dist,
                    'wall_thickness': default_wall,
                    'adaptive': True,
                    'max_depth': 3,
                    'threshold': 0.1,
                    'resolution': [32, 32, 32],
                    'shell_offset': 0.0,
                    'auto_cap': False
                }
                new_nodes.append({'primitive': {'voronoi': vor_spec}})
            else:
                new_nodes.append(node)
        summary = generate_summary(new_nodes)
        return new_nodes, summary

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
    if isinstance(interpreted, dict) and interpreted.get("needsClarification"):
        response = {
            "needsClarification": True,
            "question": interpreted["question"]
        }
        # Include sid if present in the original request
        sid = request_data.get("sid") if isinstance(request_data, dict) else None
        if sid:
            response["sid"] = sid
        return response
    spec = interpreted.get("primitives", [])
    summary = generate_summary(spec)
    return spec, summary


def update_request(sid: str, spec: list, raw: str):
    """
    Apply a follow-up instruction `raw` to an existing spec list,
    then confirm what changed.
    Returns a tuple: (new_spec, new_summary)
    """
    # Quick parse for simple infill commands like "add a 2mm gyroid infill"
    m = re.search(r'(\d+(?:\.\d+)?)\s*mm\s+(\w+)\s+infill', raw.lower())
    if m:
        thickness = float(m.group(1))
        pattern = m.group(2)
        # apply infill to all primitives
        for action in spec:
            action['infill'] = {'pattern': pattern, 'density': thickness}
        summary = f"Added {thickness}mm {pattern} infill"
        # Return updated spec and summary, without LLM confirmation
        return spec, summary

    # Quick parse for simple boolean operations like "union", "difference", or "intersection"
    cmd = raw.lower()
    if 'union' in cmd or 'difference' in cmd or 'intersection' in cmd:
        # Determine operation
        if 'union' in cmd:
            op = 'union'
        elif 'difference' in cmd:
            op = 'difference'
        else:
            op = 'intersection'
        summary = f"Applied {op} operation to shapes"
        return spec, summary
    old_spec = spec
    # Reuse review_request to get updated spec
    request_data = {"raw": raw, "spec": old_spec, "sid": sid}
    intermediate_spec, _ = review_request(request_data)

    # Post-process to promote any Voronoi infill modifiers into standalone primitives
    promoted = []
    for node in intermediate_spec:
        if 'infill' in node and node['infill'].get('_is_voronoi'):
            prim = node['children'][0]['primitive']
            bbox_min, bbox_max = _compute_bbox_from_primitive(prim)
            size = [bbox_max[i] - bbox_min[i] for i in range(3)]
            default_min_dist = max(size) / 20.0
            default_wall = size[2] / 50.0
            vor_spec = {
                'shape': 'voronoi',
                'seed_points': [],
                'bbox_min': list(bbox_min),
                'bbox_max': list(bbox_max),
                'min_dist': default_min_dist,
                'wall_thickness': default_wall,
                'adaptive': True,
                'max_depth': 3,
                'threshold': 0.1,
                'resolution': [32, 32, 32],
                'shell_offset': 0.0,
                'auto_cap': False
            }
            promoted.append({'primitive': {'voronoi': vor_spec}})
        else:
            promoted.append(node)

    new_spec = promoted
    new_summary = generate_summary(new_spec)
    return new_spec, new_summary
import uuid
from ai_adapter.schema.implicitus_pb2 import Primitive

def map_primitive(spec: dict) -> dict:
    # unwrap wrapper nodes with 'primitive'
    if isinstance(spec, dict) and 'primitive' in spec:
        inner = spec['primitive']
        prim_type, prim_vals = next(iter(inner.items()))
        if prim_type == 'sphere':
            spec = {'shape': 'sphere', 'size_mm': prim_vals['radius'] * 2}
        elif prim_type == 'box':
            size_dict = prim_vals['size']
            # uniform cube
            if size_dict['x'] == size_dict['y'] == size_dict['z']:
                spec = {'shape': 'box', 'size_mm': size_dict['x']}
            else:
                spec = {'shape': 'box', 'size': (size_dict['x'], size_dict['y'], size_dict['z'])}
        elif prim_type == 'cylinder':
            spec = {
                'shape': 'cylinder',
                'radius_mm': prim_vals['radius'],
                'height_mm': prim_vals['height']
            }
        else:
            raise ValueError(f"Unknown primitive type: {prim_type}")
    # handle grouping nodes with 'children'
    if isinstance(spec, dict) and 'children' in spec:
        # flatten all children primitives
        children = spec['children']
        return [map_primitive(child) for child in children]
    # Handle a list of primitive specs by mapping each element
    if isinstance(spec, list):
        return [map_primitive(s) for s in spec]
    id_str = str(uuid.uuid4())
    shape = spec.get("shape", "").lower()
    if shape == "sphere":
        radius = spec["size_mm"] / 2
        primitive = {"sphere": {"radius": radius}}
    elif shape in ("cube", "box"):
        size = spec.get("size_mm", spec.get("size", 0))
        if isinstance(size, (int, float)):
            size_dict = {"x": size, "y": size, "z": size}
        else:
            size_dict = {"x": size[0], "y": size[1], "z": size[2]}
        primitive = {"box": {"size": size_dict}}
    elif shape == "cylinder":
        primitive = {"cylinder": {"radius": spec["radius_mm"], "height": spec["height_mm"]}}
    else:
        raise ValueError(f"Unknown shape: {shape}")
    return {"id": id_str, "root": {"primitive": primitive}}

def get_response_fields(proto_model):
    """
    Extract shape and parameters from a Model protobuf message.

    Returns:
        shape (str): e.g., "sphere", "cube", or "cylinder"
        params (dict): keys are dimension names and numeric values
    """
    primitive = proto_model.root.primitive
    primitive_type = primitive.type
    params = {}
    if primitive_type == "sphere":
        radius = primitive.params.get("radius", 0)
        return "sphere", {"size_mm": radius * 2}
    elif primitive_type in ("cube", "box"):
        x = primitive.params.get("x", 0)
        y = primitive.params.get("y", 0)
        z = primitive.params.get("z", 0)
        if x == y == z:
            return "cube", {"size_mm": x}
        else:
            return "box", {"size_x_mm": x, "size_y_mm": y, "size_z_mm": z}
    elif primitive_type == "cylinder":
        radius = primitive.params.get("radius", 0)
        height = primitive.params.get("height", 0)
        return "cylinder", {"radius_mm": radius, "height_mm": height}
    else:
        raise ValueError(f"Unsupported primitive type in proto: {primitive_type}")


# New function for mapping spec to proto dict model
def map_to_proto_dict(spec):
    """
    Convert a spec (dict or list of specs) into a full Model dict ready for JSON-to-proto.
    If multiple primitives, wraps them in a union booleanOp under root.
    """
    mapped = map_primitive(spec)
    # If map_primitive returned a list, wrap in a union op
    if isinstance(mapped, list):
        model_id = str(uuid.uuid4())
        return {
            "id": model_id,
            "root": {
                "booleanOp": {"union": {}},
                "children": mapped
            }
        }
    # Otherwise it's already a single-node dict
    return mapped

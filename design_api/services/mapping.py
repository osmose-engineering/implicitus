import uuid
from ai_adapter.schema.implicitus_pb2 import Primitive
from ai_adapter.schema.implicitus_pb2 import Modifier, Infill, Shell, BooleanOp, VoronoiLattice
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class SomeMappingError(Exception):
    """Raised when mapping a primitive spec fails due to unknown shape."""
    pass

def _map_base_shape(spec: dict) -> dict:
    """
    Internal: map a simple shape spec to the proto dict structure.
    """
    id_str = str(uuid.uuid4())
    shape = spec['shape'].lower()
    if shape == 'sphere':
        radius = spec['size_mm'] / 2
        primitive = {'sphere': {'radius': radius}}
    elif shape in ('cube', 'box'):
        size = spec.get('size_mm', spec.get('size', 0))
        if isinstance(size, (int, float)):
            size_dict = {'x': size, 'y': size, 'z': size}
        else:
            size_dict = {'x': size[0], 'y': size[1], 'z': size[2]}
        primitive = {'box': {'size': size_dict}}
    elif shape == 'cylinder':
        primitive = {'cylinder': {'radius': spec['radius_mm'], 'height': spec['height_mm']}}
    else:
        raise SomeMappingError(f"Unknown shape: {shape}")
    return {'id': id_str, 'root': {'primitive': primitive}}

def map_primitive(node: dict) -> dict:
    logger.debug(f"map_primitive called with node: {node}")
    """
    Convert a primitive node with optional modifiers into a proto-ready dict.
    Applies modifiers in order: shell -> infill -> boolean.
    """
    # Extract modifiers if present
    modifiers = node.get('modifiers', {})
    logger.debug(f"Modifiers: {modifiers}")
    # Unwrap the base primitive spec
    if isinstance(node, dict) and 'primitive' in node:
        inner = node['primitive']
        prim_type, prim_vals = next(iter(inner.items()))
        # Build base spec for mapping
        base_spec = {'shape': prim_type}
        if prim_type == 'sphere':
            base_spec['size_mm'] = prim_vals['radius'] * 2
        elif prim_type == 'box':
            size_dict = prim_vals['size']
            if size_dict['x'] == size_dict['y'] == size_dict['z']:
                base_spec['size_mm'] = size_dict['x']
            else:
                base_spec['size'] = (size_dict['x'], size_dict['y'], size_dict['z'])
        elif prim_type == 'cylinder':
            base_spec['radius_mm'] = prim_vals['radius']
            base_spec['height_mm'] = prim_vals['height']
        else:
            raise SomeMappingError(f"Unknown primitive type: {prim_type}")
    else:
        # Already a raw spec dict
        base_spec = node

    # Map base primitive to proto dict
    mapped = _map_base_shape(base_spec)  # Helper that returns {"id":..., "root":{...}}
    root = mapped['root']

    # Apply shell modifier
    if 'shell' in modifiers:
        shell_params = modifiers['shell']
        root = {
            "booleanOp": {"union": {}},
            "children": [root, {"primitive": {"shell": shell_params}}]
        }

    # Apply infill modifier (supports Voronoi)
    if 'infill' in modifiers:
        logger.debug(f"Applying infill with params: {modifiers.get('infill')}")
        infill_params = modifiers['infill']
        root = {
            "booleanOp": {"intersection": {}},
            "children": [
                root,
                {"primitive": {'lattice': infill_params}}
            ]
        }

    # Apply boolean_op modifier
    if 'boolean_op' in modifiers:
        bool_params = modifiers['boolean_op']
        root = {
            "booleanOp": {bool_params['op']: {}},
            "children": [root, map_primitive(bool_params['shape_node'])]
        }

    # Return final wrapped dict
    mapped['root'] = root
    logger.debug(f"map_primitive output mapped: {mapped}")
    return mapped

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
    logger.debug(f"map_to_proto_dict called with spec: {spec}")
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

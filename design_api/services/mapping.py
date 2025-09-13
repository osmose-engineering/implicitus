import uuid
from ai_adapter.schema.implicitus_pb2 import Primitive
from ai_adapter.schema.implicitus_pb2 import Modifier, Infill, Shell, BooleanOp, VoronoiLattice
import logging

logger = logging.getLogger(__name__)

# Logging configuration is handled by the application entry point. Guard the
# configuration call so importing this module doesn't overwrite global logging
# settings.
if __name__ == "__main__":  # pragma: no cover - manual debugging helper
    logging.basicConfig(level=logging.DEBUG)

class SomeMappingError(Exception):
    """Raised when mapping a primitive spec fails due to unknown shape."""
    pass


def _ensure_node_lists(node: dict | list) -> None:
    """Recursively add empty list fields to mapping results.

    Every node in the JSON representation is expected to include ``children``,
    ``modifiers`` and ``constraints`` keys. When mapping primitives and
    composing boolean operations, some nodes might omit these fields. This
    helper mutates ``node`` in place to guarantee the lists exist.
    """

    if isinstance(node, dict):
        node.setdefault("children", [])
        node.setdefault("modifiers", [])
        node.setdefault("constraints", [])
        for child in node.get("children", []):
            _ensure_node_lists(child)
    elif isinstance(node, list):
        for item in node:
            _ensure_node_lists(item)

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
    # Ensure leaf nodes include explicit empty lists for consistency
    return {
        'id': id_str,
        'root': {'primitive': primitive, 'children': [], 'modifiers': [], 'constraints': []},
        'constraints': [],
        'modifiers': [],
    }

def map_primitive(node: dict, request_id: str | None = None) -> dict:
    logger.debug(
        "map_primitive called",
        extra={"request_id": request_id, "node": node},
    )
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
            "children": [
                root,
                {"primitive": {"shell": shell_params}, "children": [], "modifiers": []},
            ],
            "modifiers": [],
        }

    # Apply infill modifier (supports Voronoi)
    if 'infill' in modifiers:
        logger.debug(
            "Applying infill",
            extra={"request_id": request_id, "params": modifiers.get("infill")},
        )
        infill_params = modifiers['infill']
        if 'bboxMin' in infill_params or 'bboxMax' in infill_params:
            raise SomeMappingError("Bounding box keys must use snake_case (bbox_min/bbox_max)")
        root = {
            "booleanOp": {"intersection": {}},
            "children": [
                root,
                {"primitive": {'lattice': infill_params}, "children": [], "modifiers": []},
            ],
            "modifiers": [],
        }

    # Apply boolean_op modifier
    if 'boolean_op' in modifiers:
        bool_params = modifiers['boolean_op']
        root = {
            "booleanOp": {bool_params['op']: {}},
            "children": [
                root,
                map_primitive(bool_params['shape_node'], request_id=request_id),
            ],
            "modifiers": [],
        }

    # Return final wrapped dict with version information
    _ensure_node_lists(root)
    mapped['root'] = root
    # Attach a top-level version so downstream consumers can assert
    # compatibility with the spec format.
    mapped['version'] = 1
    mapped.setdefault('constraints', [])
    mapped.setdefault('modifiers', [])
    logger.debug("map_primitive output mapped", extra={"request_id": request_id, "mapped": mapped})
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
def map_to_proto_dict(spec, request_id: str | None = None):
    request_id = request_id or str(uuid.uuid4())
    logger.debug(
        "map_to_proto_dict called",
        extra={"request_id": request_id, "spec": spec},
    )
    """
    Convert a spec (dict or list of specs) into a full Model dict ready for JSON-to-proto.
    If multiple primitives, wraps them in a union booleanOp under root.
    """
    mapped = map_primitive(spec, request_id=request_id)
    # If map_primitive returned a list, wrap in a union op
    if isinstance(mapped, list):
        model_id = str(uuid.uuid4())
        root = {
            "booleanOp": {"union": {}},
            "children": mapped,
            "modifiers": [],
        }
        _ensure_node_lists(root)
        return {
            "id": model_id,
            "version": 1,
            "root": root,
            "constraints": [],
            "modifiers": [],
        }
    # Log uniform seed points and associated lattice data when present
    infill = spec.get("modifiers", {}).get("infill", {}) if isinstance(spec, dict) else {}
    seeds = infill.get("seed_points") or []
    cell_vertices = infill.get("cell_vertices") or []
    edge_list = infill.get("edge_list") or []
    if seeds and cell_vertices and edge_list:
        for idx, seed in enumerate(seeds):
            logger.info(
                "uniform_seed_cell",
                extra={
                    "request_id": request_id,
                    "seed_index": idx,
                    "seed_point": seed,
                    "cell_vertices": cell_vertices,
                    "edge_indices": edge_list,
                },
            )

    # Otherwise it's already a single-node dict
    _ensure_node_lists(mapped.get("root"))
    mapped.setdefault("constraints", [])
    mapped.setdefault("modifiers", [])
    return mapped

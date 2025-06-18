import uuid
from ai_adapter.schema.implicitus_pb2 import Primitive

def map_primitive(spec: dict) -> dict:
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

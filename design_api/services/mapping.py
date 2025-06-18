import uuid
from ai_adapter.schema.implicitus_pb2 import Primitive

def map_primitive(spec: dict) -> dict:
    id_str = str(uuid.uuid4())
    shape = spec.get("shape", "").lower()
    if shape == "sphere":
        radius = spec["size_mm"] / 2
        primitive_dict = {"primitive": {"sphere": {"radius": str(radius)}}}
        return {
            "id": id_str,
            "root": primitive_dict
        }
    elif shape == "cube":
        size_mm = spec["size_mm"]
        primitive_dict = {"primitive": {"box": {"size": {"x": str(size_mm), "y": str(size_mm), "z": str(size_mm)}}}}
        return {
            "id": id_str,
            "root": primitive_dict
        }
    elif shape == "box":
        size = spec.get("size_mm") or spec.get("size")
        if isinstance(size, (int, float)):
            size_list = [str(size), str(size), str(size)]
        elif isinstance(size, (list, tuple)) and len(size) == 3:
            size_list = [str(x) for x in size]
        else:
            raise ValueError(f"Invalid size for box: {size}")
        primitive_dict = {"primitive": {"box": {"size": {"x": size_list[0], "y": size_list[1], "z": size_list[2]}}}}
        return {
            "id": id_str,
            "root": primitive_dict
        }
    elif shape == "cylinder":
        radius_mm = spec["radius_mm"]
        height_mm = spec["height_mm"]
        primitive_dict = {"primitive": {"cylinder": {"radius": str(radius_mm), "height": str(height_mm)}}}
        return {
            "id": id_str,
            "root": primitive_dict
        }
    else:
        raise ValueError(f"Unknown shape: {shape}")

def get_response_fields(proto_model):
    """
    Extract shape and parameters from a Model protobuf message.

    Returns:
        shape (str): e.g., "sphere", "cube", or "cylinder"
        params (dict): keys are dimension names and numeric values
    """
    primitive = proto_model.root.primitive
    # Sphere
    if primitive.HasField("sphere"):
        radius = primitive.sphere.radius
        return "sphere", {"size_mm": radius * 2}
    # Box / Cube
    if primitive.HasField("box"):
        size = primitive.box.size
        # if all equal, treat as cube
        if size.x == size.y == size.z:
            return "cube", {"size_mm": size.x}
        else:
            return "box", {"size_x_mm": size.x, "size_y_mm": size.y, "size_z_mm": size.z}
    # Cylinder
    if primitive.HasField("cylinder"):
        cyl = primitive.cylinder
        return "cylinder", {"radius_mm": cyl.radius, "height_mm": cyl.height}
    # Fallback: unknown
    raise ValueError(f"Unsupported primitive type in proto: {primitive.WhichOneof('type')}")

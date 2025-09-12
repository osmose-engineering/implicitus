import pytest

from design_api.services.mapping import map_primitive
from design_api.services.mapping import SomeMappingError

def test_map_sphere_minimal():
    inp = {"shape":"sphere","size_mm":10}
    out = map_primitive(inp)
    assert out["root"]["primitive"]["sphere"]["radius"] == 5.0
    assert out["root"]["children"] == []

def test_map_box_nonuniform():
    inp = {"shape":"box","size_mm":[10,20,30]}
    out = map_primitive(inp)
    sz = out["root"]["primitive"]["box"]["size"]
    assert (sz["x"], sz["y"], sz["z"]) == (10,20,30)
    assert out["root"]["children"] == []

def test_map_unknown_shape_raises():
    with pytest.raises(SomeMappingError):
        map_primitive({"shape":"torus","r":5})

def test_map_box_with_infill():
    # Box with a voronoi infill modifier should produce an intersection CSG node
    node = {
        "primitive": {"box": {"size": {"x": 10.0, "y": 10.0, "z": 10.0}}},
        "modifiers": {
            "infill": {"pattern": "voronoi", "min_dist": 0.5, "wall_thickness": 0.1}
        }
    }
    out = map_primitive(node)
    root = out["root"]
    # Should be an intersection boolean for infill
    assert "booleanOp" in root
    assert list(root["booleanOp"].keys())[0] == "intersection"
    children = root["children"]
    assert len(children) == 2
    # First child: original box primitive
    assert "primitive" in children[0]
    assert "box" in children[0]["primitive"]
    assert children[0]["children"] == []
    # Second child: lattice primitive
    assert "primitive" in children[1]
    assert "lattice" in children[1]["primitive"]
    assert children[1]["children"] == []
    lattice = children[1]["primitive"]["lattice"]
    assert lattice["pattern"] == "voronoi"
    assert lattice["min_dist"] == 0.5
    assert lattice["wall_thickness"] == 0.1

def test_map_sphere_with_shell():
    # Sphere with a shell modifier should produce a union CSG node
    node = {
        "primitive": {"sphere": {"radius": 5.0}},
        "modifiers": {
            "shell": {"thickness_mm": 2.0}
        }
    }
    out = map_primitive(node)
    root = out["root"]
    # Should be a union boolean for shell
    assert "booleanOp" in root
    assert list(root["booleanOp"].keys())[0] == "union"
    children = root["children"]
    assert len(children) == 2
    # Second child: shell primitive
    assert "primitive" in children[1]
    assert "shell" in children[1]["primitive"]
    assert children[1]["children"] == []
    shell = children[1]["primitive"]["shell"]
    assert shell["thickness_mm"] == 2.0
import pytest

from design_api.services.mapping import map_primitive

def test_map_sphere_minimal():
    inp = {"shape":"sphere","size_mm":10}
    out = map_primitive(inp)
    assert out["root"]["primitive"]["sphere"]["radius"] == 5.0

def test_map_box_nonuniform():
    inp = {"shape":"box","size_mm":[10,20,30]}
    out = map_primitive(inp)
    sz = out["root"]["primitive"]["box"]["size"]
    assert (sz["x"], sz["y"], sz["z"]) == (10,20,30)

def test_map_unknown_shape_raises():
    with pytest.raises(SomeMappingError):
        map_primitive({"shape":"torus","r":5})
# tests/ai_adapter/test_csg_adapter.py
import pytest
import json
from ai_adapter.csg_adapter import parse_raw_spec, generate_summary, review_request, update_request, MAX_SEED_POINTS
from design_api.services.voronoi_gen.voronoi_gen  import _call_sdf

def test_parse_sphere_spec():
    raw = '{"shape":"sphere","size_mm":10}'
    spec = parse_raw_spec(raw)
    assert isinstance(spec, list)
    assert len(spec) == 1
    sphere_action = spec[0]
    assert 'primitive' in sphere_action
    sphere = sphere_action['primitive']['sphere']
    assert sphere['radius'] == 5.0

def test_invalid_shape_spec():
    raw = '{"shape":"torus","size_mm":10}'
    with pytest.raises(ValueError):
        parse_raw_spec(raw)

def test_parse_multiple_primitives_spec():
    raw = '{"primitives":[{"shape":"sphere","size_mm":8},{"shape":"cylinder","size_mm":10}]}'
    spec = parse_raw_spec(raw)
    assert isinstance(spec, list)
    assert len(spec) == 2
    for action in spec:
        assert 'primitive' in action
    sphere = spec[0]['primitive']['sphere']
    assert sphere['radius'] == 4.0
    cyl = spec[1]['primitive']['cylinder']
    assert cyl['radius'] == 5.0
    assert cyl['height'] == 10.0

def test_parse_infill_modifier_spec():
    raw = '{"shape":"box","size_mm":12,"infill":{"pattern":"hex","density":0.5}}'
    spec = parse_raw_spec(raw)
    assert isinstance(spec, list)
    assert len(spec) == 1
    action = spec[0]
    # Primitive should be present
    assert 'primitive' in action
    box = action['primitive']['box']['size']
    assert box['x'] == 12.0
    assert box['y'] == 12.0
    assert box['z'] == 12.0
    # Modifier 'infill' should nest under 'modifiers'
    assert 'modifiers' in action
    assert 'infill' in action['modifiers']
    infill = action['modifiers']['infill']
    assert infill['pattern'] == 'hex'
    assert infill['density'] == 0.5

def test_parse_boolean_union_spec():
    raw = '{"primitives":[{"shape":"sphere","size_mm":5},{"shape":"box","size_mm":5}],"boolean":"union"}'
    spec = parse_raw_spec(raw)
    assert isinstance(spec, list)
    assert len(spec) == 2
    for action in spec:
        assert 'primitive' in action

def test_parse_invalid_boolean_without_primitives_spec():
    raw = '{"shape":"sphere","size_mm":4,"boolean":"difference"}'
    spec = parse_raw_spec(raw)
    assert isinstance(spec, list)
    assert len(spec) == 1
    action = spec[0]
    assert 'primitive' in action
    sphere = action['primitive']['sphere']
    assert sphere['radius'] == 2.0


def test_generate_summary_sphere():
    raw = '{"shape":"sphere","size_mm":10}'
    spec = parse_raw_spec(raw)
    summary = generate_summary(spec)
    # summary should mention the sphere and its radius
    assert isinstance(summary, str)
    assert "sphere" in summary.lower()
    assert "5.0" in summary


def test_review_request_endpoint():
    raw = '{"shape":"box","size_mm":12,"infill":{"pattern":"hex","density":0.5}}'
    request_data = json.loads(raw)
    spec, summary = review_request(request_data)
    # endpoint should return the parsed spec and a text summary
    assert isinstance(spec, list)
    assert len(spec) == 1
    assert isinstance(summary, str)
    assert "box" in summary.lower()


def test_review_request_with_dict_input_sphere():
    # Direct dict input without invoking LLM
    request_data = {'shape': 'sphere', 'size_mm': 6}
    spec, summary = review_request(request_data)
    assert isinstance(spec, list)
    assert len(spec) == 1
    sphere = spec[0]['primitive']['sphere']
    assert sphere['radius'] == 3.0
    assert isinstance(summary, str)
    assert "sphere" in summary.lower()
    assert "3.0" in summary


def test_review_request_with_dict_input_multiple_primitives():
    request_data = {
        'primitives': [
            {'shape': 'sphere', 'size_mm': 8},
            {'shape': 'box', 'size_mm': 4}
        ]
    }
    spec, summary = review_request(request_data)
    assert isinstance(spec, list)
    assert len(spec) == 2
    # Check both primitives are present
    shapes = [list(action['primitive'].keys())[0] for action in spec]
    assert set(shapes) == {'sphere', 'box'}
    assert isinstance(summary, str)
    # summary should mention both shapes
    assert "sphere" in summary.lower()
    assert "box" in summary.lower()


# Additional tests for review_request with existing/modified spec
def test_review_request_with_existing_spec():
    # Simulate UI sending back the spec without parsing again
    raw = '{"shape":"sphere","size_mm":10}'
    initial_spec = parse_raw_spec(raw)
    request_data = {'spec': initial_spec}
    spec, summary = review_request(request_data)
    assert spec == initial_spec
    assert isinstance(summary, str)
    assert "sphere" in summary.lower()
    assert "5.0" in summary

def test_review_request_with_modified_spec():
    # Simulate user modifying an existing spec before review
    modified_spec = [{'primitive': {'sphere': {'radius': 4.5}}}]
    request_data = {'spec': modified_spec}
    spec, summary = review_request(request_data)
    assert spec == modified_spec
    assert isinstance(summary, str)
    assert "sphere" in summary.lower()
    assert "4.5" in summary


# Test update_request for adding infill to a sphere
def test_update_request_add_infill():
    raw = '{"shape":"sphere","size_mm":10}'
    initial_spec = parse_raw_spec(raw)
    request_data = {
        'sid': 'test-session',
        'spec': initial_spec,
        'raw': 'Please add a 2mm gyroid infill to the sphere'
    }
    spec, summary = update_request(request_data['sid'], request_data['spec'], request_data['raw'])
    # Should return a list of length 1
    assert isinstance(spec, list)
    assert len(spec) == 1
    action = spec[0]
    # There should be a nested 'infill' modifier
    assert 'modifiers' in action
    assert 'infill' in action['modifiers']
    infill = action['modifiers']['infill']
    assert isinstance(infill, dict)
    # Should have type or pattern mentioning gyroid
    pattern = infill.get('pattern') or infill.get('type')
    assert pattern is not None
    assert 'gyroid' in str(pattern).lower()
    # Should have a thickness or density that is numeric and positive
    thickness = infill.get('thickness') or infill.get('density')
    assert isinstance(thickness, (int, float))
    assert thickness > 0
    # Summary should mention gyroid
    assert isinstance(summary, str)
    assert 'gyroid' in summary.lower()


# Test update_request for union of multiple primitives
def test_update_request_multiple_primitives():
    raw = '{"primitives":[{"shape":"cube","size_mm":8},{"shape":"sphere","size_mm":6}]}'
    initial_spec = parse_raw_spec(raw)
    request_data = {
        'sid': 'test-session',
        'spec': initial_spec,
        'raw': 'Please union these two shapes'
    }
    spec, summary = update_request(request_data['sid'], request_data['spec'], request_data['raw'])
    # Should not add new primitives, just apply boolean operation
    assert isinstance(spec, list)
    assert len(spec) == 2
    # Each action should still have its primitive
    for action in spec:
        assert 'primitive' in action
        primitive = action['primitive']
        assert isinstance(primitive, dict)
        assert list(primitive.keys())[0] in ('cube', 'sphere')
    # Summary should mention union
    assert isinstance(summary, str)
    assert 'union' in summary.lower()


# Voronoi primitive and infill tests
def test_parse_voronoi_shape_spec():
    # Explicit Voronoi primitive spec
    raw = json.dumps({
        "shape": "voronoi",
        "min_dist": 0.3,
        "bbox_min": [0, 0, 0],
        "bbox_max": [1, 1, 1]
    })
    spec = parse_raw_spec(raw)
    assert isinstance(spec, list)
    assert len(spec) == 1
    primitive = spec[0].get('primitive', {})
    assert 'voronoi' in primitive
    vor = primitive['voronoi']
    assert vor['min_dist'] == 0.3
    assert vor['bbox_min'] == [0, 0, 0]
    assert vor['bbox_max'] == [1, 1, 1]


def test_review_request_with_voronoi_infill():
    # Voronoi infill on a box should transform into a standalone Voronoi primitive
    raw_data = {
        "shape": "box",
        "size_mm": 10,
        "infill": {"pattern": "voronoi", "density": 0.5}
    }
    spec, summary = review_request(raw_data)
    assert isinstance(spec, list) and len(spec) == 1
    action = spec[0]
    # Primitive remains the box
    assert 'primitive' in action
    box = action['primitive']['box']
    assert box['size']['x'] == 10.0
    # Infill modifier should be nested
    assert 'modifiers' in action
    assert 'infill' in action['modifiers']
    infill = action['modifiers']['infill']
    assert infill['pattern'] == 'voronoi'
    # Default lattice params should be present
    assert 'min_dist' in infill and isinstance(infill['min_dist'], float)
    assert 'bbox_min' in infill and 'bbox_max' in infill
    # Summary should mention voronoi
    assert isinstance(summary, str)
    assert 'voronoi' in summary.lower()

def test_update_request_seed_generation_count():
    # Ensure update_request caps seed points to MAX_SEED_POINTS
    raw = '{"shape":"sphere","size_mm":10}'
    initial_spec = parse_raw_spec(raw)
    request_data = {
        'sid': 'seed-test-session',
        'spec': initial_spec,
        'raw': 'Can you add a voronoi infill?'
    }
    spec, summary = update_request(request_data['sid'], request_data['spec'], request_data['raw'])
    assert isinstance(spec, list) and len(spec) == 1
    action = spec[0]
    assert 'modifiers' in action and 'infill' in action['modifiers']
    infill = action['modifiers']['infill']
    num_points = infill.get('num_points')
    assert isinstance(num_points, int)
    # Should be capped to MAX_SEED_POINTS
    assert num_points <= MAX_SEED_POINTS
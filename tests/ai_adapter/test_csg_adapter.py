# tests/ai_adapter/test_csg_adapter.py
import pytest
import json
from ai_adapter.csg_adapter import parse_raw_spec, generate_summary, review_request

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
    assert 'children' in action
    assert len(action['children']) == 1
    child = action['children'][0]
    assert 'primitive' in child
    box = child['primitive']['box']['size']
    assert box['x'] == 12.0
    assert box['y'] == 12.0
    assert box['z'] == 12.0
    assert 'infill' in action
    infill = action['infill']
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
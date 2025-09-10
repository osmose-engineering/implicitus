import pytest

try:
    from ai_adapter.csg_adapter import parse_raw_spec
except ImportError:  # pragma: no cover - skip if adapter unavailable
    pytest.skip("parse_raw_spec not available", allow_module_level=True)


def test_lattice_alias_parsed_as_infill():
    spec = {'shape': 'sphere', 'size_mm': 20, 'lattice': 'voronoi'}
    nodes = parse_raw_spec(spec)
    assert len(nodes) == 1
    node = nodes[0]
    assert node['primitive']['sphere']['radius'] == 10.0
    infill = node.get('modifiers', {}).get('infill', {})
    assert infill.get('pattern') == 'voronoi'


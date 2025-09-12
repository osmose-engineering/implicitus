# tests/conftest.py
import os
import sys
import types

from fastapi.testclient import TestClient
import pytest

# add the project root (one level up) onto sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Provide lightweight stubs for the adapter modules so importing the API
# does not require building the Rust core engine during tests.
stub_rust = types.ModuleType("ai_adapter.rust_primitives")
stub_rust.sample_inside = lambda *args, **kwargs: []
sys.modules["ai_adapter.rust_primitives"] = stub_rust

# Allow tests to import the real csg_adapter now that heavy dependencies are mocked

# Stub out infill generation helpers to avoid heavy dependencies.
stub_infill = types.ModuleType("design_api.services.infill_service")
stub_infill.generate_hex_lattice = lambda *args, **kwargs: {}
stub_infill.generate_voronoi = lambda *args, **kwargs: {}
stub_infill.build_hex_lattice = lambda *args, **kwargs: ([], [], [], [])
sys.modules["design_api.services.infill_service"] = stub_infill
import design_api.services as _services
_services.infill_service = stub_infill

# Minimal stubs for Voronoi helpers used by seed utilities.
stub_voro_core = types.ModuleType("design_api.services.voronoi_gen.voronoi_gen")
stub_voro_core.derive_bbox_from_primitive = lambda *args, **kwargs: ([0, 0, 0], [1, 1, 1])
stub_voro_core.build_hex_lattice = lambda *args, **kwargs: ([], [], [], [])
sys.modules["design_api.services.voronoi_gen.voronoi_gen"] = stub_voro_core
stub_voro_pkg = types.ModuleType("design_api.services.voronoi_gen")
stub_voro_pkg.voronoi_gen = stub_voro_core
sys.modules["design_api.services.voronoi_gen"] = stub_voro_pkg
_services.voronoi_gen = stub_voro_pkg

# Stub protobuf runtime version check expected by generated modules
stub_runtime_version = types.ModuleType("google.protobuf.runtime_version")
stub_runtime_version.Domain = types.SimpleNamespace(PUBLIC=0)
stub_runtime_version.ValidateProtobufRuntimeVersion = lambda *args, **kwargs: None
sys.modules["google.protobuf.runtime_version"] = stub_runtime_version

from design_api.main import app, models, design_states


@pytest.fixture
def client():
    """FastAPI test client that resets global state before each test."""
    models.clear()
    design_states.clear()
    return TestClient(app)


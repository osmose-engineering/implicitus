# tests/conftest.py
import os
import sys

from fastapi.testclient import TestClient
import pytest

# add the project root (one level up) onto sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from design_api.main import app, models, design_states


@pytest.fixture
def client():
    """FastAPI test client that resets global state before each test."""
    models.clear()
    design_states.clear()
    return TestClient(app)


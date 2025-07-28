import numpy as np
import pytest

class DummyMesh:
    def __init__(self, vertices):
        self.vertices = np.array(vertices)

@pytest.fixture
def square_mesh():
    # four corners of a unit square in XY plane
    return DummyMesh([[0,0,0], [1,0,0], [1,1,0], [0,1,0]])

@pytest.fixture
def simple_seeds():
    # center plus one offset
    return np.array([[0.5,0.5,0], [0.2,0.2,0]])
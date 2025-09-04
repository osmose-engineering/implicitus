from fastapi.testclient import TestClient
from design_api.main import app

def test_mesh_endpoint_returns_data():
    client = TestClient(app)
    seeds = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    payload = {
        'seed_points': seeds,
        'num_points': len(seeds),
        'mode': 'organic',
    }
    resp = client.post('/design/mesh', json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert 'vertices' in data and 'edges' in data
    assert len(data['vertices']) > 0

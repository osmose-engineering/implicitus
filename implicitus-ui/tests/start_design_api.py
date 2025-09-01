import os, sys, types, importlib.util, importlib.machinery, importlib.abc
# ensure project root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# stub transformers
transformers_stub = types.ModuleType('transformers')

def pipeline(*args, **kwargs):
    raise RuntimeError('pipeline should not be called in tests')

class AutoTokenizer:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

transformers_stub.pipeline = pipeline
transformers_stub.AutoTokenizer = AutoTokenizer
sys.modules['transformers'] = transformers_stub

# stub rust_primitives to avoid compiling the Rust extension during tests
rust_primitives_stub = types.ModuleType('ai_adapter.rust_primitives')
def sample_inside(*args, **kwargs):
    return []
rust_primitives_stub.sample_inside = sample_inside
sys.modules['ai_adapter.rust_primitives'] = rust_primitives_stub

# stub core_engine core module to avoid building Rust extensions
core_stub = types.ModuleType('core_engine.core_engine')
def prune_adjacency_via_grid(points, spacing):
    edges = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            edges.append((i, j))
    return edges
core_stub.prune_adjacency_via_grid = prune_adjacency_via_grid
core_stub.OctreeNode = object
def generate_adaptive_grid(*args, **kwargs):
    return None
core_stub.generate_adaptive_grid = generate_adaptive_grid
def compute_uniform_cells(*args, **kwargs):
    return [], {}
core_stub.compute_uniform_cells = compute_uniform_cells

class CoreLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return core_stub
    def exec_module(self, module):
        return

orig_find_spec = importlib.util.find_spec
def fake_find_spec(name, package=None):
    if name == 'core_engine.core_engine':
        return importlib.machinery.ModuleSpec(name, CoreLoader())
    return orig_find_spec(name, package)

importlib.util.find_spec = fake_find_spec
sys.modules['core_engine.core_engine'] = core_stub

from design_api.main import app
import uvicorn

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8001, log_level='warning')

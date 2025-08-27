import os, sys, types
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

from design_api.main import app
import uvicorn

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8001, log_level='warning')

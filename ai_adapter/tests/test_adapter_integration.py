

# ai_adapter/tests/test_adapter_integration.py

import pytest
from ai_adapter import adapter
from ai_adapter.schema import implicitus_pb2

@pytest.mark.slow
def test_real_model_outputs_json():
    """
    End-to-end integration test using the real Mistral 7B-Instruct model.
    This test is marked slow and should only run when GPU or sufficient resources are available.
    """
    # Invoke the live LLM path
    m = adapter.generate_model("create a sphere with radius 2.5", use_llm=True)

    # Verify it returned a Model protobuf
    assert isinstance(m, implicitus_pb2.Model)
    # Check that the model id is non-empty
    assert m.id, "Expected a non-empty model.id"
    # Verify the root primitive is a sphere with approximately the correct radius
    assert m.root.primitive.sphere.radius == pytest.approx(2.5, rel=1e-2)
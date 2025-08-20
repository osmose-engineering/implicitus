import pytest

from design_api.services import llm_service


def test_generate_design_spec_happy(monkeypatch):
    monkeypatch.setattr(
        llm_service,
        "generate",
        lambda prompt: '{"shape":"sphere","size_mm":10}',
    )
    result = llm_service.generate_design_spec("foo")
    assert isinstance(result, dict)
    assert "shape" in result and "size_mm" in result


def test_generate_design_spec_error(monkeypatch):
    def bad(prompt):
        raise RuntimeError("oom")

    monkeypatch.setattr(llm_service, "generate", bad)
    with pytest.raises(RuntimeError):
        llm_service.generate_design_spec("foo")
import pytest
import json

from design_api.services.json_cleaner import clean_llm_output

def test_strip_fences_and_labels():
    raw = 'JSON:\n```json\n{"foo":1}\n```'
    result = clean_llm_output(raw)
    # Validate JSON equivalence
    assert json.loads(result) == {"foo": 1}

def test_leading_trailing_noise():
    raw = '…\n{ "bar": 2 }\n…'
    result = clean_llm_output(raw)
    assert json.loads(result) == {"bar": 2}

def test_noop_on_clean():
    clean = '{"baz":3}'
    result = clean_llm_output(clean)
    assert json.loads(result) == json.loads(clean)
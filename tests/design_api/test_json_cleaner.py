import pytest

from design_api.services.json_cleaner import clean_llm_output

def test_strip_fences_and_labels():
    raw = 'JSON:\n```json\n{"foo":1}\n```'
    assert clean_llm_output(raw) == '{"foo":1}'

def test_leading_trailing_noise():
    raw = '…\n{ "bar": 2 }\n…'
    assert clean_llm_output(raw) == '{ "bar": 2 }'

def test_noop_on_clean():
    clean = '{"baz":3}'
    assert clean_llm_output(clean) == clean
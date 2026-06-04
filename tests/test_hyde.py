import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import MagicMock
from retrieval import hyde


@pytest.fixture(autouse=True)
def _clear():
    hyde.clear_cache()
    yield
    hyde.clear_cache()


def make_client(text="hypothetical answer"):
    c = MagicMock()
    c.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=text))]
    )
    return c


def test_generates_and_returns_answer():
    c = make_client("Transformers use attention.")
    assert hyde.hypothetical_answer("What is a transformer?", c) == "Transformers use attention."
    c.chat.completions.create.assert_called_once()


def test_same_question_cached_single_call():
    c = make_client()
    hyde.hypothetical_answer("Q1", c)
    hyde.hypothetical_answer("Q1", c)
    hyde.hypothetical_answer("Q1", c)
    assert c.chat.completions.create.call_count == 1


def test_different_questions_each_call():
    c = make_client()
    hyde.hypothetical_answer("Q1", c)
    hyde.hypothetical_answer("Q2", c)
    assert c.chat.completions.create.call_count == 2


def test_clear_cache_forces_regeneration():
    c = make_client()
    hyde.hypothetical_answer("Q1", c)
    hyde.clear_cache()
    hyde.hypothetical_answer("Q1", c)
    assert c.chat.completions.create.call_count == 2

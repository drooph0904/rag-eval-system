import json
from unittest.mock import MagicMock, patch
from multi_context_generator import MultiContextGenerator


def _section(section_id=1, text="This passage defines RAG as retrieval-augmented generation. "
             "Three pipelines were tested. Dense retrieval achieved 0.87 precision."):
    return {"section_id": section_id, "text": text, "word_count": len(text.split())}


def _openai_response(content: str):
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


VALID_PAYLOAD = json.dumps({"questions": [
    {
        "question": "How does the training time relate to the number of attention heads?",
        "ground_truth": "The base Transformer with 8 attention heads was trained for 12 hours on 8 GPUs.",
        "question_type": "multi_context",
        "source_sections": [1, 2],
    }
]})


def _make_generator():
    with patch("multi_context_generator.OpenAI") as MockOpenAI:
        client = MagicMock()
        MockOpenAI.return_value = client
        gen = MultiContextGenerator()
        gen._client = client
        return gen, client


def test_generate_returns_list():
    gen, client = _make_generator()
    client.chat.completions.create.return_value = _openai_response(VALID_PAYLOAD)
    result = gen.generate_for_pair(_section(1), _section(2))
    assert isinstance(result, list)


def test_generate_valid_question_passes_through():
    gen, client = _make_generator()
    client.chat.completions.create.return_value = _openai_response(VALID_PAYLOAD)
    result = gen.generate_for_pair(_section(1), _section(2))
    assert len(result) == 1
    assert result[0]["question_type"] == "multi_context"
    assert result[0]["source_sections"] == [1, 2]


def test_generate_drops_question_without_question_mark():
    payload = json.dumps({"questions": [{
        "question": "How does training time relate to attention heads",  # no ?
        "ground_truth": "The base Transformer with 8 attention heads was trained for 12 hours.",
        "question_type": "multi_context",
        "source_sections": [1, 2],
    }]})
    gen, client = _make_generator()
    client.chat.completions.create.return_value = _openai_response(payload)
    result = gen.generate_for_pair(_section(1), _section(2))
    assert result == []


def test_generate_drops_wrong_question_type():
    payload = json.dumps({"questions": [{
        "question": "What is the attention head count?",
        "ground_truth": "The model uses 8 attention heads running in parallel for multi-head attention.",
        "question_type": "factual",   # should be multi_context
        "source_sections": [1, 2],
    }]})
    gen, client = _make_generator()
    client.chat.completions.create.return_value = _openai_response(payload)
    result = gen.generate_for_pair(_section(1), _section(2))
    assert result == []


def test_generate_drops_short_ground_truth():
    payload = json.dumps({"questions": [{
        "question": "How does training relate to attention?",
        "ground_truth": "12 hours",   # too short
        "question_type": "multi_context",
        "source_sections": [1, 2],
    }]})
    gen, client = _make_generator()
    client.chat.completions.create.return_value = _openai_response(payload)
    result = gen.generate_for_pair(_section(1), _section(2))
    assert result == []


def test_generate_drops_visual_reference_in_answer():
    payload = json.dumps({"questions": [{
        "question": "How does training time relate to the model size?",
        "ground_truth": "As shown in the figure, the big model takes longer to train.",
        "question_type": "multi_context",
        "source_sections": [1, 2],
    }]})
    gen, client = _make_generator()
    client.chat.completions.create.return_value = _openai_response(payload)
    result = gen.generate_for_pair(_section(1), _section(2))
    assert result == []


def test_generate_fills_missing_source_sections():
    payload = json.dumps({"questions": [{
        "question": "How does training time relate to the number of encoder layers?",
        "ground_truth": "The 6-layer encoder Transformer was trained for 12 hours on 8 GPUs.",
        "question_type": "multi_context",
        "source_sections": [],   # empty — should be filled with [id_a, id_b]
    }]})
    gen, client = _make_generator()
    client.chat.completions.create.return_value = _openai_response(payload)
    result = gen.generate_for_pair(_section(3), _section(7))
    assert result[0]["source_sections"] == [3, 7]


def test_generate_retries_on_invalid_json():
    gen, client = _make_generator()
    client.chat.completions.create.return_value = _openai_response("not json")
    with patch("time.sleep"):
        result = gen.generate_for_pair(_section(1), _section(2))
    assert result == []
    assert client.chat.completions.create.call_count == 3

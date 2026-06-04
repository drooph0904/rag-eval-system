import json
import time
from unittest.mock import MagicMock, patch
from question_generator import QuestionGenerator


def _section(section_id=1, text="This passage defines RAG as retrieval-augmented generation. "
             "Three pipelines were tested. Dense retrieval achieved 0.87 precision. "
             "Sparse retrieval scored 0.72. The minimum chunk is 100 words."):
    return {"section_id": section_id, "text": text, "word_count": len(text.split()), "source_pages": "1"}


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
        "question": "What does RAG stand for?",
        "ground_truth": "RAG stands for retrieval-augmented generation as defined in the passage.",
        "question_type": "definition",
        "source_section": 1,
    }
]})


def _make_generator():
    with patch("question_generator.OpenAI") as MockOpenAI:
        client = MagicMock()
        MockOpenAI.return_value = client
        gen = QuestionGenerator()
        gen._client = client   # expose for assertions
        return gen, client


def test_generate_returns_list():
    gen, client = _make_generator()
    client.chat.completions.create.return_value = _openai_response(VALID_PAYLOAD)
    result = gen.generate(_section())
    assert isinstance(result, list)


def test_generate_valid_question_passes_through():
    gen, client = _make_generator()
    client.chat.completions.create.return_value = _openai_response(VALID_PAYLOAD)
    result = gen.generate(_section())
    assert len(result) == 1
    assert result[0]["question"] == "What does RAG stand for?"


def test_generate_drops_question_without_question_mark():
    payload = json.dumps({"questions": [{
        "question": "What does RAG stand for",  # no ?
        "ground_truth": "RAG stands for retrieval-augmented generation.",
        "question_type": "definition",
        "source_section": 1,
    }]})
    gen, client = _make_generator()
    client.chat.completions.create.return_value = _openai_response(payload)
    result = gen.generate(_section())
    assert result == []


def test_generate_drops_question_with_short_ground_truth():
    payload = json.dumps({"questions": [{
        "question": "What is X?",
        "ground_truth": "X is it",    # very short
        "question_type": "factual",
        "source_section": 1,
    }]})
    gen, client = _make_generator()
    client.chat.completions.create.return_value = _openai_response(payload)
    result = gen.generate(_section())
    assert result == []


def test_generate_keeps_short_numerical_answer():
    # "28.4 BLEU" is 9 chars — below MIN_ANSWER_LENGTH (15) but above numerical floor (8)
    payload = json.dumps({"questions": [{
        "question": "How many BLEU points did the Transformer achieve?",
        "ground_truth": "28.4 BLEU",
        "question_type": "numerical",
        "source_section": 1,
    }]})
    gen, client = _make_generator()
    client.chat.completions.create.return_value = _openai_response(payload)
    result = gen.generate(_section())
    assert len(result) == 1


def test_generate_drops_short_non_numerical_answer():
    # same short answer but factual type — should still be dropped
    payload = json.dumps({"questions": [{
        "question": "What score did the model achieve?",
        "ground_truth": "28.4 BLEU",
        "question_type": "factual",
        "source_section": 1,
    }]})
    gen, client = _make_generator()
    client.chat.completions.create.return_value = _openai_response(payload)
    result = gen.generate(_section())
    assert result == []


def test_generate_drops_non_string_ground_truth():
    payload = json.dumps({"questions": [{
        "question": "How many layers are there?",
        "ground_truth": 6,   # LLM returned an integer, not a string
        "question_type": "numerical",
        "source_section": 1,
    }]})
    gen, client = _make_generator()
    client.chat.completions.create.return_value = _openai_response(payload)
    result = gen.generate(_section())
    assert result == []


def test_generate_drops_answer_over_max_length():
    long_answer = "word " * 100   # 500 chars+
    payload = json.dumps({"questions": [{
        "question": "What are all the compulsory registrable documents?",
        "ground_truth": long_answer,
        "question_type": "factual",
        "source_section": 1,
    }]})
    gen, client = _make_generator()
    client.chat.completions.create.return_value = _openai_response(payload)
    result = gen.generate(_section())
    assert result == []


def test_generate_drops_circular_answer_with_punctuation():
    # Apostrophes in "Sterility" and "Infertility" should not prevent detection
    payload = json.dumps({"questions": [{
        "question": "What is the difference between 'Sterility' and 'Infertility'?",
        "ground_truth": "Sterility and Infertility.",
        "question_type": "comparison",
        "source_section": 1,
    }]})
    gen, client = _make_generator()
    client.chat.completions.create.return_value = _openai_response(payload)
    result = gen.generate(_section())
    assert result == []


def test_generate_drops_author_perspective_questions():
    payload = json.dumps({"questions": [{
        "question": "What dataset did we train on?",
        "ground_truth": "We trained on the WMT 2014 English-German dataset for all experiments.",
        "question_type": "factual",
        "source_section": 1,
    }]})
    gen, client = _make_generator()
    client.chat.completions.create.return_value = _openai_response(payload)
    result = gen.generate(_section())
    assert result == []


def test_generate_drops_visual_reference_answers():
    payload = json.dumps({"questions": [{
        "question": "What does the diagram show?",
        "ground_truth": "The figure shows the encoder-decoder attention mechanism in detail.",
        "question_type": "factual",
        "source_section": 1,
    }]})
    gen, client = _make_generator()
    client.chat.completions.create.return_value = _openai_response(payload)
    result = gen.generate(_section())
    assert result == []


def test_generate_drops_circular_answers():
    payload = json.dumps({"questions": [{
        "question": "What is a positional embedding?",
        "ground_truth": "positional embedding instead of sinusoids",
        "question_type": "definition",
        "source_section": 1,
    }]})
    gen, client = _make_generator()
    client.chat.completions.create.return_value = _openai_response(payload)
    result = gen.generate(_section())
    assert result == []


def test_generate_keeps_non_circular_answer():
    payload = json.dumps({"questions": [{
        "question": "What is self-attention?",
        "ground_truth": "A mechanism that relates different positions of a single sequence to compute its representation.",
        "question_type": "definition",
        "source_section": 1,
    }]})
    gen, client = _make_generator()
    client.chat.completions.create.return_value = _openai_response(payload)
    result = gen.generate(_section())
    assert len(result) == 1


def test_generate_drops_question_with_invalid_type():
    payload = json.dumps({"questions": [{
        "question": "What is X?",
        "ground_truth": "X is the primary component in the system described.",
        "question_type": "tricky",    # not a valid type
        "source_section": 1,
    }]})
    gen, client = _make_generator()
    client.chat.completions.create.return_value = _openai_response(payload)
    result = gen.generate(_section())
    assert result == []


def test_generate_retries_on_invalid_json_and_returns_empty():
    gen, client = _make_generator()
    client.chat.completions.create.return_value = _openai_response("not json at all")
    with patch("time.sleep"):   # skip actual sleep in tests
        result = gen.generate(_section())
    assert result == []
    assert client.chat.completions.create.call_count == 3   # MAX_RETRIES


def test_generate_succeeds_on_second_attempt():
    gen, client = _make_generator()
    client.chat.completions.create.side_effect = [
        _openai_response("bad json"),
        _openai_response(VALID_PAYLOAD),
    ]
    with patch("time.sleep"):
        result = gen.generate(_section())
    assert len(result) == 1
    assert client.chat.completions.create.call_count == 2

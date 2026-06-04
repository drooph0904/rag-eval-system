import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from unittest.mock import MagicMock
from evaluation.stage2_eval import Stage2Evaluator


def make_chunks(n=3):
    return [{"chunk_id": f"fixed_{i:03d}", "text": f"Context sentence {i} about the topic.", "score": 0.8} for i in range(n)]


def make_openai_client(answer="The answer is 42.", judge_score="80"):
    """
    Stage 2 makes two OpenAI calls per question:
      1. answer generation  -> returns `answer`
      2. correctness judge   -> returns `judge_score` (integer 0-100 as text)
    side_effect returns them in that order.
    """
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        MagicMock(choices=[MagicMock(message=MagicMock(content=answer))]),
        MagicMock(choices=[MagicMock(message=MagicMock(content=judge_score))]),
    ]
    return client


def make_embedder(similarity=0.9):
    """
    Embedder mock where embed_one(answer) and embed_one(ground_truth) produce
    unit vectors whose cosine == `similarity`.
    """
    embedder = MagicMock()
    a = np.zeros(384, dtype=np.float32)
    b = np.zeros(384, dtype=np.float32)
    a[0] = 1.0
    b[0] = similarity
    b[1] = float(np.sqrt(max(0.0, 1.0 - similarity ** 2)))
    embedder.embed_one.side_effect = lambda text: a if text == "GENANS" else b
    return embedder


class TestStage2Evaluator:
    def test_evaluate_makes_two_openai_calls(self):
        ev = Stage2Evaluator()
        client = make_openai_client(answer="GENANS")
        embedder = make_embedder()
        ev.evaluate("Q?", "ground truth", make_chunks(), client, embedder)
        assert client.chat.completions.create.call_count == 2

    def test_evaluate_returns_required_keys(self):
        ev = Stage2Evaluator()
        client = make_openai_client(answer="GENANS")
        embedder = make_embedder()
        result = ev.evaluate("Q?", "ground truth", make_chunks(), client, embedder)
        assert {"question", "generated_answer", "ground_truth",
                "answer_similarity", "answer_correctness"}.issubset(result.keys())

    def test_answer_correctness_parsed_and_normalized(self):
        ev = Stage2Evaluator()
        client = make_openai_client(answer="GENANS", judge_score="80")
        embedder = make_embedder()
        result = ev.evaluate("Q?", "ground truth", make_chunks(), client, embedder)
        assert abs(result["answer_correctness"] - 0.80) < 1e-6

    def test_answer_correctness_handles_noisy_judge_output(self):
        ev = Stage2Evaluator()
        client = make_openai_client(answer="GENANS", judge_score="Score: 75/100")
        embedder = make_embedder()
        result = ev.evaluate("Q?", "ground truth", make_chunks(), client, embedder)
        assert abs(result["answer_correctness"] - 0.75) < 1e-6

    def test_answer_correctness_clamped_to_unit_range(self):
        ev = Stage2Evaluator()
        client = make_openai_client(answer="GENANS", judge_score="130")
        embedder = make_embedder()
        result = ev.evaluate("Q?", "ground truth", make_chunks(), client, embedder)
        assert result["answer_correctness"] == 1.0

    def test_answer_similarity_is_cosine_clamped(self):
        ev = Stage2Evaluator()
        client = make_openai_client(answer="GENANS")
        embedder = make_embedder(similarity=0.6)
        result = ev.evaluate("Q?", "ground truth", make_chunks(), client, embedder)
        assert abs(result["answer_similarity"] - 0.6) < 1e-5

    def test_answer_similarity_negative_cosine_clamped_to_zero(self):
        ev = Stage2Evaluator()
        client = make_openai_client(answer="GENANS")
        embedder = make_embedder(similarity=-0.3)
        result = ev.evaluate("Q?", "ground truth", make_chunks(), client, embedder)
        assert result["answer_similarity"] == 0.0

    def test_evaluate_returns_none_on_exception(self):
        ev = Stage2Evaluator()
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("API error")
        embedder = make_embedder()
        result = ev.evaluate("Q?", "ground truth", make_chunks(), client, embedder)
        assert result is None

    def test_aggregate_returns_means_skipping_none(self):
        ev = Stage2Evaluator()
        results = [
            {"question": "Q1", "generated_answer": "A", "ground_truth": "G",
             "answer_similarity": 0.8, "answer_correctness": 0.9},
            None,
            {"question": "Q3", "generated_answer": "A", "ground_truth": "G",
             "answer_similarity": 0.4, "answer_correctness": 0.5},
        ]
        agg = ev.aggregate(results)
        assert abs(agg["mean_answer_similarity"] - 0.6) < 1e-6
        assert abs(agg["mean_answer_correctness"] - 0.7) < 1e-6

    def test_aggregate_all_none_returns_zeros(self):
        ev = Stage2Evaluator()
        agg = ev.aggregate([None, None])
        assert agg["mean_answer_similarity"] == 0.0
        assert agg["mean_answer_correctness"] == 0.0

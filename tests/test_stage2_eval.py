import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import MagicMock, patch
from evaluation.stage2_eval import Stage2Evaluator


def make_chunks(n=3):
    return [{"chunk_id": f"fixed_{i:03d}", "text": f"Context sentence {i} about the topic.", "score": 0.8} for i in range(n)]


def make_openai_client(answer="The answer is 42."):
    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=answer))]
    )
    return client


class TestStage2Evaluator:
    def test_evaluate_calls_openai_for_answer(self):
        ev = Stage2Evaluator()
        client = make_openai_client()
        with patch("evaluation.stage2_eval.evaluate_with_ragas", return_value={"faithfulness": 0.9, "answer_relevancy": 0.85, "context_precision": 0.8, "context_recall": 0.75}):
            ev.evaluate("Q?", "ground truth", make_chunks(), client)
        client.chat.completions.create.assert_called_once()

    def test_evaluate_returns_dict_with_required_keys(self):
        ev = Stage2Evaluator()
        client = make_openai_client()
        with patch("evaluation.stage2_eval.evaluate_with_ragas", return_value={"faithfulness": 0.9, "answer_relevancy": 0.85, "context_precision": 0.8, "context_recall": 0.75}):
            result = ev.evaluate("Q?", "ground truth", make_chunks(), client)
        assert {"question", "generated_answer", "ground_truth", "faithfulness", "answer_relevancy", "context_precision", "context_recall"}.issubset(result.keys())

    def test_evaluate_returns_none_on_exception(self):
        ev = Stage2Evaluator()
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("API error")
        result = ev.evaluate("Q?", "ground truth", make_chunks(), client)
        assert result is None

    def test_aggregate_returns_means_skipping_none(self):
        ev = Stage2Evaluator()
        results = [
            {"question": "Q1", "generated_answer": "A", "ground_truth": "G", "faithfulness": 0.8, "answer_relevancy": 0.7, "context_precision": 0.6, "context_recall": 0.9},
            None,
            {"question": "Q3", "generated_answer": "A", "ground_truth": "G", "faithfulness": 0.4, "answer_relevancy": 0.5, "context_precision": 0.6, "context_recall": 0.7},
        ]
        agg = ev.aggregate(results)
        assert abs(agg["mean_faithfulness"] - 0.6) < 1e-6
        assert abs(agg["mean_answer_relevancy"] - 0.6) < 1e-6

    def test_aggregate_all_none_returns_zeros(self):
        ev = Stage2Evaluator()
        agg = ev.aggregate([None, None])
        assert agg["mean_faithfulness"] == 0.0

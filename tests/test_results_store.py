# rag_eval_phase2/tests/test_results_store.py
import sys, os, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from evaluation.results_store import ResultsStore


def make_stage1_results():
    return {
        "fixed_pipeline_1": {"mean_context_precision": 0.61, "mean_context_recall": 0.72, "per_question": []},
        "semantic_pipeline_2": {"mean_context_precision": 0.80, "mean_context_recall": 0.85, "per_question": []},
        "parent_child_pipeline_3": {"mean_context_precision": 0.55, "mean_context_recall": 0.60, "per_question": []},
    }


def make_stage2_results():
    return {
        "semantic_pipeline_2": {"mean_faithfulness": 0.91, "mean_answer_relevancy": 0.88, "mean_context_precision": 0.87, "mean_context_recall": 0.83, "per_question": []},
        "fixed_pipeline_1": {"mean_faithfulness": 0.75, "mean_answer_relevancy": 0.70, "mean_context_precision": 0.68, "mean_context_recall": 0.65, "per_question": []},
        "parent_child_pipeline_3": {"mean_faithfulness": 0.82, "mean_answer_relevancy": 0.80, "mean_context_precision": 0.78, "mean_context_recall": 0.76, "per_question": []},
    }


class TestResultsStore:
    def test_save_creates_json_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ResultsStore(results_dir=tmpdir)
            store.save("doc.pdf", "./golden.json", make_stage1_results(), make_stage2_results(), total_questions=50)
            files = os.listdir(tmpdir)
            assert any(f.endswith("_eval_results.json") for f in files)

    def test_saved_json_has_required_top_level_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ResultsStore(results_dir=tmpdir)
            store.save("doc.pdf", "./golden.json", make_stage1_results(), make_stage2_results(), total_questions=50)
            files = [f for f in os.listdir(tmpdir) if f.endswith(".json")]
            with open(os.path.join(tmpdir, files[0])) as f:
                data = json.load(f)
            assert {"metadata", "stage1_results", "stage2_results", "winner"}.issubset(data.keys())

    def test_winner_has_highest_faithfulness(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ResultsStore(results_dir=tmpdir)
            store.save("doc.pdf", "./golden.json", make_stage1_results(), make_stage2_results(), total_questions=50)
            files = [f for f in os.listdir(tmpdir) if f.endswith(".json")]
            with open(os.path.join(tmpdir, files[0])) as f:
                data = json.load(f)
            assert data["winner"]["combination"] == "semantic_pipeline_2"

    def test_load_returns_dict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ResultsStore(results_dir=tmpdir)
            store.save("doc.pdf", "./golden.json", make_stage1_results(), make_stage2_results(), total_questions=50)
            files = [f for f in os.listdir(tmpdir) if f.endswith(".json")]
            path = os.path.join(tmpdir, files[0])
            loaded = store.load(path)
            assert isinstance(loaded, dict)
            assert "winner" in loaded

    def test_metadata_contains_pdf_name_and_question_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ResultsStore(results_dir=tmpdir)
            store.save("my_doc.pdf", "./golden.json", make_stage1_results(), make_stage2_results(), total_questions=42)
            files = [f for f in os.listdir(tmpdir) if f.endswith(".json")]
            with open(os.path.join(tmpdir, files[0])) as f:
                data = json.load(f)
            assert data["metadata"]["pdf_name"] == "my_doc.pdf"
            assert data["metadata"]["total_questions"] == 42

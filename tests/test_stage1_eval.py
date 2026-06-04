# rag_eval_phase2/tests/test_stage1_eval.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from unittest.mock import MagicMock
from evaluation.stage1_eval import Stage1Evaluator
from config import STAGE1_SIMILARITY_THRESHOLD


def make_embedder(ground_truth_vec=None, chunk_vecs=None):
    """Embedder mock: embed_one returns ground_truth_vec; embed returns chunk_vecs."""
    mock = MagicMock()
    if ground_truth_vec is None:
        ground_truth_vec = np.ones(384, dtype=np.float32)
        ground_truth_vec /= np.linalg.norm(ground_truth_vec)
    mock.embed_one.return_value = ground_truth_vec
    if chunk_vecs is None:
        chunk_vecs = np.tile(ground_truth_vec, (3, 1))  # 3 identical vecs = cosine sim 1.0
    mock.embed.return_value = chunk_vecs
    return mock


def make_chunks(n=3):
    return [{"chunk_id": f"fixed_{i:03d}", "text": f"chunk {i}", "score": 0.8} for i in range(n)]


def make_lexical_embedder(n_chunks):
    """Embedder whose embeddings give ZERO cosine (orthogonal), so only the lexical
    path can mark a chunk relevant."""
    mock = MagicMock()
    gt = np.zeros(384, dtype=np.float32); gt[0] = 1.0
    cv = np.zeros((n_chunks, 384), dtype=np.float32); cv[:, 1] = 1.0
    mock.embed_one.return_value = gt
    mock.embed.return_value = cv
    return mock


def chunk(text):
    return {"chunk_id": "c", "text": text, "score": 0.1}


class TestStage1Lexical:
    def test_chunk_containing_answer_is_relevant_despite_low_embedding(self):
        ev = Stage1Evaluator()
        chunks = [chunk("Education: Maharaja Agrasen Institute of Technology (MAIT), Delhi, B.Tech CSE")]
        embedder = make_lexical_embedder(len(chunks))
        result = ev.evaluate("Where does he study?", "Maharaja Agrasen Institute of Technology", chunks, embedder)
        assert result["context_recall"] == 1.0
        assert result["relevant_chunks_found"] == 1

    def test_numeric_answer_present_in_chunk_is_relevant(self):
        ev = Stage1Evaluator()
        chunks = [chunk("The QA Jira CLI Tool is used daily by 5 engineers across the team.")]
        embedder = make_lexical_embedder(len(chunks))
        result = ev.evaluate("How many engineers?", "5 engineers", chunks, embedder)
        assert result["relevant_chunks_found"] == 1

    def test_unrelated_chunk_not_relevant(self):
        ev = Stage1Evaluator()
        chunks = [chunk("He enjoys hiking and photography on weekends.")]
        embedder = make_lexical_embedder(len(chunks))
        result = ev.evaluate("Where does he study?", "Maharaja Agrasen Institute of Technology", chunks, embedder)
        assert result["relevant_chunks_found"] == 0
        assert result["context_recall"] == 0.0

    def test_partial_overlap_below_threshold_not_relevant(self):
        ev = Stage1Evaluator()
        # only "institute" overlaps out of 5 significant answer tokens -> 0.2 < 0.6
        chunks = [chunk("He works at a research institute downtown.")]
        embedder = make_lexical_embedder(len(chunks))
        result = ev.evaluate("Where?", "Maharaja Agrasen Institute of Technology Delhi", chunks, embedder)
        assert result["relevant_chunks_found"] == 0


class TestStage1Evaluator:
    def test_evaluate_returns_dict(self):
        ev = Stage1Evaluator()
        embedder = make_embedder()
        result = ev.evaluate("Q?", "answer", make_chunks(), embedder)
        assert isinstance(result, dict)

    def test_evaluate_has_required_keys(self):
        ev = Stage1Evaluator()
        embedder = make_embedder()
        result = ev.evaluate("Q?", "answer", make_chunks(), embedder)
        assert {"question", "context_precision", "context_recall", "relevant_chunks_found", "total_chunks_retrieved"}.issubset(result.keys())

    def test_precision_is_float_between_0_and_1(self):
        ev = Stage1Evaluator()
        embedder = make_embedder()
        result = ev.evaluate("Q?", "answer", make_chunks(), embedder)
        assert 0.0 <= result["context_precision"] <= 1.0

    def test_recall_is_0_or_1(self):
        ev = Stage1Evaluator()
        embedder = make_embedder()
        result = ev.evaluate("Q?", "answer", make_chunks(), embedder)
        assert result["context_recall"] in (0.0, 1.0)

    def test_all_relevant_chunks_gives_precision_1(self):
        ev = Stage1Evaluator()
        # ground truth vec identical to chunk vecs → cosine sim = 1.0 > threshold
        embedder = make_embedder()
        result = ev.evaluate("Q?", "answer", make_chunks(3), embedder)
        assert result["context_precision"] == 1.0

    def test_no_relevant_chunks_gives_precision_0(self):
        ev = Stage1Evaluator()
        gt_vec = np.zeros(384, dtype=np.float32)
        gt_vec[0] = 1.0
        # chunk vecs orthogonal to gt_vec → cosine sim = 0.0 < threshold
        chunk_vecs = np.zeros((3, 384), dtype=np.float32)
        chunk_vecs[:, 1] = 1.0
        embedder = make_embedder(gt_vec, chunk_vecs)
        result = ev.evaluate("Q?", "answer", make_chunks(3), embedder)
        assert result["context_precision"] == 0.0
        assert result["context_recall"] == 0.0

    def test_recall_is_1_when_any_chunk_relevant(self):
        ev = Stage1Evaluator()
        # first chunk identical to gt, others orthogonal
        gt_vec = np.zeros(384, dtype=np.float32)
        gt_vec[0] = 1.0
        chunk_vecs = np.zeros((3, 384), dtype=np.float32)
        chunk_vecs[0, 0] = 1.0   # relevant
        chunk_vecs[1, 1] = 1.0   # not relevant
        chunk_vecs[2, 2] = 1.0   # not relevant
        embedder = make_embedder(gt_vec, chunk_vecs)
        result = ev.evaluate("Q?", "answer", make_chunks(3), embedder)
        assert result["context_recall"] == 1.0

    def test_aggregate_returns_means(self):
        ev = Stage1Evaluator()
        per_q = [
            {"question": "Q1", "context_precision": 0.5, "context_recall": 1.0, "relevant_chunks_found": 2, "total_chunks_retrieved": 4},
            {"question": "Q2", "context_precision": 1.0, "context_recall": 0.0, "relevant_chunks_found": 0, "total_chunks_retrieved": 4},
        ]
        agg = ev.aggregate(per_q)
        assert abs(agg["mean_context_precision"] - 0.75) < 1e-6
        assert abs(agg["mean_context_recall"] - 0.5) < 1e-6

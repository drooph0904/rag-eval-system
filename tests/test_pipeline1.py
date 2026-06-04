# tests/test_pipeline1.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from unittest.mock import MagicMock
from retrieval.pipeline1 import Pipeline1
from config import TOP_K_FINAL, TOP_K_RETRIEVAL


def make_mock_store(n=5):
    store = MagicMock()
    store.search.return_value = [
        {"chunk_id": f"fixed_{i:03d}", "text": f"chunk text {i}", "score": 1.0 - i * 0.1}
        for i in range(n)
    ]
    return store


def make_mock_embedder():
    embedder = MagicMock()
    vec = np.random.randn(384).astype(np.float32)
    vec /= np.linalg.norm(vec)
    embedder.embed_one.return_value = vec
    return embedder


class TestPipeline1:
    def test_retrieve_returns_list(self):
        p = Pipeline1()
        store = make_mock_store()
        embedder = make_mock_embedder()
        result = p.retrieve("What is mitochondria?", store, embedder)
        assert isinstance(result, list)

    def test_retrieve_calls_embed_one(self):
        p = Pipeline1()
        store = make_mock_store()
        embedder = make_mock_embedder()
        p.retrieve("What is mitochondria?", store, embedder)
        embedder.embed_one.assert_called_once_with("What is mitochondria?")

    def test_retrieve_calls_store_search_with_top_k_final(self):
        p = Pipeline1()
        store = make_mock_store()
        embedder = make_mock_embedder()
        p.retrieve("What is mitochondria?", store, embedder)
        call_args = store.search.call_args
        assert call_args[1]["k"] == TOP_K_FINAL or call_args[0][1] == TOP_K_FINAL

    def test_retrieve_result_has_chunk_id_text_score(self):
        p = Pipeline1()
        store = make_mock_store()
        embedder = make_mock_embedder()
        results = p.retrieve("test?", store, embedder)
        assert len(results) > 0
        assert "chunk_id" in results[0]
        assert "text" in results[0]
        assert "score" in results[0]

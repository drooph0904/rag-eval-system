# tests/test_pipeline3.py
# Self-contained tests for Pipeline3 (HyDE + top-K + Cohere Rerank)
# Uses cohere 7.0.3 ClientV2 API shapes: response.results items have .index and .relevance_score

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from unittest.mock import MagicMock

from retrieval.pipeline3 import Pipeline3
from config import TOP_K_FINAL, TOP_K_RETRIEVAL


# ---------------------------------------------------------------------------
# Shared helpers (copied from plan's Task 5 helpers)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# TestPipeline3
# ---------------------------------------------------------------------------

class TestPipeline3:
    def _make_openai_client(self, response_text="A hypothetical answer."):
        client = MagicMock()
        client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=response_text))]
        )
        return client

    def _make_cohere_client(self, top_n=5):
        """
        Mock for cohere 7.0.3 ClientV2.rerank response.
        response.results is a list of objects with .index (int) and .relevance_score (float).
        """
        cohere_client = MagicMock()
        cohere_client.rerank.return_value = MagicMock(
            results=[
                MagicMock(index=i, relevance_score=1.0 - i * 0.1)
                for i in range(top_n)
            ]
        )
        return cohere_client

    def test_retrieve_returns_list(self):
        p = Pipeline3()
        store = make_mock_store(10)
        embedder = make_mock_embedder()
        client = self._make_openai_client()
        cohere_client = self._make_cohere_client()
        result = p.retrieve("What is RAG?", store, embedder, client, cohere_client)
        assert isinstance(result, list)

    def test_retrieve_fetches_top_k_retrieval_from_faiss(self):
        p = Pipeline3()
        store = make_mock_store(10)
        embedder = make_mock_embedder()
        client = self._make_openai_client()
        cohere_client = self._make_cohere_client()
        p.retrieve("What is RAG?", store, embedder, client, cohere_client)
        call_args = store.search.call_args
        k_used = call_args[1].get("k") if call_args[1] else call_args[0][1]
        assert k_used == TOP_K_RETRIEVAL

    def test_retrieve_calls_cohere_rerank(self):
        p = Pipeline3()
        store = make_mock_store(10)
        embedder = make_mock_embedder()
        client = self._make_openai_client()
        cohere_client = self._make_cohere_client()
        p.retrieve("What is RAG?", store, embedder, client, cohere_client)
        cohere_client.rerank.assert_called_once()

    def test_cohere_rerank_query_is_original_question(self):
        p = Pipeline3()
        store = make_mock_store(10)
        embedder = make_mock_embedder()
        client = self._make_openai_client()
        cohere_client = self._make_cohere_client()
        p.retrieve("What is RAG?", store, embedder, client, cohere_client)
        call_kwargs = cohere_client.rerank.call_args[1]
        assert call_kwargs["query"] == "What is RAG?"

    def test_cohere_rerank_uses_correct_model(self):
        from config import COHERE_RERANK_MODEL
        p = Pipeline3()
        store = make_mock_store(10)
        embedder = make_mock_embedder()
        client = self._make_openai_client()
        cohere_client = self._make_cohere_client()
        p.retrieve("What is RAG?", store, embedder, client, cohere_client)
        call_kwargs = cohere_client.rerank.call_args[1]
        assert call_kwargs["model"] == COHERE_RERANK_MODEL

    def test_cohere_rerank_top_n_is_top_k_final(self):
        p = Pipeline3()
        store = make_mock_store(10)
        embedder = make_mock_embedder()
        client = self._make_openai_client()
        cohere_client = self._make_cohere_client()
        p.retrieve("What is RAG?", store, embedder, client, cohere_client)
        call_kwargs = cohere_client.rerank.call_args[1]
        assert call_kwargs["top_n"] == TOP_K_FINAL

    def test_cohere_rerank_documents_are_candidate_texts(self):
        p = Pipeline3()
        store = make_mock_store(10)
        embedder = make_mock_embedder()
        client = self._make_openai_client()
        cohere_client = self._make_cohere_client()
        p.retrieve("What is RAG?", store, embedder, client, cohere_client)
        call_kwargs = cohere_client.rerank.call_args[1]
        expected_docs = [f"chunk text {i}" for i in range(10)]
        assert call_kwargs["documents"] == expected_docs

    def test_result_has_correct_keys(self):
        p = Pipeline3()
        store = make_mock_store(10)
        embedder = make_mock_embedder()
        client = self._make_openai_client()
        cohere_client = self._make_cohere_client()
        results = p.retrieve("What is RAG?", store, embedder, client, cohere_client)
        assert len(results) > 0
        assert {"chunk_id", "text", "score"}.issubset(results[0].keys())

    def test_result_score_comes_from_cohere_relevance_score(self):
        p = Pipeline3()
        store = make_mock_store(10)
        embedder = make_mock_embedder()
        client = self._make_openai_client()
        cohere_client = self._make_cohere_client(top_n=5)
        results = p.retrieve("What is RAG?", store, embedder, client, cohere_client)
        # First result index=0, relevance_score=1.0
        assert results[0]["score"] == pytest.approx(1.0)

    def test_result_length_is_top_k_final(self):
        p = Pipeline3()
        store = make_mock_store(10)
        embedder = make_mock_embedder()
        client = self._make_openai_client()
        cohere_client = self._make_cohere_client(top_n=TOP_K_FINAL)
        results = p.retrieve("What is RAG?", store, embedder, client, cohere_client)
        assert len(results) == TOP_K_FINAL

    def test_hyde_embeds_hypothetical_answer_not_question(self):
        hyp_answer = "RAG stands for Retrieval-Augmented Generation."
        p = Pipeline3()
        store = make_mock_store(10)
        embedder = make_mock_embedder()
        client = self._make_openai_client(response_text=hyp_answer)
        cohere_client = self._make_cohere_client()
        p.retrieve("What is RAG?", store, embedder, client, cohere_client)
        embedder.embed_one.assert_called_once_with(hyp_answer)

    def test_fallback_to_question_embedding_on_hyde_failure(self):
        p = Pipeline3()
        store = make_mock_store(10)
        embedder = make_mock_embedder()
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("OpenAI API error")
        cohere_client = self._make_cohere_client()
        result = p.retrieve("What is RAG?", store, embedder, client, cohere_client)
        # Falls back to embedding the original question
        embedder.embed_one.assert_called_once_with("What is RAG?")
        assert isinstance(result, list)

    def test_fallback_to_faiss_results_when_cohere_fails(self):
        p = Pipeline3()
        store = make_mock_store(10)
        embedder = make_mock_embedder()
        client = self._make_openai_client()
        cohere_client = MagicMock()
        cohere_client.rerank.side_effect = Exception("Cohere API error")
        result = p.retrieve("What is RAG?", store, embedder, client, cohere_client)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_fallback_cohere_returns_top_k_final_faiss_results(self):
        p = Pipeline3()
        store = make_mock_store(10)
        embedder = make_mock_embedder()
        client = self._make_openai_client()
        cohere_client = MagicMock()
        cohere_client.rerank.side_effect = Exception("Cohere API error")
        result = p.retrieve("What is RAG?", store, embedder, client, cohere_client)
        # Fallback: top TOP_K_FINAL from FAISS candidates
        assert len(result) == TOP_K_FINAL

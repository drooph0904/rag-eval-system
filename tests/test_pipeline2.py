import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from unittest.mock import MagicMock
from retrieval.pipeline2 import Pipeline2


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


class TestPipeline2:
    def _make_openai_client(self, response_text="A hypothetical answer about the topic."):
        client = MagicMock()
        client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=response_text))]
        )
        return client

    def test_retrieve_returns_list(self):
        p = Pipeline2()
        store = make_mock_store()
        embedder = make_mock_embedder()
        client = self._make_openai_client()
        result = p.retrieve("What is mitochondria?", store, embedder, client)
        assert isinstance(result, list)

    def test_retrieve_calls_openai(self):
        p = Pipeline2()
        store = make_mock_store()
        embedder = make_mock_embedder()
        client = self._make_openai_client()
        p.retrieve("What is mitochondria?", store, embedder, client)
        client.chat.completions.create.assert_called_once()

    def test_retrieve_embeds_hypothetical_answer_not_question(self):
        hyp_answer = "Mitochondria produce ATP through oxidative phosphorylation."
        p = Pipeline2()
        store = make_mock_store()
        embedder = make_mock_embedder()
        client = self._make_openai_client(response_text=hyp_answer)
        p.retrieve("What is mitochondria?", store, embedder, client)
        embedder.embed_one.assert_called_once_with(hyp_answer)

    def test_fallback_to_question_embedding_on_llm_failure(self):
        p = Pipeline2()
        store = make_mock_store()
        embedder = make_mock_embedder()
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("API error")
        result = p.retrieve("What is mitochondria?", store, embedder, client)
        # should fall back to embedding the original question
        embedder.embed_one.assert_called_once_with("What is mitochondria?")
        assert isinstance(result, list)

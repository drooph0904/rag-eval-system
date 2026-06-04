# rag_eval_phase2/tests/test_faiss_store.py
import sys, os, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from unittest.mock import MagicMock
from indexer.faiss_store import FAISSStore


def make_mock_embedder(n: int, dim: int = 384):
    """Return an Embedder mock that returns normalized random vectors."""
    mock = MagicMock()
    vecs = np.random.randn(n, dim).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    mock.embed.return_value = vecs
    return mock, vecs


def make_chunks(n: int, strategy: str = "fixed") -> list[dict]:
    return [
        {"chunk_id": f"{strategy}_{i:03d}", "text": f"sentence about topic {i}", "token_count": 50, "strategy": strategy, "char_start": i * 100}
        for i in range(n)
    ]


@pytest.fixture
def store_with_data():
    chunks = make_chunks(10)
    mock_embedder, _ = make_mock_embedder(10)
    store = FAISSStore("fixed")
    store.build(chunks, mock_embedder)
    return store, chunks


def test_build_sets_index(store_with_data):
    store, _ = store_with_data
    assert store.index is not None

def test_build_stores_chunk_texts(store_with_data):
    store, chunks = store_with_data
    assert len(store.chunk_texts) == len(chunks)

def test_build_stores_chunk_ids(store_with_data):
    store, chunks = store_with_data
    assert store.chunk_ids == [c["chunk_id"] for c in chunks]

def test_search_returns_list(store_with_data):
    store, _ = store_with_data
    query = np.random.randn(384).astype(np.float32)
    query /= np.linalg.norm(query)
    results = store.search(query, k=3)
    assert isinstance(results, list)
    assert len(results) == 3

def test_search_result_has_required_keys(store_with_data):
    store, _ = store_with_data
    query = np.random.randn(384).astype(np.float32)
    query /= np.linalg.norm(query)
    results = store.search(query, k=1)
    assert {"chunk_id", "text", "score"}.issubset(results[0].keys())

def test_search_scores_are_floats(store_with_data):
    store, _ = store_with_data
    query = np.random.randn(384).astype(np.float32)
    query /= np.linalg.norm(query)
    results = store.search(query, k=3)
    assert all(isinstance(r["score"], float) for r in results)

def test_save_and_load(store_with_data):
    store, chunks = store_with_data
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_index")
        store.save(path)
        assert os.path.exists(f"{path}.faiss")
        assert os.path.exists(f"{path}_meta.json")

        loaded = FAISSStore("fixed")
        loaded.load(path)
        assert len(loaded.chunk_texts) == len(chunks)
        assert loaded.chunk_ids == store.chunk_ids

def test_parent_child_search_returns_parent_text():
    chunks = make_chunks(4, strategy="parent_child")
    for c in chunks:
        c["chunk_id"] = f"pc_child_{chunks.index(c):03d}"
    parent_map = {c["chunk_id"]: f"PARENT TEXT for {c['chunk_id']}" for c in chunks}

    mock_embedder, _ = make_mock_embedder(4)
    store = FAISSStore("parent_child")
    store.build(chunks, mock_embedder, parent_map=parent_map)

    query = np.random.randn(384).astype(np.float32)
    query /= np.linalg.norm(query)
    results = store.search(query, k=2)

    for r in results:
        assert r["text"].startswith("PARENT TEXT for")

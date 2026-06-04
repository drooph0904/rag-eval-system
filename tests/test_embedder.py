# rag_eval_phase2/tests/test_embedder.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from indexer.embedder import Embedder


@pytest.fixture(scope="module")
def embedder():
    return Embedder()


def test_embed_returns_ndarray(embedder):
    result = embedder.embed(["hello world"])
    assert isinstance(result, np.ndarray)

def test_embed_shape(embedder):
    texts = ["hello world", "foo bar baz"]
    result = embedder.embed(texts)
    assert result.shape == (2, 384)

def test_embed_dtype_float32(embedder):
    result = embedder.embed(["test"])
    assert result.dtype == np.float32

def test_embed_one_returns_1d(embedder):
    result = embedder.embed_one("test sentence")
    assert result.shape == (384,)

def test_embed_one_dtype_float32(embedder):
    result = embedder.embed_one("test")
    assert result.dtype == np.float32

def test_embeddings_are_unit_normalized(embedder):
    result = embedder.embed(["hello", "world", "test"])
    norms = np.linalg.norm(result, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)

def test_embed_one_is_unit_normalized(embedder):
    result = embedder.embed_one("hello world")
    norm = np.linalg.norm(result)
    assert abs(norm - 1.0) < 1e-5

def test_different_texts_produce_different_embeddings(embedder):
    a = embedder.embed_one("the sky is blue")
    b = embedder.embed_one("neural networks learn features")
    assert not np.allclose(a, b)

def test_large_batch_works(embedder):
    texts = ["sentence number " + str(i) for i in range(200)]
    result = embedder.embed(texts)
    assert result.shape == (200, 384)

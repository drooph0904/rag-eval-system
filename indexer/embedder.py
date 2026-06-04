# rag_eval_phase2/indexer/embedder.py
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL


class Embedder:
    def __init__(self):
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        self._model = SentenceTransformer(EMBEDDING_MODEL)

    def embed(self, texts: list[str]) -> np.ndarray:
        batch_size = 64
        all_embeddings = []
        start = time.time()
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            emb = self._model.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
            all_embeddings.append(emb.astype(np.float32))
        elapsed = time.time() - start
        batches = (len(texts) + batch_size - 1) // batch_size
        print(f"Embedded {len(texts)} texts in {batches} batch(es) — {elapsed:.2f}s")
        return np.vstack(all_embeddings)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

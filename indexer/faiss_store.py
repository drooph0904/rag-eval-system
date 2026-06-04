# rag_eval_phase2/indexer/faiss_store.py
import json
import numpy as np
import faiss
from config import INDEX_DIR


class FAISSStore:
    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name
        self.dimension = 384
        self.index = None
        self.chunk_texts = []
        self.chunk_ids = []
        self.parent_map = {}

    def build(self, chunks: list[dict], embedder, parent_map: dict = None) -> None:
        texts = [c["text"] for c in chunks]
        embeddings = embedder.embed(texts)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        self.chunk_texts = [c["text"] for c in chunks]
        self.chunk_ids = [c["chunk_id"] for c in chunks]
        self.parent_map = parent_map or {}
        print(f"Built FAISS index: strategy={self.strategy_name}, vectors={self.index.ntotal}, dim={self.dimension}")

    def search(self, query_vector: np.ndarray, k: int) -> list[dict]:
        scores, indices = self.index.search(query_vector.reshape(1, -1), k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk_id = self.chunk_ids[idx]
            text = self.chunk_texts[idx]
            if self.parent_map and chunk_id in self.parent_map:
                text = self.parent_map[chunk_id]
            results.append({"chunk_id": chunk_id, "text": text, "score": float(score)})
        return results

    def save(self, path: str) -> None:
        faiss.write_index(self.index, f"{path}.faiss")
        meta = {"chunk_texts": self.chunk_texts, "chunk_ids": self.chunk_ids, "parent_map": self.parent_map}
        with open(f"{path}_meta.json", "w") as f:
            json.dump(meta, f)

    def load(self, path: str) -> None:
        self.index = faiss.read_index(f"{path}.faiss")
        with open(f"{path}_meta.json") as f:
            meta = json.load(f)
        self.chunk_texts = meta["chunk_texts"]
        self.chunk_ids = meta["chunk_ids"]
        self.parent_map = meta.get("parent_map", {})


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    embedder_path = os.path.join(os.path.dirname(__file__), "embedder.py")
    if not os.path.exists(embedder_path):
        print("Smoke test skipped: indexer/embedder.py does not exist yet.")
        sys.exit(0)

    from indexer.embedder import Embedder
    sentences = [
        "The sky is blue and the sun is bright.",
        "Machine learning models learn from data.",
        "Python is a high-level programming language.",
        "Neural networks are inspired by the brain.",
        "FAISS enables fast similarity search.",
        "Embeddings map text to vector space.",
        "Cosine similarity measures angle between vectors.",
        "Transformers use attention mechanisms.",
        "RAG combines retrieval with generation.",
        "Dense retrieval outperforms sparse methods.",
    ]
    chunks = [{"chunk_id": f"fixed_{i:03d}", "text": s, "token_count": 10, "strategy": "fixed", "char_start": 0} for i, s in enumerate(sentences)]
    embedder = Embedder()
    store = FAISSStore("fixed")
    store.build(chunks, embedder)
    query_vec = embedder.embed_one("What is FAISS used for?")
    results = store.search(query_vec, k=3)
    for r in results:
        print(f"Score: {r['score']:.4f} | {r['text']}")

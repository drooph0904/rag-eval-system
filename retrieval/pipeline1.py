# retrieval/pipeline1.py
from config import TOP_K_FINAL


class Pipeline1:
    def retrieve(self, question: str, store, embedder) -> list[dict]:
        query_vector = embedder.embed_one(question)
        results = store.search(query_vector, k=TOP_K_FINAL)
        print(f"Pipeline1 | Q: {question[:60]!r} | chunks: {len(results)} | top score: {results[0]['score']:.4f}" if results else "Pipeline1 | no results")
        return results

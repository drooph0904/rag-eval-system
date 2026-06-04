# rag_eval_phase2/retrieval/pipeline2.py
from config import TOP_K_FINAL
from retrieval.hyde import hypothetical_answer


class Pipeline2:
    def retrieve(self, question: str, store, embedder, openai_client) -> list[dict]:
        try:
            hyp_answer = hypothetical_answer(question, openai_client)
            query_vector = embedder.embed_one(hyp_answer)
        except Exception as e:
            print(f"Pipeline2 | HyDE failed ({e}), falling back to direct embedding")
            query_vector = embedder.embed_one(question)
        results = store.search(query_vector, k=TOP_K_FINAL)
        print(f"Pipeline2 | Q: {question[:60]!r} | chunks: {len(results)}")
        return results

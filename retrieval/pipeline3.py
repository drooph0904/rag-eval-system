# retrieval/pipeline3.py
# Pipeline 3: HyDE → FAISS top-K retrieval → Cohere Rerank → top TOP_K_FINAL
#
# Cohere client: cohere 7.0.3 ClientV2
# rerank() signature (all keyword-only):
#   model: str, query: str, documents: Sequence[str], top_n: Optional[int], ...
# Response: V2RerankResponse with .results list of V2RerankResponseResultsItem
#   each item has: .index (int), .relevance_score (float)

from config import TOP_K_RETRIEVAL, TOP_K_FINAL, COHERE_RERANK_MODEL
from retrieval.hyde import hypothetical_answer


class Pipeline3:
    def retrieve(
        self,
        question: str,
        store,
        embedder,
        openai_client,
        cohere_client,
    ) -> list[dict]:
        # --- Step 1: HyDE — generate a hypothetical answer and embed it (cached) ---
        try:
            hyp_answer = hypothetical_answer(question, openai_client)
            query_vector = embedder.embed_one(hyp_answer)
        except Exception as e:
            print(f"Pipeline3 | HyDE failed ({e}), using direct embedding")
            query_vector = embedder.embed_one(question)

        # --- Step 2: FAISS retrieval — fetch TOP_K_RETRIEVAL candidates ---
        candidates = store.search(query_vector, k=TOP_K_RETRIEVAL)

        # --- Step 3: Cohere rerank — reorder candidates and keep TOP_K_FINAL ---
        try:
            rerank_response = cohere_client.rerank(
                model=COHERE_RERANK_MODEL,
                query=question,
                documents=[c["text"] for c in candidates],
                top_n=TOP_K_FINAL,
            )
            results = [
                {
                    "chunk_id": candidates[r.index]["chunk_id"],
                    "text": candidates[r.index]["text"],
                    "score": float(r.relevance_score),
                }
                for r in rerank_response.results
            ]
        except Exception as e:
            print(
                f"Pipeline3 | Cohere rerank failed ({e}), "
                f"falling back to top {TOP_K_FINAL} FAISS results"
            )
            results = candidates[:TOP_K_FINAL]

        print(f"Pipeline3 | Q: {question[:60]!r} | chunks: {len(results)}")
        return results

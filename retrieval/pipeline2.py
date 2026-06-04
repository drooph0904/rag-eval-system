# rag_eval_phase2/retrieval/pipeline2.py
from config import TOP_K_FINAL, HYDE_MODEL

HYDE_PROMPT = """Generate a concise factual paragraph that would directly answer this question.
Write as if extracted from a document. Do not mention the question itself.
Question: {question}
Paragraph:"""


class Pipeline2:
    def retrieve(self, question: str, store, embedder, openai_client) -> list[dict]:
        try:
            response = openai_client.chat.completions.create(
                model=HYDE_MODEL,
                messages=[{"role": "user", "content": HYDE_PROMPT.format(question=question)}],
                temperature=0,
                max_tokens=200,
            )
            hyp_answer = response.choices[0].message.content.strip()
            query_vector = embedder.embed_one(hyp_answer)
        except Exception as e:
            print(f"Pipeline2 | HyDE failed ({e}), falling back to direct embedding")
            query_vector = embedder.embed_one(question)
        results = store.search(query_vector, k=TOP_K_FINAL)
        print(f"Pipeline2 | Q: {question[:60]!r} | chunks: {len(results)}")
        return results

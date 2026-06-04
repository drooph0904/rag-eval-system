# rag_eval_phase2/retrieval/hyde.py
"""HyDE (Hypothetical Document Embeddings) helper with per-run caching.

The hypothetical answer depends ONLY on the question (and the fixed model/prompt),
not on the chunking strategy, pipeline, or retrieved chunks. Without caching it is
regenerated up to 6× per question in Stage 1 (pipelines 2 & 3 × 3 strategies) and
again in Stage 2. Caching by question text collapses that to one LLM call per unique
question per run.

The cache is process-scoped: each `main.py` invocation is a fresh process (and the
UI spawns a subprocess per run), so it never leaks across runs. Call `clear_cache()`
between unit tests.
"""

from config import HYDE_MODEL

HYDE_PROMPT = """Generate a concise factual paragraph that would directly answer this question.
Write as if extracted from a document. Do not mention the question itself.
Question: {question}
Paragraph:"""

_cache: dict[str, str] = {}


def hypothetical_answer(question: str, openai_client) -> str:
    """Return the HyDE hypothetical answer for `question`, generating it once and
    caching by question text. Raises on API failure (callers handle the fallback)."""
    if question in _cache:
        return _cache[question]
    response = openai_client.chat.completions.create(
        model=HYDE_MODEL,
        messages=[{"role": "user", "content": HYDE_PROMPT.format(question=question)}],
        temperature=0,
        max_tokens=200,
    )
    answer = response.choices[0].message.content.strip()
    _cache[question] = answer
    return answer


def clear_cache() -> None:
    _cache.clear()

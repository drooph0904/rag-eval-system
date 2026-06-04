# rag_eval_phase2/evaluation/stage1_eval.py
import re

import numpy as np

from config import STAGE1_SIMILARITY_THRESHOLD, STAGE1_LEXICAL_OVERLAP


_STOPWORDS = {
    "a", "an", "the", "of", "in", "at", "on", "to", "and", "or", "is", "are",
    "for", "with", "by", "from", "as", "that", "this", "it", "its", "he", "she",
    "they", "was", "were", "be", "his", "her", "their",
}


def _significant_tokens(text: str) -> list[str]:
    """Lowercase content tokens: keep alphanumerics that are non-stopword words
    (len >= 2) or pure numbers (so '5' in '5 engineers' counts)."""
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in toks if t.isdigit() or (len(t) >= 2 and t not in _STOPWORDS)]


def _lexical_relevant(answer: str, chunk_text: str, threshold: float) -> bool:
    """True if the chunk literally contains most of the answer's significant tokens.
    High-precision signal for short factual answers that embeddings under-score."""
    answer_tokens = set(_significant_tokens(answer))
    if not answer_tokens:
        return False
    chunk_tokens = set(_significant_tokens(chunk_text))
    overlap = sum(1 for t in answer_tokens if t in chunk_tokens) / len(answer_tokens)
    return overlap >= threshold


class Stage1Evaluator:
    def evaluate(self, question: str, ground_truth: str, retrieved_chunks: list[dict], embedder) -> dict:
        chunk_texts = [c["text"] for c in retrieved_chunks]
        total = len(retrieved_chunks)

        if total == 0:
            return {
                "question": question,
                "context_precision": 0.0,
                "context_recall": 0.0,
                "relevant_chunks_found": 0,
                "total_chunks_retrieved": 0,
            }

        gt_vec = embedder.embed_one(ground_truth)
        chunk_vecs = embedder.embed(chunk_texts)
        similarities = chunk_vecs @ gt_vec  # cosine (vectors are normalized)

        # A chunk is relevant if the answer embeds similarly OR the chunk literally
        # contains the answer's key terms. The embedding threshold is unchanged, so
        # documents that already matched via embeddings behave exactly as before.
        relevant = [
            float(s) > STAGE1_SIMILARITY_THRESHOLD
            or _lexical_relevant(ground_truth, text, STAGE1_LEXICAL_OVERLAP)
            for s, text in zip(similarities, chunk_texts)
        ]

        relevant_count = sum(relevant)
        context_precision = relevant_count / total
        context_recall = 1.0 if any(relevant) else 0.0

        return {
            "question": question,
            "context_precision": context_precision,
            "context_recall": context_recall,
            "relevant_chunks_found": relevant_count,
            "total_chunks_retrieved": total,
        }

    def aggregate(self, results: list[dict]) -> dict:
        return {
            "mean_context_precision": sum(r["context_precision"] for r in results) / len(results),
            "mean_context_recall": sum(r["context_recall"] for r in results) / len(results),
        }

# rag_eval_phase2/evaluation/stage1_eval.py
import numpy as np
from config import STAGE1_SIMILARITY_THRESHOLD


class Stage1Evaluator:
    def evaluate(self, question: str, ground_truth: str, retrieved_chunks: list[dict], embedder) -> dict:
        gt_vec = embedder.embed_one(ground_truth)
        chunk_texts = [c["text"] for c in retrieved_chunks]
        chunk_vecs = embedder.embed(chunk_texts)

        # cosine similarity (vectors are already normalized)
        similarities = chunk_vecs @ gt_vec
        relevant = [float(s) > STAGE1_SIMILARITY_THRESHOLD for s in similarities]

        relevant_count = sum(relevant)
        total = len(retrieved_chunks)
        context_precision = relevant_count / total if total > 0 else 0.0
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

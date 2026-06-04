# rag_eval_phase2/evaluation/stage2_eval.py
#
# Stage 2 — answer generation + answer-quality scoring against the golden
# ground-truth answer. No RAGAS: both scores reference the Phase 1 golden set.
#
#   answer_similarity  = cosine(embed(generated_answer), embed(ground_truth))
#                        using the same MiniLM embedder as Stage 1 (free, deterministic).
#   answer_correctness = gpt-4o-mini judges the generated answer vs the reference
#                        answer on a 0-100 scale, normalized to 0-1.

import re

import numpy as np

from config import ANSWER_MODEL, JUDGE_MODEL


ANSWER_PROMPT = """Answer the question using ONLY the provided context.
If the context does not contain enough information, say "I don't know".

Context:
{context}

Question: {question}
Answer:"""


JUDGE_PROMPT = """You are grading a candidate answer against a reference (correct) answer.

Question: {question}
Reference answer: {reference}
Candidate answer: {candidate}

Score from 0 to 100 how well the candidate answer conveys the same information as
the reference answer (100 = fully correct / equivalent, 0 = wrong or unrelated).
Reply with ONLY a single integer from 0 to 100."""


def _parse_score(text: str) -> float:
    """Extract the first integer from the judge's reply and normalize to [0, 1]."""
    match = re.search(r"\d+", text)
    if not match:
        return 0.0
    score = int(match.group())
    return max(0.0, min(1.0, score / 100.0))


class Stage2Evaluator:
    def evaluate(
        self,
        question: str,
        ground_truth: str,
        retrieved_chunks: list,
        openai_client,
        embedder,
    ) -> dict | None:
        """
        Generate an answer from the retrieved chunks, then score it against the
        golden ground-truth answer two ways: embedding similarity and an LLM judge.

        Returns a dict with keys:
          question, generated_answer, ground_truth, answer_similarity, answer_correctness
        Returns None on any exception (API failure, etc.) so a single bad question
        does not crash the whole run.
        """
        try:
            # --- 1. Generate the answer from retrieved context ---
            context = "\n\n".join([c["text"] for c in retrieved_chunks])
            answer_resp = openai_client.chat.completions.create(
                model=ANSWER_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": ANSWER_PROMPT.format(context=context, question=question),
                    }
                ],
                temperature=0,
                max_tokens=300,
            )
            generated_answer = answer_resp.choices[0].message.content.strip()

            # --- 2. answer_similarity: cosine(answer, ground_truth) ---
            ans_vec = embedder.embed_one(generated_answer)
            gt_vec = embedder.embed_one(ground_truth)
            cosine = float(np.dot(ans_vec, gt_vec))  # embeddings are unit-normalized
            answer_similarity = max(0.0, min(1.0, cosine))

            # --- 3. answer_correctness: LLM judge vs ground_truth ---
            judge_resp = openai_client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": JUDGE_PROMPT.format(
                            question=question,
                            reference=ground_truth,
                            candidate=generated_answer,
                        ),
                    }
                ],
                temperature=0,
                max_tokens=10,
            )
            answer_correctness = _parse_score(judge_resp.choices[0].message.content.strip())

            return {
                "question": question,
                "generated_answer": generated_answer,
                "ground_truth": ground_truth,
                "answer_similarity": answer_similarity,
                "answer_correctness": answer_correctness,
            }
        except Exception as e:
            print(f"Stage2Evaluator | failed for question '{question[:60]}': {e}")
            return None

    def aggregate(self, results: list) -> dict:
        """
        Mean scores across non-None results.

        Returns dict with keys: mean_answer_similarity, mean_answer_correctness
        """
        valid = [r for r in results if r is not None]
        if not valid:
            return {"mean_answer_similarity": 0.0, "mean_answer_correctness": 0.0}
        return {
            "mean_answer_similarity": sum(r["answer_similarity"] for r in valid) / len(valid),
            "mean_answer_correctness": sum(r["answer_correctness"] for r in valid) / len(valid),
        }

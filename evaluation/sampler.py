# rag_eval_phase2/evaluation/sampler.py
"""Stratified sampling of golden questions by question_type.

Used by UI-driven runs to evaluate a balanced subset that covers every question
type while bounding runtime. Deterministic: preserves input order and takes the
first `per_type` questions of each type.
"""


def stratified_sample(questions: list[dict], per_type: int) -> list[dict]:
    """
    Return up to `per_type` questions for each distinct question_type, preserving
    the original order. Types with fewer than `per_type` questions contribute all
    of theirs. Questions missing a question_type are grouped under "unknown".

    The returned list keeps questions in their original relative order.
    """
    if per_type <= 0 or not questions:
        return []

    counts: dict[str, int] = {}
    sampled: list[dict] = []
    for q in questions:
        qtype = q.get("question_type") or "unknown"
        if counts.get(qtype, 0) < per_type:
            sampled.append(q)
            counts[qtype] = counts.get(qtype, 0) + 1
    return sampled

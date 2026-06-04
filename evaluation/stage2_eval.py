# rag_eval_phase2/evaluation/stage2_eval.py
#
# ragas 0.4.3 API notes:
#   - SingleTurnSample(user_input, response, retrieved_contexts, reference)
#   - EvaluationDataset([sample, ...])
#   - evaluate(dataset, metrics=[...]) → EvaluationResult
#   - EvaluationResult[metric_name] → list of float scores (one per sample)
#   - metrics imported from ragas.metrics.collections (non-deprecated path)
#   - LLM wrapper: from openai import OpenAI; ragas.llms.llm_factory(model, client=client)
#   - evaluate() auto-creates an OpenAI LLM if none provided (uses OPENAI_API_KEY env var)

from config import ANSWER_MODEL


def _ensure_ragas_importable():
    """
    ragas 0.4.3 imports Vertex AI classes unconditionally at import time:
        from langchain_community.chat_models.vertexai import ChatVertexAI
        from langchain_community.llms import VertexAI
    but langchain-community>=0.3 removed those, so `import ragas` raises
    ModuleNotFoundError on a clean install. We never use Vertex AI, so we inject
    harmless stubs before importing ragas. Keeping this in-repo (rather than
    editing site-packages) makes the fix reproducible from requirements.txt alone.
    """
    import sys
    import types

    submodule = "langchain_community.chat_models.vertexai"
    if submodule not in sys.modules:
        try:
            __import__(submodule)
        except ModuleNotFoundError:
            stub = types.ModuleType(submodule)
            stub.ChatVertexAI = None
            sys.modules[submodule] = stub

    try:
        import langchain_community.llms as _llms
        if not hasattr(_llms, "VertexAI"):
            _llms.VertexAI = None
    except Exception:
        pass


def evaluate_with_ragas(
    question: str,
    answer: str,
    contexts: list,
    ground_truth: str,
) -> dict:
    """
    Run four RAGAS metrics for a single (question, answer, contexts, ground_truth) tuple.

    Uses ragas 0.4.3 API:
      - SingleTurnSample fields: user_input, response, retrieved_contexts, reference
      - evaluate() accepts EvaluationDataset and a list of metric objects
      - If no LLM is configured, ragas auto-creates one from OPENAI_API_KEY env var
      - EvaluationResult[metric_name] returns a list; [0] fetches the first (only) sample's score

    ragas is imported lazily (after stubbing Vertex AI) so that importing this
    module never triggers ragas's broken import chain.

    Returns
    -------
    dict with keys: faithfulness, answer_relevancy, context_precision, context_recall
    """
    _ensure_ragas_importable()
    from ragas import SingleTurnSample, EvaluationDataset, evaluate
    from ragas.metrics.collections import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )

    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
        reference=ground_truth,
    )
    dataset = EvaluationDataset(samples=[sample])
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    return {
        "faithfulness": float(result["faithfulness"][0]),
        "answer_relevancy": float(result["answer_relevancy"][0]),
        "context_precision": float(result["context_precision"][0]),
        "context_recall": float(result["context_recall"][0]),
    }


ANSWER_PROMPT = """Answer the question using ONLY the provided context.
If the context does not contain enough information, say "I don't know".

Context:
{context}

Question: {question}
Answer:"""


class Stage2Evaluator:
    def evaluate(
        self,
        question: str,
        ground_truth: str,
        retrieved_chunks: list,
        openai_client,
    ) -> dict | None:
        """
        Generate an answer with gpt-4o-mini, then score it with RAGAS.

        Returns a dict with keys:
          question, generated_answer, ground_truth,
          faithfulness, answer_relevancy, context_precision, context_recall

        Returns None on any exception (API failures, ragas errors, etc.).
        """
        try:
            context = "\n\n".join([c["text"] for c in retrieved_chunks])
            response = openai_client.chat.completions.create(
                model=ANSWER_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": ANSWER_PROMPT.format(
                            context=context, question=question
                        ),
                    }
                ],
                temperature=0,
                max_tokens=300,
            )
            generated_answer = response.choices[0].message.content.strip()
            scores = evaluate_with_ragas(
                question=question,
                answer=generated_answer,
                contexts=[c["text"] for c in retrieved_chunks],
                ground_truth=ground_truth,
            )
            return {
                "question": question,
                "generated_answer": generated_answer,
                "ground_truth": ground_truth,
                **scores,
            }
        except Exception as e:
            print(
                f"Stage2Evaluator | failed for question '{question[:60]}': {e}"
            )
            return None

    def aggregate(self, results: list) -> dict:
        """
        Compute mean scores across all non-None results.

        Parameters
        ----------
        results : list of dict | None

        Returns
        -------
        dict with keys: mean_faithfulness, mean_answer_relevancy,
                        mean_context_precision, mean_context_recall
        """
        valid = [r for r in results if r is not None]
        if not valid:
            return {
                "mean_faithfulness": 0.0,
                "mean_answer_relevancy": 0.0,
                "mean_context_precision": 0.0,
                "mean_context_recall": 0.0,
            }
        return {
            "mean_faithfulness": sum(r["faithfulness"] for r in valid) / len(valid),
            "mean_answer_relevancy": sum(r["answer_relevancy"] for r in valid) / len(valid),
            "mean_context_precision": sum(r["context_precision"] for r in valid) / len(valid),
            "mean_context_recall": sum(r["context_recall"] for r in valid) / len(valid),
        }

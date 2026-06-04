# rag_eval_phase2/main.py
"""
Main orchestrator for RAG Evaluation Phase 2.

Usage:
    python main.py <pdf_path> <golden_dataset_path>

Orchestration sequence:
    1. Load golden questions from JSON
    2. Extract PDF text via Phase 1's PDFExtractor
    3. Build 3 FAISS indexes (fixed / semantic / parent_child) and save to INDEX_DIR
    4. STAGE 1: run all 9 (strategy × pipeline) combos, evaluate with Stage1Evaluator
    5. Select top 3 combos by mean_context_precision
    6. STAGE 2: run those 3 with Stage2Evaluator
    7. Save via ResultsStore
    8. Print final summary table + winner + streamlit run command
"""

import sys
import os
import json

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# sys.path setup
# 1. Ensure THIS package's directory is at the front so our config.py wins.
# 2. Append the vendored Phase 1 package (subdirectory) so pdf_extractor can be
#    imported WITHOUT shadowing local modules (append, not insert-at-0).
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_GDG_DIR = os.path.join(_THIS_DIR, "golden_dataset_generator")

if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

if _GDG_DIR not in sys.path:
    sys.path.append(_GDG_DIR)

# ---------------------------------------------------------------------------
# Local package imports — safe at module level; no API keys required
# ---------------------------------------------------------------------------
from config import CHUNKING_STRATEGIES, PIPELINE_NAMES, INDEX_DIR, RESULTS_DIR
from indexer.chunker import Chunker
from indexer.embedder import Embedder
from indexer.faiss_store import FAISSStore
from retrieval.pipeline1 import Pipeline1
from retrieval.pipeline2 import Pipeline2
from retrieval.pipeline3 import Pipeline3
from evaluation.stage1_eval import Stage1Evaluator
from evaluation.stage2_eval import Stage2Evaluator
from evaluation.results_store import ResultsStore

# PDFExtractor lives in the sibling golden_dataset_generator package
from pdf_extractor import PDFExtractor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_golden_dataset(path: str) -> list[dict]:
    """Load golden dataset JSON and return the questions list."""
    with open(path) as f:
        data = json.load(f)
    questions = data.get("questions", [])
    if not questions:
        print(f"ERROR: Golden dataset at {path} contains 0 questions.")
        sys.exit(1)
    return questions


def _get_ground_truth(q: dict) -> str:
    """Return ground-truth answer; supports both 'ground_truth' and legacy 'answer' key."""
    return q.get("ground_truth") or q.get("answer", "")


def _parse_combo_key(combo_key: str):
    """
    Split a combo key such as 'parent_child_pipeline_2' into (strategy, pipeline_name).

    Strategy names may contain underscores (e.g. 'parent_child'), so we split on
    '_pipeline_' from the right to avoid ambiguity.
    """
    parts = combo_key.rsplit("_pipeline_", 1)
    strategy = parts[0]
    pipeline_name = f"pipeline_{parts[1]}"
    return strategy, pipeline_name


def _retrieve(pipeline_name: str, question: str, store, embedder,
              openai_client, cohere_client, p1, p2, p3) -> list[dict]:
    """Dispatch retrieval to the correct pipeline."""
    if pipeline_name == "pipeline_1":
        return p1.retrieve(question, store, embedder)
    elif pipeline_name == "pipeline_2":
        return p2.retrieve(question, store, embedder, openai_client)
    else:
        return p3.retrieve(question, store, embedder, openai_client, cohere_client)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- Validate CLI args ---
    if len(sys.argv) < 3:
        print("Usage: python main.py <pdf_path> <golden_dataset_path>")
        sys.exit(1)

    # --- Validate API keys (inside main so the module can be imported without them) ---
    openai_api_key = os.getenv("OPENAI_API_KEY")
    cohere_api_key = os.getenv("COHERE_API_KEY")

    if not openai_api_key:
        print("ERROR: OPENAI_API_KEY not set. Copy .env.example to .env and fill in your key.")
        sys.exit(1)
    if not cohere_api_key:
        print("ERROR: COHERE_API_KEY not set. Copy .env.example to .env and fill in your key.")
        sys.exit(1)

    # --- Create API clients (inside main, after key validation) ---
    import openai
    import cohere

    openai_client = openai.OpenAI(api_key=openai_api_key)
    cohere_client = cohere.ClientV2(api_key=cohere_api_key)

    # --- Parse paths ---
    pdf_path = sys.argv[1]
    golden_path = sys.argv[2]
    pdf_name = os.path.basename(pdf_path)

    print(f"\n{'='*60}")
    print("RAG Evaluation Phase 2")
    print(f"PDF: {pdf_path}")
    print(f"Golden dataset: {golden_path}")
    print(f"{'='*60}\n")

    # --- Load golden questions ---
    questions = load_golden_dataset(golden_path)
    print(f"Loaded {len(questions)} questions from golden dataset.\n")

    # --- Extract PDF text ---
    extractor = PDFExtractor()
    pages = extractor.extract(pdf_path)
    raw_text = " ".join(p["text"] for p in pages)
    print(f"Extracted {len(pages)} pages from PDF.\n")

    # --- Shared components ---
    embedder = Embedder()
    chunker = Chunker()
    os.makedirs(INDEX_DIR, exist_ok=True)

    # --- Build 3 FAISS indexes ---
    print(f"{'='*60}")
    print("Building indexes for all 3 chunking strategies…")
    print(f"{'='*60}\n")

    stores = {}
    for strategy in CHUNKING_STRATEGIES:
        if strategy == "parent_child":
            children, parent_map = chunker.parent_child(raw_text)
            chunks = children
            pm = parent_map
        elif strategy == "semantic":
            chunks = chunker.semantic(raw_text)
            pm = None
        else:  # fixed
            chunks = chunker.fixed(raw_text)
            pm = None

        store = FAISSStore(strategy)
        store.build(chunks, embedder, parent_map=pm)

        index_path = os.path.join(INDEX_DIR, strategy)
        store.save(index_path)
        stores[strategy] = store
        print(f"Built and saved index: {strategy} — {len(chunks)} chunks\n")

    # --- Stage 1: all 9 combinations ---
    print(f"\n{'='*60}")
    print("STAGE 1 — Running all 9 combinations (embedding similarity, no LLM call per question)")
    print(f"{'='*60}\n")

    p1 = Pipeline1()
    p2 = Pipeline2()
    p3 = Pipeline3()
    stage1_evaluator = Stage1Evaluator()
    stage1_results = {}

    for strategy in CHUNKING_STRATEGIES:
        store = stores[strategy]
        for pipeline_name in PIPELINE_NAMES:
            combo_key = f"{strategy}_{pipeline_name}"
            per_question = []

            for q in questions:
                question = q["question"]
                ground_truth = _get_ground_truth(q)
                chunks = _retrieve(pipeline_name, question, store, embedder,
                                   openai_client, cohere_client, p1, p2, p3)
                result = stage1_evaluator.evaluate(question, ground_truth, chunks, embedder)
                if result is not None:
                    result["question_type"] = q.get("question_type", "unknown")
                per_question.append(result)

            agg = stage1_evaluator.aggregate(per_question)
            stage1_results[combo_key] = {**agg, "per_question": per_question}
            print(
                f"Stage 1 | {combo_key:<30} | "
                f"precision: {agg['mean_context_precision']:.3f} | "
                f"recall: {agg['mean_context_recall']:.3f}"
            )

    # --- Select top 3 by mean_context_precision ---
    top3 = sorted(
        stage1_results.items(),
        key=lambda kv: kv[1]["mean_context_precision"],
        reverse=True,
    )[:3]
    top3_keys = [k for k, _ in top3]

    print(f"\n{'='*60}")
    print(f"Top 3 combos selected for Stage 2: {top3_keys}")
    print(f"{'='*60}\n")

    # --- Stage 2: top 3 combinations ---
    print(f"{'='*60}")
    print("STAGE 2 — Running top 3 combinations (answer generation + golden-truth scoring)")
    print(f"{'='*60}\n")

    stage2_evaluator = Stage2Evaluator()
    stage2_results = {}

    for combo_key in top3_keys:
        strategy, pipeline_name = _parse_combo_key(combo_key)
        store = stores[strategy]
        per_question = []

        for q in questions:
            question = q["question"]
            ground_truth = _get_ground_truth(q)
            chunks = _retrieve(pipeline_name, question, store, embedder,
                               openai_client, cohere_client, p1, p2, p3)
            result = stage2_evaluator.evaluate(question, ground_truth, chunks, openai_client, embedder)
            if result is not None:
                result["question_type"] = q.get("question_type", "unknown")
            per_question.append(result)

        agg = stage2_evaluator.aggregate(per_question)
        stage2_results[combo_key] = {**agg, "per_question": per_question}
        print(
            f"Stage 2 | {combo_key:<30} | "
            f"correctness: {agg['mean_answer_correctness']:.3f} | "
            f"similarity: {agg['mean_answer_similarity']:.3f}"
        )

    # --- Save results ---
    results_store = ResultsStore()
    results_path = results_store.save(
        pdf_name,
        golden_path,
        stage1_results,
        stage2_results,
        total_questions=len(questions),
    )

    # --- Final summary ---
    print(f"\n{'='*60}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*60}")

    print("\nStage 1 Rankings (by context precision):")
    for k, v in sorted(
        stage1_results.items(),
        key=lambda kv: kv[1]["mean_context_precision"],
        reverse=True,
    ):
        marker = " <- Stage 2" if k in top3_keys else ""
        print(
            f"  {k:<30} precision={v['mean_context_precision']:.3f}  "
            f"recall={v['mean_context_recall']:.3f}{marker}"
        )

    print("\nStage 2 Rankings (by answer correctness vs golden ground-truth):")
    for k, v in sorted(
        stage2_results.items(),
        key=lambda kv: kv[1]["mean_answer_correctness"],
        reverse=True,
    ):
        print(
            f"  {k:<30} correctness={v['mean_answer_correctness']:.3f}  "
            f"similarity={v['mean_answer_similarity']:.3f}"
        )

    with open(results_path) as f:
        saved = json.load(f)
    winner = saved["winner"]
    print(f"\nWINNER: {winner['combination']}")
    print(f"Reason: {winner['reason']}")
    print(f"\nTo view results in the UI, run:")
    print(f"  streamlit run ui/app.py -- --results {results_path}")


if __name__ == "__main__":
    main()

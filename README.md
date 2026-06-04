# RAG Evaluation System — Phase 2

A RAG (Retrieval-Augmented Generation) evaluation engine. Give it a PDF and it
benchmarks **9 retrieval pipelines** against a golden Q&A set, scoring each
pipeline's answers against ground truth and crowning the best setup — all in a
Streamlit dashboard.

It builds on **Phase 1** (a golden-dataset generator that turns a PDF into typed
Q&A pairs), which lives in a sibling project, `golden_dataset_generator/`.

## What it does

1. **Index** the PDF three ways — `fixed`, `semantic`, and `parent_child` chunking.
2. **Retrieve** with three pipelines:
   - `pipeline_1` — embed query → FAISS top-K
   - `pipeline_2` — HyDE (hypothetical answer) → embed → FAISS top-K
   - `pipeline_3` — HyDE → FAISS → **Cohere rerank** → top-K
3. **Stage 1 (cheap, no generation):** score retrieval quality per combo —
   context precision/recall via embedding similarity **and** a lexical
   answer-presence check (robust for short factual answers).
4. **Stage 2 (top 3 combos):** generate an answer from the retrieved chunks with
   `gpt-4o-mini`, then score it **against the golden ground-truth**:
   - `answer_similarity` — cosine of answer vs ground-truth embeddings
   - `answer_correctness` — `gpt-4o-mini` judge (0–100, normalized)
5. **Winner** = highest mean answer correctness. Results saved as JSON and shown
   in the dashboard with a per-question-type breakdown.

The 9 combinations: `{fixed, semantic, parent_child} × {pipeline_1, pipeline_2, pipeline_3}`.

## Stack

Python 3.11+, `faiss-cpu`, `sentence-transformers` (all-MiniLM-L6-v2), `tiktoken`,
`openai` (HyDE + answer + judge), `cohere` (rerank), `streamlit`, `pymupdf`.

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env        # add OPENAI_API_KEY and COHERE_API_KEY

# Evaluate a PDF against a Phase 1 golden dataset (CLI):
.venv/bin/python main.py path/to/doc.pdf path/to/doc_golden.json

# Or use the UI — upload any PDF, it runs Phase 1 + Phase 2 end to end:
.venv/bin/streamlit run ui/app.py
```

## Tests

```bash
.venv/bin/python -m pytest -q
```

## Configuration

All tunables live in `config.py` (chunk sizes, top-K, models, Stage 1 thresholds,
`SAMPLE_PER_TYPE` for UI-run stratified sampling).

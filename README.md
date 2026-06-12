# RAG Evaluation System

A RAG (Retrieval-Augmented Generation) evaluation engine. Give it a PDF and it
benchmarks **9 retrieval pipelines** against a golden Q&A set, scoring each
pipeline's answers against ground truth and crowning the best setup — all in a
Streamlit dashboard.

**Self-contained:** this repo includes **Phase 1** — the golden-dataset generator
that turns a PDF into typed Q&A pairs — under `golden_dataset_generator/`. Both
phases share one virtualenv and one `.env`, so a single clone runs the whole
project end to end.

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

One venv and one `.env` at the repo root serve both phases.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env        # add OPENAI_API_KEY and COHERE_API_KEY

# Easiest: the UI — upload any PDF; it runs Phase 1 (generate golden Q&A)
# then Phase 2 (evaluate) end to end and shows the dashboard:
.venv/bin/streamlit run ui/app.py
```

### Or run the phases from the CLI

```bash
# Phase 1 — generate a golden Q&A dataset from a PDF:
cd golden_dataset_generator
../.venv/bin/python main.py /path/to/doc.pdf      # -> output/doc_golden.json
cd ..

# Phase 2 — evaluate the 9 pipelines against that golden set:
.venv/bin/python main.py /path/to/doc.pdf golden_dataset_generator/output/doc_golden.json
```

## Project layout

```
.                              # Phase 2 — evaluation engine
├── indexer/  retrieval/  evaluation/  ui/
├── main.py                    # Phase 2 orchestrator
├── pipeline_runner.py         # UI flow: runs Phase 1 + sample + Phase 2
└── golden_dataset_generator/  # Phase 1 — golden Q&A generator (vendored)
```

## Tests

```bash
.venv/bin/python -m pytest -q                         # Phase 2 suite
cd golden_dataset_generator && ../.venv/bin/python -m pytest -q   # Phase 1 suite
```

## Configuration

All tunables live in `config.py` (chunk sizes, top-K, models, Stage 1 thresholds,
`SAMPLE_PER_TYPE` for UI-run stratified sampling).

# rag_eval_phase2/pipeline_runner.py
"""
Orchestrate an end-to-end evaluation of an uploaded PDF for the Streamlit UI:

    Phase 1 (golden Q&A generation, whole PDF, its own venv)
      -> stratified sample (<= SAMPLE_PER_TYPE per question_type)
      -> Phase 2 (retrieval + answer-quality evaluation)
      -> results JSON

`run_pipeline()` is a generator that yields progress events so the UI can stream
live logs:

    {"type": "log",   "line": str}
    {"type": "phase", "name": str}
    {"type": "done",  "results_path": str}
    {"type": "error", "message": str}

Pure helpers (path derivation, sample-file writing) are factored out for testing.
"""

import os
import sys
import json
import subprocess

from config import SAMPLE_PER_TYPE
from evaluation.sampler import stratified_sample

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# Phase 1 (golden generator) is vendored as a subdirectory of this repo and shares
# the same virtualenv, so a single clone runs the whole project end to end.
_GDG_DIR = os.path.join(_THIS_DIR, "golden_dataset_generator")

_PHASE2_PY = os.path.join(_THIS_DIR, ".venv", "bin", "python")
_PHASE1_PY = _PHASE2_PY  # one shared venv for both phases


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #

def _stem(pdf_path: str) -> str:
    return os.path.splitext(os.path.basename(pdf_path))[0]


def golden_output_path(pdf_path: str) -> str:
    """Where Phase 1 writes the generated golden dataset."""
    return os.path.join(_GDG_DIR, "output", f"{_stem(pdf_path)}_golden.json")


def sampled_output_path(pdf_path: str) -> str:
    """Where we write the stratified-sampled golden dataset."""
    return os.path.join(_GDG_DIR, "output", f"{_stem(pdf_path)}_golden_sampled.json")


def results_path_for(pdf_path: str) -> str:
    """Where Phase 2 writes the eval results."""
    return os.path.join(_THIS_DIR, "results", f"{_stem(pdf_path)}_eval_results.json")


def write_sampled_golden(golden_path: str, sampled_path: str, per_type: int) -> int:
    """
    Read a Phase 1 golden dataset, stratified-sample its questions (<= per_type per
    question_type), write the reduced dataset (metadata preserved), return the count.
    """
    with open(golden_path) as f:
        data = json.load(f)
    questions = data.get("questions", [])
    sampled = stratified_sample(questions, per_type)
    data["questions"] = sampled
    with open(sampled_path, "w") as f:
        json.dump(data, f, indent=2)
    return len(sampled)


# --------------------------------------------------------------------------- #
# Subprocess streaming
# --------------------------------------------------------------------------- #

def _stream_subprocess(cmd: list[str], cwd: str):
    """Run cmd, yielding {'type':'log','line':...} per output line. Final yield is
    {'type':'exit','code':int}."""
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=os.environ.copy(),
    )
    for line in proc.stdout:
        yield {"type": "log", "line": line.rstrip()}
    proc.wait()
    yield {"type": "exit", "code": proc.returncode}


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def run_pipeline(pdf_path: str, per_type: int = SAMPLE_PER_TYPE):
    """Generator: run Phase 1 -> sample -> Phase 2 for an uploaded PDF."""
    pdf_path = os.path.abspath(pdf_path)

    if not os.path.exists(pdf_path):
        yield {"type": "error", "message": f"PDF not found: {pdf_path}"}
        return
    if not pdf_path.lower().endswith(".pdf"):
        yield {"type": "error", "message": "File must be a .pdf"}
        return
    if not os.path.exists(_PHASE1_PY):
        yield {"type": "error", "message": f"Phase 1 venv python not found at {_PHASE1_PY}"}
        return

    # --- Phase 1: generate golden dataset (whole PDF) ---
    yield {"type": "phase", "name": "Phase 1 — generating golden Q&A (whole PDF)"}
    for ev in _stream_subprocess([_PHASE1_PY, "main.py", pdf_path], cwd=_GDG_DIR):
        if ev["type"] == "exit":
            if ev["code"] != 0:
                yield {"type": "error", "message": f"Phase 1 exited with code {ev['code']}"}
                return
        else:
            yield ev

    golden = golden_output_path(pdf_path)
    if not os.path.exists(golden):
        yield {"type": "error", "message": f"Phase 1 produced no golden file at {golden}"}
        return

    # --- Stratified sample ---
    yield {"type": "phase", "name": f"Sampling up to {per_type} questions per type"}
    sampled = sampled_output_path(pdf_path)
    count = write_sampled_golden(golden, sampled, per_type)
    if count == 0:
        yield {"type": "error", "message": "No questions generated from this PDF (scanned or no extractable text?)."}
        return
    yield {"type": "log", "line": f"Evaluating {count} questions (stratified by type)."}

    # --- Phase 2: evaluate ---
    yield {"type": "phase", "name": "Phase 2 — retrieval + answer-quality evaluation"}
    for ev in _stream_subprocess([_PHASE2_PY, "main.py", pdf_path, sampled], cwd=_THIS_DIR):
        if ev["type"] == "exit":
            if ev["code"] != 0:
                yield {"type": "error", "message": f"Phase 2 exited with code {ev['code']}"}
                return
        else:
            yield ev

    results = results_path_for(pdf_path)
    if not os.path.exists(results):
        yield {"type": "error", "message": f"Phase 2 produced no results file at {results}"}
        return

    yield {"type": "done", "results_path": results}

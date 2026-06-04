# rag_eval_phase2/ui/app.py
import sys
import os
import json
import glob
import argparse

import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import RESULTS_DIR, UPLOADS_DIR, CHUNK_SIZE, EMBEDDING_MODEL, COHERE_RERANK_MODEL, SAMPLE_PER_TYPE
from pipeline_runner import run_pipeline


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def find_latest_results() -> str | None:
    files = glob.glob(os.path.join(RESULTS_DIR, "*_eval_results.json"))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def parse_args() -> str | None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=None)
    try:
        args, _ = parser.parse_known_args()
        return args.results
    except SystemExit:
        return None


def render_run_panel():
    """Upload a PDF and run the full Phase 1 -> sample -> Phase 2 pipeline live."""
    with st.expander("🚀 Evaluate a new PDF (runs Phase 1 + Phase 2)", expanded=False):
        st.caption(
            f"Uploads a PDF, generates golden Q&A over the whole document (Phase 1), "
            f"evaluates a stratified sample of up to {SAMPLE_PER_TYPE} questions per type "
            f"across all 9 pipelines (Phase 2), then shows the dashboard below. "
            f"Large PDFs can take several minutes."
        )
        uploaded_pdf = st.file_uploader("PDF to evaluate", type="pdf", key="pdf_uploader")
        run_clicked = st.button("Run full evaluation", type="primary", disabled=uploaded_pdf is None)

        if run_clicked and uploaded_pdf is not None:
            os.makedirs(UPLOADS_DIR, exist_ok=True)
            pdf_path = os.path.join(UPLOADS_DIR, uploaded_pdf.name)
            with open(pdf_path, "wb") as f:
                f.write(uploaded_pdf.getbuffer())

            log_lines: list[str] = []
            results_path = None
            error = None
            with st.status("Running evaluation…", expanded=True) as status:
                log_box = st.empty()
                for ev in run_pipeline(pdf_path):
                    if ev["type"] == "phase":
                        log_lines.append(f"\n=== {ev['name']} ===")
                        status.update(label=ev["name"])
                    elif ev["type"] == "log":
                        log_lines.append(ev["line"])
                    elif ev["type"] == "error":
                        error = ev["message"]
                        break
                    elif ev["type"] == "done":
                        results_path = ev["results_path"]
                    log_box.code("\n".join(log_lines[-250:]))

                if error:
                    status.update(label="Evaluation failed", state="error")
                elif results_path:
                    status.update(label="Evaluation complete", state="complete")

            if error:
                st.error(error)
            elif results_path:
                st.session_state["active_results_path"] = results_path
                st.success("Done — loading results below.")
                st.rerun()


def resolve_data(uploaded_results):
    """Pick which results to display, by priority."""
    if uploaded_results is not None:
        return json.load(uploaded_results)

    active = st.session_state.get("active_results_path")
    if active and os.path.exists(active):
        return load_results(active)

    cli_path = parse_args()
    results_path = cli_path or find_latest_results()
    if results_path and os.path.exists(results_path):
        return load_results(results_path)
    return None


def render_stage1(stage1, top3_keys):
    st.header("Stage 1 — All 9 Combinations")
    rows = [
        {
            "Combination": combo,
            "Context Precision": round(v["mean_context_precision"], 3),
            "Context Recall": round(v["mean_context_recall"], 3),
            "Passed to Stage 2": "✓" if combo in top3_keys else "",
        }
        for combo, v in stage1.items()
    ]
    df1 = pd.DataFrame(rows).sort_values("Context Precision", ascending=False).reset_index(drop=True)

    def highlight_top3(row):
        return ["background-color: #d4edda"] * len(row) if row["Passed to Stage 2"] == "✓" else [""] * len(row)

    st.dataframe(df1.style.apply(highlight_top3, axis=1), use_container_width=True)


def render_stage2(stage2, winner):
    st.header("Stage 2 — Top 3 Combinations (scored vs golden ground-truth)")
    rows = [
        {
            "Combination": combo,
            "Answer Correctness": round(v["mean_answer_correctness"], 3),
            "Answer Similarity": round(v["mean_answer_similarity"], 3),
            "Best": "🏆" if combo == winner else "",
        }
        for combo, v in sorted(stage2.items(), key=lambda kv: kv[1]["mean_answer_correctness"], reverse=True)
    ]
    df2 = pd.DataFrame(rows)

    def highlight_winner(row):
        return ["background-color: #d4edda"] * len(row) if row["Best"] == "🏆" else [""] * len(row)

    st.dataframe(df2.style.apply(highlight_winner, axis=1), use_container_width=True)


def render_per_type(valid_q):
    """Accuracy broken down by question_type — proves every type was covered."""
    agg: dict[str, dict] = {}
    for q in valid_q:
        t = q.get("question_type", "unknown")
        d = agg.setdefault(t, {"n": 0, "corr": 0.0, "sim": 0.0})
        d["n"] += 1
        d["corr"] += q.get("answer_correctness", 0.0)
        d["sim"] += q.get("answer_similarity", 0.0)
    if not agg:
        return
    rows = [
        {
            "Question Type": t,
            "Questions": d["n"],
            "Mean Correctness": round(d["corr"] / d["n"], 3),
            "Mean Similarity": round(d["sim"] / d["n"], 3),
        }
        for t, d in sorted(agg.items())
    ]
    st.subheader("Accuracy by question type")
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def render_drilldown(stage2):
    st.header("Per-Question Drill-Down")
    combo_choice = st.selectbox("Select combination", options=list(stage2.keys()))
    if not combo_choice:
        return
    per_q = stage2[combo_choice].get("per_question", [])
    valid_q = [q for q in per_q if q is not None]
    if not valid_q:
        st.info("No per-question data available for this combination.")
        return

    render_per_type(valid_q)

    df_q = pd.DataFrame([
        {
            "Question": q["question"][:80],
            "Type": q.get("question_type", "unknown"),
            "Answer Correctness": round(q.get("answer_correctness", 0), 3),
            "Answer Similarity": round(q.get("answer_similarity", 0), 3),
        }
        for q in valid_q
    ])

    def color_rows(row):
        f = row["Answer Correctness"]
        if f > 0.8:
            return ["background-color: #d4edda"] * len(row)
        elif f >= 0.5:
            return ["background-color: #fff3cd"] * len(row)
        return ["background-color: #f8d7da"] * len(row)

    selection = st.dataframe(
        df_q.style.apply(color_rows, axis=1),
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
    )
    selected_rows = selection.selection.rows if hasattr(selection, "selection") else []
    if selected_rows:
        q = valid_q[selected_rows[0]]
        st.subheader("Question Detail")
        st.write(f"**Question:** {q['question']}")
        st.write(f"**Type:** {q.get('question_type', 'unknown')}")
        st.write(f"**Generated Answer:** {q.get('generated_answer', 'N/A')}")
        st.write(f"**Ground Truth:** {q.get('ground_truth', 'N/A')}")


def main():
    st.set_page_config(page_title="RAG Eval Results", layout="wide")
    st.title("RAG Pipeline Evaluation Results")

    # --- Sidebar ---
    with st.sidebar:
        st.header("Settings")
        uploaded_results = st.file_uploader("Load a results JSON", type="json")
        st.subheader("Config")
        st.write(f"Chunk size: {CHUNK_SIZE} tokens")
        st.write(f"Embedding model: {EMBEDDING_MODEL}")
        st.write(f"Reranker: {COHERE_RERANK_MODEL}")
        st.write(f"Sample per type: {SAMPLE_PER_TYPE}")

    # --- New-PDF evaluation panel ---
    render_run_panel()

    # --- Resolve which results to display ---
    data = resolve_data(uploaded_results)
    if data is None:
        st.info("Upload a PDF above to run an evaluation, or load a results JSON from the sidebar.")
        st.stop()
        return

    meta = data.get("metadata", {})
    st.subheader(
        f"PDF: {meta.get('pdf_name', 'unknown')} | "
        f"Evaluated: {meta.get('evaluated_at', '?')} | "
        f"Questions: {meta.get('total_questions', '?')}"
    )
    st.divider()

    stage1 = data.get("stage1_results", {})
    stage2 = data.get("stage2_results", {})
    winner = data.get("winner", {}).get("combination", "")
    top3_keys = set(stage2.keys())

    render_stage1(stage1, top3_keys)
    st.divider()
    render_stage2(stage2, winner)
    st.divider()

    st.header("Winner")
    winner_data = data.get("winner", {})
    st.markdown(f"### {winner_data.get('combination', 'N/A')}")
    st.write(winner_data.get("reason", ""))
    st.divider()

    render_drilldown(stage2)


if __name__ == "__main__":
    main()

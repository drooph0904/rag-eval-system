# rag_eval_phase2/ui/app.py
import sys
import os
import json
import glob
import argparse
import subprocess

import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import RESULTS_DIR, CHUNK_SIZE, EMBEDDING_MODEL, COHERE_RERANK_MODEL


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


def main():
    st.set_page_config(page_title="RAG Eval Results", layout="wide")
    st.title("RAG Pipeline Evaluation Results")

    # --- Sidebar ---
    with st.sidebar:
        st.header("Settings")
        uploaded = st.file_uploader("Load results JSON", type="json")
        if uploaded:
            data = json.load(uploaded)
        else:
            cli_path = parse_args()
            results_path = cli_path or find_latest_results()
            if results_path and os.path.exists(results_path):
                data = load_results(results_path)
            else:
                st.warning("No results file found. Run main.py first.")
                st.stop()
                return

        st.subheader("Config")
        st.write(f"Chunk size: {CHUNK_SIZE} tokens")
        st.write(f"Embedding model: {EMBEDDING_MODEL}")
        st.write(f"Reranker: {COHERE_RERANK_MODEL}")

        main_py = os.path.join(os.path.dirname(__file__), "..", "main.py")
        if st.button("Re-run Evaluation"):
            result = subprocess.run(["pgrep", "-f", "main.py"], capture_output=True)
            if result.returncode == 0:
                st.warning("main.py is already running.")
            else:
                st.info("Start main.py from your terminal: python main.py <pdf> <golden>")

    # --- Header ---
    meta = data.get("metadata", {})
    st.subheader(f"PDF: {meta.get('pdf_name', 'unknown')} | Evaluated: {meta.get('evaluated_at', '?')} | Questions: {meta.get('total_questions', '?')}")

    st.divider()

    # --- Stage 1 table ---
    st.header("Stage 1 — All 9 Combinations")
    stage1 = data.get("stage1_results", {})
    stage2 = data.get("stage2_results", {})
    top3_keys = set(stage2.keys())

    rows = []
    for combo, v in stage1.items():
        rows.append({
            "Combination": combo,
            "Context Precision": round(v["mean_context_precision"], 3),
            "Context Recall": round(v["mean_context_recall"], 3),
            "Passed to Stage 2": "✓" if combo in top3_keys else "",
        })
    df1 = pd.DataFrame(rows).sort_values("Context Precision", ascending=False).reset_index(drop=True)

    def highlight_top3(row):
        if row["Passed to Stage 2"] == "✓":
            return ["background-color: #d4edda"] * len(row)
        return [""] * len(row)

    st.dataframe(df1.style.apply(highlight_top3, axis=1), use_container_width=True)

    st.divider()

    # --- Stage 2 cards ---
    st.header("Stage 2 — Top 3 Combinations (scored vs golden ground-truth)")
    winner = data.get("winner", {}).get("combination", "")
    cols = st.columns(len(stage2))
    for col, (combo, v) in zip(cols, sorted(stage2.items(), key=lambda kv: kv[1]["mean_answer_correctness"], reverse=True)):
        with col:
            badge = " 🏆 BEST" if combo == winner else ""
            st.subheader(f"{combo}{badge}")
            st.progress(v["mean_answer_correctness"], text=f"Answer Correctness: {v['mean_answer_correctness']:.2f}")
            st.progress(v["mean_answer_similarity"], text=f"Answer Similarity: {v['mean_answer_similarity']:.2f}")

    st.divider()

    # --- Winner announcement ---
    st.header("Winner")
    winner_data = data.get("winner", {})
    st.markdown(f"### {winner_data.get('combination', 'N/A')}")
    st.write(winner_data.get("reason", ""))

    st.divider()

    # --- Per-question drill-down ---
    st.header("Per-Question Drill-Down")
    combo_choice = st.selectbox("Select combination", options=list(stage2.keys()))
    if combo_choice:
        per_q = stage2[combo_choice].get("per_question", [])
        valid_q = [q for q in per_q if q is not None]
        if valid_q:
            df_q = pd.DataFrame([
                {
                    "Question": q["question"][:80],
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
                idx = selected_rows[0]
                q = valid_q[idx]
                st.subheader("Question Detail")
                st.write(f"**Question:** {q['question']}")
                st.write(f"**Generated Answer:** {q.get('generated_answer', 'N/A')}")
                st.write(f"**Ground Truth:** {q.get('ground_truth', 'N/A')}")
        else:
            st.info("No per-question data available for this combination.")


if __name__ == "__main__":
    main()

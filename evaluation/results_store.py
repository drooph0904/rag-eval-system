# rag_eval_phase2/evaluation/results_store.py
import json
import os
from datetime import datetime
from config import RESULTS_DIR


class ResultsStore:
    def __init__(self, results_dir: str = RESULTS_DIR):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    def save(self, pdf_name: str, golden_dataset_path: str, stage1_results: dict, stage2_results: dict, total_questions: int) -> str:
        winner = self._determine_winner(stage2_results)
        data = {
            "metadata": {
                "pdf_name": pdf_name,
                "golden_dataset_path": golden_dataset_path,
                "evaluated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_combinations": 9,
                "total_questions": total_questions,
            },
            "stage1_results": stage1_results,
            "stage2_results": stage2_results,
            "winner": winner,
        }
        stem = os.path.splitext(pdf_name)[0]
        path = os.path.join(self.results_dir, f"{stem}_eval_results.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Results saved to {path}")
        return path

    def load(self, path: str) -> dict:
        with open(path) as f:
            return json.load(f)

    def _determine_winner(self, stage2_results: dict) -> dict:
        if not stage2_results:
            return {"combination": "none", "reason": "No Stage 2 results available"}
        best = max(stage2_results.items(), key=lambda kv: kv[1].get("mean_answer_correctness", 0.0))
        combo, scores = best
        return {
            "combination": combo,
            "reason": f"highest mean answer correctness vs golden ground-truth in Stage 2 ({scores['mean_answer_correctness']:.2f})",
        }

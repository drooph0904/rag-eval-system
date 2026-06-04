import os
import json
import math
import logging
from datetime import datetime
from pathlib import Path
from openai import OpenAI
from config import OUTPUT_DIR

logger = logging.getLogger(__name__)


class DatasetManager:
    def __init__(self, output_dir: str = OUTPUT_DIR):
        self.output_dir = output_dir
        self._embed_client = None

    def save(self, questions: list[dict], pdf_path: str, sections_count: int, model: str) -> str:
        questions = self._deduplicate(questions)
        questions = [{"id": f"q{i + 1:03d}", **q} for i, q in enumerate(questions)]

        by_type: dict[str, int] = {}
        for q in questions:
            t = q.get("question_type", "unknown")
            by_type[t] = by_type.get(t, 0) + 1

        dataset = {
            "metadata": {
                "source_pdf": Path(pdf_path).name,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_sections": sections_count,
                "total_questions": len(questions),
                "questions_by_type": by_type,
                "model_used": model,
            },
            "questions": questions,
        }

        os.makedirs(self.output_dir, exist_ok=True)
        out_path = os.path.join(self.output_dir, f"{Path(pdf_path).stem}_golden.json")
        with open(out_path, "w") as f:
            json.dump(dataset, f, indent=2)

        logger.info(f"Saved {len(questions)} questions → {out_path}")
        logger.info(f"By type: {by_type}")
        return out_path

    def load(self, dataset_path: str) -> dict:
        with open(dataset_path) as f:
            return json.load(f)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if self._embed_client is None:
            self._embed_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = self._embed_client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )
        return [item.embedding for item in response.data]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(y * y for y in b))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    def _deduplicate(self, questions: list[dict]) -> list[dict]:
        if len(questions) <= 1:
            return questions

        texts = [q.get("question", "") for q in questions]
        embeddings = self._embed(texts)

        unique: list[dict] = []
        unique_embeddings: list[list[float]] = []

        for q, emb in zip(questions, embeddings):
            text = q.get("question", "")
            is_dup = any(
                self._cosine_similarity(emb, u_emb) > 0.85
                for u_emb in unique_embeddings
            )
            if is_dup:
                logger.info(f"Removed duplicate: {text[:70]!r}")
            else:
                unique.append(q)
                unique_embeddings.append(emb)

        return unique


if __name__ == "__main__":
    import logging as log
    log.basicConfig(level=log.INFO)
    sample = [
        {
            "question": "What is RAG?",
            "ground_truth": "RAG is retrieval-augmented generation, combining retrieval with generation.",
            "question_type": "definition",
            "source_section": 1,
        }
    ]
    manager = DatasetManager()
    path = manager.save(sample, "test.pdf", sections_count=3, model="gpt-4o-mini")
    print(f"Saved to: {path}")
    print(manager.load(path))

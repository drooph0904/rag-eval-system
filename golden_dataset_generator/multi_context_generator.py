import os
import json
import time
import logging
from openai import OpenAI
from dotenv import load_dotenv
from config import LLM_MODEL, TEMPERATURE, MAX_RETRIES, MIN_ANSWER_LENGTH, MULTI_CONTEXT_QUESTIONS_PER_PAIR
from prompts import MULTI_CONTEXT_PROMPT

load_dotenv()
logger = logging.getLogger(__name__)

_VISUAL_REFS = {"row", "rows", "column", "columns", "figure", "figures",
                "table", "tables", "above", "below", "shown", "diagram"}
_STOP_WORDS  = frozenset({
    "what", "is", "are", "the", "a", "an", "of", "in", "to", "for",
    "does", "do", "did", "how", "why", "where", "when", "which", "who",
    "difference", "between", "and", "or", "but", "its", "it", "this",
    "that", "these", "those", "was", "were", "has", "have", "been",
    "being", "be", "with", "by", "at", "from", "as", "on", "not",
    "used", "can", "could", "would", "will", "there", "their", "they",
})


class MultiContextGenerator:
    def __init__(self):
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate_for_pair(self, section_a: dict, section_b: dict) -> list[dict]:
        prompt = MULTI_CONTEXT_PROMPT.format(
            num_questions=MULTI_CONTEXT_QUESTIONS_PER_PAIR,
            section_id_a=section_a["section_id"],
            section_text_a=section_a["text"],
            section_id_b=section_b["section_id"],
            section_text_b=section_b["text"],
        )
        for attempt in range(MAX_RETRIES):
            try:
                response = self._client.chat.completions.create(
                    model=LLM_MODEL,
                    temperature=TEMPERATURE,
                    response_format={"type": "json_object"},
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = json.loads(response.choices[0].message.content)
                questions = raw.get("questions", raw) if isinstance(raw, dict) else raw
                return self._validate(
                    questions,
                    section_a["section_id"],
                    section_b["section_id"],
                )
            except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as exc:
                logger.warning(
                    f"Pair ({section_a['section_id']},{section_b['section_id']}) "
                    f"attempt {attempt + 1}/{MAX_RETRIES} failed: {exc}"
                )
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2)

        logger.error(
            f"All retries failed for pair "
            f"({section_a['section_id']},{section_b['section_id']})"
        )
        return []

    def _validate(self, questions: list, id_a: int, id_b: int) -> list[dict]:
        valid = []
        for q in questions:
            q_text = q.get("question", "")
            if not isinstance(q_text, str) or not q_text.endswith("?"):
                logger.debug(f"MC dropped (no '?'): {q_text[:60]!r}")
                continue

            gt = q.get("ground_truth", "")
            if not isinstance(gt, str) or len(gt) <= MIN_ANSWER_LENGTH:
                logger.debug(f"MC dropped (short/invalid ground_truth): {q_text[:60]!r}")
                continue

            if len(gt) > 300:
                logger.debug(f"MC dropped (answer too long): {q_text[:60]!r}")
                continue

            if set(gt.lower().split()) & _VISUAL_REFS:
                logger.debug(f"MC dropped (visual reference): {q_text[:60]!r}")
                continue

            if q.get("question_type") != "multi_context":
                logger.debug(f"MC dropped (wrong type '{q.get('question_type')}'): {q_text[:60]!r}")
                continue

            # Ensure source_sections references both sections
            src = q.get("source_sections", [])
            if not isinstance(src, list) or len(src) < 2:
                q["source_sections"] = [id_a, id_b]

            valid.append(q)

        dropped = len(questions) - len(valid)
        if dropped:
            logger.info(f"Pair ({id_a},{id_b}): dropped {dropped} invalid multi-context questions")
        return valid


if __name__ == "__main__":
    import json as _json
    import logging as log
    log.basicConfig(level=log.INFO)
    sample_a = {
        "section_id": 1,
        "text": (
            "The Transformer uses multi-head attention with 8 parallel heads. "
            "Each head operates on a dimension of 64, giving a total model dimension of 512. "
            "The encoder consists of 6 identical layers stacked on top of each other."
        ),
    }
    sample_b = {
        "section_id": 2,
        "text": (
            "Training the base Transformer model took 12 hours on 8 GPUs. "
            "The model was trained for 100,000 steps with a batch size of 25,000 tokens. "
            "The big model was trained for 300,000 steps and took 3.5 days."
        ),
    }
    gen = MultiContextGenerator()
    result = gen.generate_for_pair(sample_a, sample_b)
    print(_json.dumps(result, indent=2))

import os
import json
import time
import logging
from openai import OpenAI
from dotenv import load_dotenv
from config import LLM_MODEL, TEMPERATURE, MAX_RETRIES, MIN_ANSWER_LENGTH, QUESTIONS_PER_SECTION
from prompts import QUESTION_GENERATION_PROMPT

load_dotenv()
logger = logging.getLogger(__name__)

_VALID_TYPES = {"factual", "numerical", "inferential", "definition", "comparison"}
_FIRST_PERSON = {"we", "our", "i", "us", "my", "we've", "we'll"}
_VISUAL_REFS  = {"row", "rows", "column", "columns", "figure", "figures",
                 "table", "tables", "above", "below", "shown", "diagram"}
_MAX_ANSWER_LENGTH = 300
_STOP_WORDS   = frozenset({
    "what", "is", "are", "the", "a", "an", "of", "in", "to", "for",
    "does", "do", "did", "how", "why", "where", "when", "which", "who",
    "difference", "between", "and", "or", "but", "its", "it", "this",
    "that", "these", "those", "was", "were", "has", "have", "been",
    "being", "be", "with", "by", "at", "from", "as", "on", "not",
    "used", "can", "could", "would", "will", "there", "their", "they",
})


class QuestionGenerator:
    def __init__(self):
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate(self, section: dict) -> list[dict]:
        prompt = QUESTION_GENERATION_PROMPT.format(
            section_id=section["section_id"],
            section_text=section["text"],
            num_questions=QUESTIONS_PER_SECTION,
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
                return self._validate(questions, section["section_id"])
            except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as exc:
                logger.warning(
                    f"Section {section['section_id']} attempt {attempt + 1}/{MAX_RETRIES} failed: {exc}"
                )
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2)

        logger.error(f"All {MAX_RETRIES} retries failed for section {section['section_id']}")
        return []

    def _validate(self, questions: list, section_id: int) -> list[dict]:
        valid = []
        for q in questions:
            q_text = q.get("question", "")
            if not isinstance(q_text, str) or not q_text.endswith("?"):
                logger.debug(f"Dropped (no '?'): {q_text[:60]!r}")
                continue

            q_words = set(q_text.lower().split())

            if q_words & _FIRST_PERSON:
                logger.debug(f"Dropped (author-perspective): {q_text[:60]!r}")
                continue

            gt = q.get("ground_truth", "")
            if not isinstance(gt, str):
                logger.debug(f"Dropped (non-string ground_truth): {q_text[:60]!r}")
                continue
            min_len = 8 if q.get("question_type") == "numerical" else MIN_ANSWER_LENGTH
            if len(gt) <= min_len:
                logger.debug(f"Dropped (short ground_truth): {q_text[:60]!r}")
                continue
            if len(gt) > _MAX_ANSWER_LENGTH:
                logger.debug(f"Dropped (answer too long, {len(gt)} chars): {q_text[:60]!r}")
                continue

            if q.get("question_type") not in _VALID_TYPES:
                logger.debug(f"Dropped (invalid type {q.get('question_type')!r}): {q_text[:60]!r}")
                continue

            gt_words_set = set(gt.lower().split())
            if gt_words_set & _VISUAL_REFS:
                logger.debug(f"Dropped (visual reference): {q_text[:60]!r}")
                continue

            if q.get("question_type") != "numerical" and self._is_circular(q_text, gt):
                logger.debug(f"Dropped (circular answer): {q_text[:60]!r}")
                continue

            valid.append(q)

        dropped = len(questions) - len(valid)
        if dropped:
            logger.info(f"Section {section_id}: dropped {dropped} invalid questions")
        return valid

    @staticmethod
    def _is_circular(question: str, ground_truth: str) -> bool:
        import re as _re
        _strip = lambda s: _re.sub(r"[^\w\s]", "", s.lower())
        q_content = {w for w in _strip(question).split() if w not in _STOP_WORDS}
        gt_content = [w for w in _strip(ground_truth).split() if w not in _STOP_WORDS]
        if not gt_content or not q_content:
            return False
        # Use soft matching: a gt word matches if it shares a root with a q word
        def overlaps(w: str) -> bool:
            return any(w == qw or (len(w) >= 5 and (w.startswith(qw[:5]) or qw.startswith(w[:5])))
                       for qw in q_content)
        overlap_ratio = sum(1 for w in gt_content if overlaps(w)) / len(gt_content)
        return overlap_ratio > 0.4


if __name__ == "__main__":
    import json as _json
    import logging as log
    log.basicConfig(level=log.INFO)
    sample = {
        "section_id": 1,
        "text": (
            "RAG stands for retrieval-augmented generation. "
            "Three retrieval pipelines were evaluated. Dense retrieval achieved 0.87 precision "
            "while sparse retrieval scored 0.72. The minimum recommended chunk size is 100 words. "
            "Precision is the ratio of relevant retrieved documents to total retrieved documents."
        ),
        "word_count": 55,
        "source_pages": "1",
    }
    gen = QuestionGenerator()
    result = gen.generate(sample)
    print(_json.dumps(result, indent=2))

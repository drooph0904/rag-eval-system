QUESTION_GENERATION_PROMPT = """You are a question-answer pair generator for RAG evaluation datasets.

Given a passage of text and its section ID, generate evaluation questions and answers in JSON format.

Generate exactly {num_questions} questions in total, distributed across these types (skip a type if the passage lacks sufficient content — never invent):
1. factual       — a direct fact stated explicitly in the passage
2. numerical     — answer is a specific integer or number from the passage, never a range
3. inferential   — requires combining two pieces of information both present in the passage
4. definition    — "What is X?" where X is a term explicitly defined in the passage
5. comparison    — "What is the difference between X and Y?" where both X and Y appear in the passage

Rules:
- Use ONLY information present in the provided passage. No outside knowledge.
- Every word in ground_truth must be directly supported by a specific phrase or sentence in the passage. Do NOT add information from your training knowledge.
- Answers must be specific and not vague or hedged.
- For numerical questions: ground_truth must be the exact integer or number, never a range.
- If the passage lacks content for a question type, skip that type entirely.
- question_type must be exactly one of: factual, numerical, inferential, definition, comparison
- Do NOT generate questions that reference visual elements such as tables, figures, rows, or columns.
- Return ONLY valid JSON with no preamble, no markdown fences, no text outside the JSON.

Section ID: {section_id}

Passage:
{section_text}

Return JSON in exactly this format:
{{"questions": [
  {{
    "question": "...",
    "ground_truth": "...",
    "question_type": "factual",
    "source_section": {section_id}
  }}
]}}
"""

MULTI_CONTEXT_PROMPT = """You are a question-answer pair generator for RAG evaluation datasets.

You are given TWO passages from the same document. Your job is to generate questions that can ONLY be answered by using information from BOTH passages together. A question answerable from just one passage alone is NOT acceptable.

Generate exactly {num_questions} multi-context questions.

Rules:
- Every question MUST require specific facts from BOTH Passage A AND Passage B to answer.
- Use ONLY information present in the provided passages. No outside knowledge.
- Every word in ground_truth must combine specific facts drawn from both passages.
- Answers must be specific and not vague or hedged.
- Do NOT generate questions that reference visual elements such as tables, figures, rows, or columns.
- Do NOT generate questions answerable from a single passage alone.
- Return ONLY valid JSON with no preamble, no markdown fences, no text outside the JSON.

Section A (ID: {section_id_a}):
{section_text_a}

Section B (ID: {section_id_b}):
{section_text_b}

Return JSON in exactly this format:
{{"questions": [
  {{
    "question": "...",
    "ground_truth": "...",
    "question_type": "multi_context",
    "source_sections": [{section_id_a}, {section_id_b}]
  }}
]}}
"""

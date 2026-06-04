import config
import prompts


def test_config_constants():
    assert config.SECTION_MAX_WORDS == 500
    assert config.SECTION_MIN_WORDS == 50
    assert config.QUESTIONS_PER_SECTION == 8
    assert config.MIN_ANSWER_LENGTH == 15
    assert config.MAX_RETRIES == 3
    assert config.LLM_MODEL == "gpt-4o-mini"
    assert config.TEMPERATURE == 0.7
    assert config.OUTPUT_DIR == "./output"
    assert config.LOG_LEVEL == "INFO"


def test_prompt_covers_all_question_types():
    p = prompts.QUESTION_GENERATION_PROMPT
    for q_type in ["factual", "numerical", "inferential", "definition", "comparison"]:
        assert q_type in p.lower(), f"Prompt missing question type: {q_type}"


def test_prompt_requires_json_output():
    assert "json" in prompts.QUESTION_GENERATION_PROMPT.lower()


def test_prompt_has_format_placeholders():
    assert "{section_id}" in prompts.QUESTION_GENERATION_PROMPT
    assert "{section_text}" in prompts.QUESTION_GENERATION_PROMPT
    assert "{num_questions}" in prompts.QUESTION_GENERATION_PROMPT

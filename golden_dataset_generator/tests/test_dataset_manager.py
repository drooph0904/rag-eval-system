import os
import json
import pytest
from unittest.mock import patch
from dataset_manager import DatasetManager

_QUESTIONS = [
    {
        "question": "What is the minimum chunk size recommended?",
        "ground_truth": "The minimum chunk size recommended is 100 words based on the specification.",
        "question_type": "factual",
        "source_section": 1,
    },
    {
        "question": "How many pipelines were tested?",
        "ground_truth": "Exactly three retrieval pipelines were tested in the evaluation.",
        "question_type": "numerical",
        "source_section": 2,
    },
]


def _orthogonal(n, dim=16):
    """n orthogonal unit vectors — cosine similarity between any two is 0."""
    vecs = []
    for i in range(n):
        v = [0.0] * dim
        v[i % dim] = 1.0
        vecs.append(v)
    return vecs


def _manager(tmp_path):
    return DatasetManager(output_dir=str(tmp_path))


@pytest.fixture(autouse=True)
def mock_embed(monkeypatch):
    """Default: each question gets a unique orthogonal vector → no deduplication."""
    def fake_embed(self, texts):
        return _orthogonal(len(texts))
    monkeypatch.setattr(DatasetManager, "_embed", fake_embed)


def test_save_creates_file(tmp_path):
    path = _manager(tmp_path).save(_QUESTIONS, "test.pdf", sections_count=5, model="gpt-4o-mini")
    assert os.path.exists(path)


def test_save_output_filename_uses_pdf_stem(tmp_path):
    path = _manager(tmp_path).save(_QUESTIONS, "/some/path/my_doc.pdf", sections_count=5, model="gpt-4o-mini")
    assert os.path.basename(path) == "my_doc_golden.json"


def test_save_json_has_metadata_and_questions(tmp_path):
    path = _manager(tmp_path).save(_QUESTIONS, "test.pdf", sections_count=5, model="gpt-4o-mini")
    with open(path) as f:
        data = json.load(f)
    assert "metadata" in data
    assert "questions" in data


def test_save_metadata_fields(tmp_path):
    path = _manager(tmp_path).save(_QUESTIONS, "test.pdf", sections_count=5, model="gpt-4o-mini")
    with open(path) as f:
        meta = json.load(f)["metadata"]
    assert meta["source_pdf"] == "test.pdf"
    assert meta["total_sections"] == 5
    assert meta["total_questions"] == 2
    assert meta["model_used"] == "gpt-4o-mini"
    assert meta["questions_by_type"]["factual"] == 1
    assert meta["questions_by_type"]["numerical"] == 1
    assert "generated_at" in meta


def test_save_adds_sequential_ids(tmp_path):
    path = _manager(tmp_path).save(_QUESTIONS, "test.pdf", sections_count=5, model="gpt-4o-mini")
    with open(path) as f:
        qs = json.load(f)["questions"]
    assert qs[0]["id"] == "q001"
    assert qs[1]["id"] == "q002"


def test_save_deduplicates_identical_questions(tmp_path, monkeypatch):
    # q1 and q3 are identical → same vector. q2 is different → orthogonal vector.
    def fake_embed(self, texts):
        return [
            [1.0, 0.0, 0.0, 0.0],  # q1 (_QUESTIONS[0])
            [0.0, 1.0, 0.0, 0.0],  # q2 (_QUESTIONS[1]) — different
            [1.0, 0.0, 0.0, 0.0],  # q3 (duplicate of q1) — dropped
        ][:len(texts)]
    monkeypatch.setattr(DatasetManager, "_embed", fake_embed)

    dupes = _QUESTIONS + [_QUESTIONS[0]]
    path = _manager(tmp_path).save(dupes, "test.pdf", sections_count=5, model="gpt-4o-mini")
    with open(path) as f:
        qs = json.load(f)["questions"]
    assert len(qs) == 2


def test_save_deduplicates_semantically_similar_questions(tmp_path, monkeypatch):
    # q1 and q2 are semantically near-identical (similarity ~0.99), q3 is different
    def fake_embed(self, texts):
        embeddings = [
            [1.0, 0.0, 0.0, 0.0],   # q1
            [0.999, 0.045, 0.0, 0.0],  # q2 — nearly identical to q1 (sim > 0.85)
            [0.0, 0.0, 1.0, 0.0],   # q3 — orthogonal to q1
        ]
        return embeddings[:len(texts)]
    monkeypatch.setattr(DatasetManager, "_embed", fake_embed)

    near_dup = dict(_QUESTIONS[0])
    near_dup["question"] = "What is the minimum chunk size that is recommended?"
    questions = [_QUESTIONS[0], near_dup, _QUESTIONS[1]]
    path = _manager(tmp_path).save(questions, "test.pdf", sections_count=5, model="gpt-4o-mini")
    with open(path) as f:
        qs = json.load(f)["questions"]
    assert len(qs) == 2  # near-dup dropped, different kept


def test_load_returns_saved_data(tmp_path):
    manager = _manager(tmp_path)
    path = manager.save(_QUESTIONS, "test.pdf", sections_count=5, model="gpt-4o-mini")
    loaded = manager.load(path)
    assert loaded["metadata"]["total_questions"] == 2
    assert loaded["questions"][0]["id"] == "q001"


def test_save_creates_output_dir_if_missing(tmp_path):
    new_dir = str(tmp_path / "deep" / "nested")
    manager = DatasetManager(output_dir=new_dir)
    path = manager.save(_QUESTIONS, "test.pdf", sections_count=5, model="gpt-4o-mini")
    assert os.path.exists(path)


def test_cosine_similarity_identical_vectors():
    sim = DatasetManager._cosine_similarity([1.0, 0.0], [1.0, 0.0])
    assert abs(sim - 1.0) < 1e-6


def test_cosine_similarity_orthogonal_vectors():
    sim = DatasetManager._cosine_similarity([1.0, 0.0], [0.0, 1.0])
    assert abs(sim - 0.0) < 1e-6


def test_cosine_similarity_zero_vector():
    sim = DatasetManager._cosine_similarity([0.0, 0.0], [1.0, 0.0])
    assert sim == 0.0

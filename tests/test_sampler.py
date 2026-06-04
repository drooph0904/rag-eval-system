import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation.sampler import stratified_sample


def make_q(i, qtype):
    return {"id": f"q{i:03d}", "question": f"Q{i}", "ground_truth": f"A{i}", "question_type": qtype}


def test_caps_each_type_to_per_type():
    qs = [make_q(i, "factual") for i in range(25)] + [make_q(100 + i, "inferential") for i in range(25)]
    out = stratified_sample(qs, per_type=10)
    by_type = {}
    for q in out:
        by_type[q["question_type"]] = by_type.get(q["question_type"], 0) + 1
    assert by_type == {"factual": 10, "inferential": 10}


def test_includes_every_type():
    types = ["factual", "inferential", "numerical", "definition", "multi_context"]
    qs = [make_q(i, t) for t, group in zip(types, range(5)) for i in range(group * 100, group * 100 + 3)]
    out = stratified_sample(qs, per_type=10)
    assert set(q["question_type"] for q in out) == set(types)


def test_type_with_fewer_than_cap_takes_all():
    qs = [make_q(i, "factual") for i in range(3)] + [make_q(100 + i, "rare") for i in range(2)]
    out = stratified_sample(qs, per_type=10)
    by_type = {}
    for q in out:
        by_type[q["question_type"]] = by_type.get(q["question_type"], 0) + 1
    assert by_type == {"factual": 3, "rare": 2}


def test_stable_order_takes_first_n_per_type():
    qs = [make_q(i, "factual") for i in range(20)]
    out = stratified_sample(qs, per_type=5)
    assert [q["id"] for q in out] == [f"q{i:03d}" for i in range(5)]


def test_empty_input_returns_empty():
    assert stratified_sample([], per_type=10) == []


def test_missing_question_type_grouped_as_unknown():
    qs = [{"id": "q1", "question": "Q", "ground_truth": "A"}]  # no question_type
    out = stratified_sample(qs, per_type=10)
    assert len(out) == 1
    assert out[0]["id"] == "q1"


def test_per_type_zero_returns_empty():
    qs = [make_q(i, "factual") for i in range(5)]
    assert stratified_sample(qs, per_type=0) == []

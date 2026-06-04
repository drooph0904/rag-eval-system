import sys, os, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pipeline_runner as pr


def test_golden_output_path_naming():
    p = pr.golden_output_path("/some/dir/My Doc.pdf")
    assert p.endswith(os.path.join("output", "My Doc_golden.json"))


def test_sampled_output_path_naming():
    p = pr.sampled_output_path("/some/dir/My Doc.pdf")
    assert p.endswith(os.path.join("output", "My Doc_golden_sampled.json"))


def test_results_path_naming():
    p = pr.results_path_for("/some/dir/My Doc.pdf")
    assert p.endswith(os.path.join("results", "My Doc_eval_results.json"))


def _write_golden(path, questions):
    with open(path, "w") as f:
        json.dump({"metadata": {"pdf_name": "x.pdf"}, "questions": questions}, f)


def test_write_sampled_golden_caps_per_type_and_preserves_metadata():
    with tempfile.TemporaryDirectory() as d:
        golden = os.path.join(d, "g.json")
        sampled = os.path.join(d, "g_sampled.json")
        qs = ([{"id": f"f{i}", "question": "Q", "ground_truth": "A", "question_type": "factual"} for i in range(15)] +
              [{"id": f"n{i}", "question": "Q", "ground_truth": "A", "question_type": "numerical"} for i in range(4)])
        _write_golden(golden, qs)

        count = pr.write_sampled_golden(golden, sampled, per_type=10)

        assert count == 14  # 10 factual + 4 numerical
        with open(sampled) as f:
            data = json.load(f)
        assert "metadata" in data
        by_type = {}
        for q in data["questions"]:
            by_type[q["question_type"]] = by_type.get(q["question_type"], 0) + 1
        assert by_type == {"factual": 10, "numerical": 4}


def test_write_sampled_golden_returns_zero_for_empty():
    with tempfile.TemporaryDirectory() as d:
        golden = os.path.join(d, "g.json")
        sampled = os.path.join(d, "g_sampled.json")
        _write_golden(golden, [])
        count = pr.write_sampled_golden(golden, sampled, per_type=10)
        assert count == 0

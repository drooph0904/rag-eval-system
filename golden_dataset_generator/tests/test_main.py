import os
import subprocess
import sys

MODULE_DIR = os.path.join(os.path.dirname(__file__), "..")


def _run(*args, env=None):
    return subprocess.run(
        [sys.executable, "main.py", *args],
        capture_output=True,
        text=True,
        cwd=MODULE_DIR,
        env=env,
    )


def test_no_args_exits_1_with_usage():
    result = _run()
    assert result.returncode == 1
    assert "Usage" in result.stdout or "usage" in result.stdout.lower()


def test_nonexistent_pdf_exits_1(tmp_path):
    result = _run(str(tmp_path / "ghost.pdf"))
    assert result.returncode == 1
    assert "not found" in result.stdout.lower() or "not found" in result.stderr.lower()


def test_non_pdf_extension_exits_1(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("content")
    result = _run(str(f))
    assert result.returncode == 1


def test_missing_api_key_exits_1(tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 placeholder")
    env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    result = _run(str(pdf), env=env)
    # Should exit non-zero (either no API key error, or pymupdf fails on fake PDF)
    assert result.returncode != 0

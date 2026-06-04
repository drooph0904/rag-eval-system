import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pytest


@pytest.fixture(autouse=True)
def _clear_hyde_cache():
    """HyDE answers are cached per-process; clear between tests so each test's
    mocked OpenAI client is exercised independently."""
    try:
        from retrieval import hyde
        hyde.clear_cache()
    except Exception:
        pass
    yield

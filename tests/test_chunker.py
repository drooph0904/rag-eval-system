# rag_eval_phase2/tests/test_chunker.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from indexer.chunker import Chunker
from config import CHUNK_SIZE, CHUNK_OVERLAP, CHILD_CHUNK_SIZE

# 2000-word passage for tests — repeat a known sentence enough times
LONG_TEXT = ("The mitochondria is the powerhouse of the cell and produces ATP through oxidative phosphorylation. " * 200).strip()


class TestFixedChunking:
    def setup_method(self):
        self.chunker = Chunker()

    def test_returns_list_of_dicts(self):
        chunks = self.chunker.fixed(LONG_TEXT)
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(c, dict) for c in chunks)

    def test_chunk_has_required_keys(self):
        chunks = self.chunker.fixed(LONG_TEXT)
        required = {"chunk_id", "text", "token_count", "strategy", "char_start"}
        assert required.issubset(chunks[0].keys())

    def test_strategy_field_is_fixed(self):
        chunks = self.chunker.fixed(LONG_TEXT)
        assert all(c["strategy"] == "fixed" for c in chunks)

    def test_chunk_ids_are_unique(self):
        chunks = self.chunker.fixed(LONG_TEXT)
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_ids_prefixed_fixed(self):
        chunks = self.chunker.fixed(LONG_TEXT)
        assert all(c["chunk_id"].startswith("fixed_") for c in chunks)

    def test_no_chunk_exceeds_chunk_size(self):
        chunks = self.chunker.fixed(LONG_TEXT)
        assert all(c["token_count"] <= CHUNK_SIZE for c in chunks)

    def test_produces_multiple_chunks_for_long_text(self):
        chunks = self.chunker.fixed(LONG_TEXT)
        assert len(chunks) > 3

    def test_short_text_produces_single_chunk(self):
        short = "Hello world. This is a short passage."
        chunks = self.chunker.fixed(short)
        assert len(chunks) == 1

    def test_char_start_increases_monotonically(self):
        chunks = self.chunker.fixed(LONG_TEXT)
        starts = [c["char_start"] for c in chunks]
        assert starts == sorted(starts)


class TestSemanticChunking:
    def setup_method(self):
        self.chunker = Chunker()

    def test_returns_list_of_dicts(self):
        chunks = self.chunker.semantic(LONG_TEXT)
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_strategy_field_is_semantic(self):
        chunks = self.chunker.semantic(LONG_TEXT)
        assert all(c["strategy"] == "semantic" for c in chunks)

    def test_chunk_ids_prefixed_semantic(self):
        chunks = self.chunker.semantic(LONG_TEXT)
        assert all(c["chunk_id"].startswith("semantic_") for c in chunks)

    def test_no_chunk_exceeds_chunk_size(self):
        chunks = self.chunker.semantic(LONG_TEXT)
        assert all(c["token_count"] <= CHUNK_SIZE for c in chunks)

    def test_discards_chunks_under_50_tokens(self):
        # All chunks should be >= 50 tokens
        chunks = self.chunker.semantic(LONG_TEXT)
        assert all(c["token_count"] >= 50 for c in chunks)

    def test_chunk_ids_are_unique(self):
        chunks = self.chunker.semantic(LONG_TEXT)
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunks_do_not_cut_mid_sentence(self):
        text = "First sentence ends here. Second sentence is this. Third goes last."
        chunks = self.chunker.semantic(text)
        for chunk in chunks:
            # Each chunk text should end at a sentence boundary or be the full text
            stripped = chunk["text"].strip()
            assert stripped.endswith(".") or stripped.endswith("?") or stripped.endswith("!")


class TestParentChildChunking:
    def setup_method(self):
        self.chunker = Chunker()

    def test_returns_tuple(self):
        result = self.chunker.parent_child(LONG_TEXT)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_child_chunks_are_list_of_dicts(self):
        children, parent_map = self.chunker.parent_child(LONG_TEXT)
        assert isinstance(children, list)
        assert all(isinstance(c, dict) for c in children)

    def test_parent_map_is_dict(self):
        children, parent_map = self.chunker.parent_child(LONG_TEXT)
        assert isinstance(parent_map, dict)

    def test_child_ids_prefixed_pc_child(self):
        children, parent_map = self.chunker.parent_child(LONG_TEXT)
        assert all(c["chunk_id"].startswith("pc_child_") for c in children)

    def test_strategy_field_is_parent_child(self):
        children, parent_map = self.chunker.parent_child(LONG_TEXT)
        assert all(c["strategy"] == "parent_child" for c in children)

    def test_all_child_ids_in_parent_map(self):
        children, parent_map = self.chunker.parent_child(LONG_TEXT)
        for child in children:
            assert child["chunk_id"] in parent_map

    def test_parent_map_values_are_strings(self):
        children, parent_map = self.chunker.parent_child(LONG_TEXT)
        assert all(isinstance(v, str) for v in parent_map.values())

    def test_child_chunks_smaller_than_parent(self):
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        children, parent_map = self.chunker.parent_child(LONG_TEXT)
        for child in children:
            parent_text = parent_map[child["chunk_id"]]
            child_tokens = len(enc.encode(child["text"]))
            parent_tokens = len(enc.encode(parent_text))
            assert child_tokens <= parent_tokens

    def test_child_token_count_at_most_child_chunk_size(self):
        children, parent_map = self.chunker.parent_child(LONG_TEXT)
        assert all(c["token_count"] <= CHILD_CHUNK_SIZE for c in children)

# rag_eval_phase2/indexer/chunker.py
import tiktoken
import re
from config import CHUNK_SIZE, CHUNK_OVERLAP, CHILD_CHUNK_SIZE


class Chunker:
    def __init__(self):
        self._enc = tiktoken.get_encoding("cl100k_base")

    def _tokenize(self, text: str) -> list[int]:
        return self._enc.encode(text)

    def _decode(self, tokens: list[int]) -> str:
        return self._enc.decode(tokens)

    def fixed(self, text: str) -> list[dict]:
        tokens = self._tokenize(text)
        step = CHUNK_SIZE - CHUNK_OVERLAP
        chunks = []
        i = 0
        idx = 0
        while i < len(tokens):
            chunk_tokens = tokens[i:i + CHUNK_SIZE]
            chunk_text = self._decode(chunk_tokens)
            # snap to nearest space to avoid mid-word cuts
            if i + CHUNK_SIZE < len(tokens):
                last_space = chunk_text.rfind(" ")
                if last_space > 0:
                    chunk_text = chunk_text[:last_space]
            char_start = len(self._decode(tokens[:i]))
            chunks.append({
                "chunk_id": f"fixed_{idx:03d}",
                "text": chunk_text.strip(),
                "token_count": len(self._tokenize(chunk_text.strip())),
                "strategy": "fixed",
                "char_start": char_start,
            })
            idx += 1
            i += step
        return chunks

    def semantic(self, text: str) -> list[dict]:
        # split on sentence boundaries
        sentences = re.split(r'(?<=[.?!])\s+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_sentences = []
        current_tokens = 0
        idx = 0
        char_pos = 0

        for sentence in sentences:
            sent_tokens = len(self._tokenize(sentence))
            if current_tokens + sent_tokens > CHUNK_SIZE and current_sentences:
                chunk_text = " ".join(current_sentences)
                token_count = len(self._tokenize(chunk_text))
                if token_count >= 50:
                    chunks.append({
                        "chunk_id": f"semantic_{idx:03d}",
                        "text": chunk_text,
                        "token_count": token_count,
                        "strategy": "semantic",
                        "char_start": char_pos - len(chunk_text),
                    })
                    idx += 1
                current_sentences = []
                current_tokens = 0
            current_sentences.append(sentence)
            current_tokens += sent_tokens
            char_pos += len(sentence) + 1

        if current_sentences:
            chunk_text = " ".join(current_sentences)
            token_count = len(self._tokenize(chunk_text))
            if token_count >= 50:
                chunks.append({
                    "chunk_id": f"semantic_{idx:03d}",
                    "text": chunk_text,
                    "token_count": token_count,
                    "strategy": "semantic",
                    "char_start": char_pos - len(chunk_text),
                })

        return chunks

    def parent_child(self, text: str) -> tuple[list[dict], dict]:
        parents = self.fixed(text)
        children = []
        parent_map = {}
        child_idx = 0

        for parent in parents:
            parent_text = parent["text"]
            parent_tokens = self._tokenize(parent_text)
            step = CHILD_CHUNK_SIZE
            i = 0
            while i < len(parent_tokens):
                child_tokens = parent_tokens[i:i + CHILD_CHUNK_SIZE]
                child_text = self._decode(child_tokens).strip()
                if not child_text:
                    i += step
                    continue
                # re-tokenize after strip; truncate if stripping caused token count to grow
                child_retokenized = self._tokenize(child_text)
                if len(child_retokenized) > CHILD_CHUNK_SIZE:
                    child_retokenized = child_retokenized[:CHILD_CHUNK_SIZE]
                    child_text = self._decode(child_retokenized)
                child_id = f"pc_child_{child_idx:03d}"
                children.append({
                    "chunk_id": child_id,
                    "text": child_text,
                    "token_count": len(child_retokenized),
                    "strategy": "parent_child",
                    "char_start": parent["char_start"] + len(self._decode(parent_tokens[:i])),
                })
                parent_map[child_id] = parent_text
                child_idx += 1
                i += step

        return children, parent_map


if __name__ == "__main__":
    sample = ("The mitochondria is the powerhouse of the cell and produces ATP. " * 80).strip()
    c = Chunker()

    fixed = c.fixed(sample)
    print(f"Fixed: {len(fixed)} chunks, avg tokens: {sum(x['token_count'] for x in fixed)//len(fixed)}")

    semantic = c.semantic(sample)
    print(f"Semantic: {len(semantic)} chunks, avg tokens: {sum(x['token_count'] for x in semantic)//max(len(semantic),1)}")

    children, parent_map = c.parent_child(sample)
    print(f"Parent-child: {len(children)} child chunks, {len(set(parent_map.values()))} unique parents")

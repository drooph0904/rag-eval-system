CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
CHILD_CHUNK_SIZE = 128
TOP_K_RETRIEVAL = 10
TOP_K_FINAL = 5
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
HYDE_MODEL = "gpt-4o-mini"
ANSWER_MODEL = "gpt-4o-mini"
JUDGE_MODEL = "gpt-4o-mini"
COHERE_RERANK_MODEL = "rerank-english-v3.0"
# Cosine-similarity threshold for marking a retrieved chunk "relevant" in Stage 1.
# Calibrated for all-MiniLM-L6-v2: a clearly-relevant short-answer↔chunk pair scores
# ~0.6-0.7, weak matches ~0.3-0.4, unrelated ~0. 0.5 separates these cleanly.
STAGE1_SIMILARITY_THRESHOLD = 0.5
# A chunk also counts as relevant if it literally contains the answer: this fraction
# of the answer's significant tokens must appear in the chunk. High-precision signal
# for short factual answers (names, numbers, titles) that embeddings under-score.
# Does NOT relax the embedding threshold, so other document types are unaffected.
STAGE1_LEXICAL_OVERLAP = 0.6
INDEX_DIR = "./indexes"
RESULTS_DIR = "./results"
UPLOADS_DIR = "./uploads"
# Stratified sampling for UI-driven runs: evaluate up to this many questions
# per question_type (covers every type while bounding runtime).
SAMPLE_PER_TYPE = 5
CHUNKING_STRATEGIES = ["fixed", "semantic", "parent_child"]
PIPELINE_NAMES = ["pipeline_1", "pipeline_2", "pipeline_3"]

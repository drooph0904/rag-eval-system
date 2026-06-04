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
INDEX_DIR = "./indexes"
RESULTS_DIR = "./results"
CHUNKING_STRATEGIES = ["fixed", "semantic", "parent_child"]
PIPELINE_NAMES = ["pipeline_1", "pipeline_2", "pipeline_3"]

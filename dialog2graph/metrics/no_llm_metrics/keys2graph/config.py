# experiments/exp2025_03_27_create_graphs_by_keys/keys2graph/config.py

"""
Global configuration for our pipeline.
All environment keys are taken from OS ENV if available,
otherwise fallback is used.
"""

import os

# Number of times to attempt JSON fixing
FIX_ATTEMPTS = 3

# Number of times to attempt regeneration if fixing fails
REGENERATION_ATTEMPTS = 1

# Approximate cost table (USD per 1000 tokens)
COST_TABLE = {
    "gpt-3.5-turbo": {"prompt": 0.0015, "completion": 0.0020},
    "gpt-4": {"prompt": 0.03, "completion": 0.06},
}

# Default model for embeddings if none is set in environment
EMBEDDING_MODEL_FALLBACK = "text-embedding-3-small"


# Read from environment or fallback
def get_embedding_model() -> str:
    return os.getenv("EMBEDDING_MODEL", EMBEDDING_MODEL_FALLBACK)


# Semantic similarity threshold for text matching
SEMANTIC_THRESHOLD = 0.75

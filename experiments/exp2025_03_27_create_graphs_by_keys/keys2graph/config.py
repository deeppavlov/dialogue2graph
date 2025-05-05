# experiments/exp2025_03_27_create_graphs_by_keys/keys2graph/config.py

"""
Global configuration for the keys-to-graph pipeline.

All constants can be overridden via OS environment variables. If a value is
*not* found in the environment, the fallback defined below is used instead.
"""

import os

# Maximum number of attempts to repair a malformed JSON string
FIX_ATTEMPTS = 3

# If JSON fixing fails, how many fresh generations to try
REGENERATION_ATTEMPTS = 1

# Default embedding model name (when EMBEDDING_MODEL is not set externally)
EMBEDDING_MODEL_FALLBACK = "text-embedding-3-small"

def get_embedding_model() -> str:
    """
    Return the embedding model name.

    The value is taken from the ``EMBEDDING_MODEL`` environment variable if it
    exists; otherwise :pydata:`EMBEDDING_MODEL_FALLBACK` is returned.

    :returns: model identifier understood by the OpenAI embeddings endpoint.
    """
    return os.getenv("EMBEDDING_MODEL", EMBEDDING_MODEL_FALLBACK)

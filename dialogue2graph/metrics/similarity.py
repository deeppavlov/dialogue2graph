"""
Similarity
-----------

The module contains functions to compare how similar dialog texts are.
"""

from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.evaluation import load_evaluator

preloaded_models = {}


def compare_strings(
    first: str, second: str, embedder: HuggingFaceEmbeddings, embedder_th: float = 0.001
) -> bool:
    """Calculate pairwise_embedding_distance between two strings based on embedder

    Returns:
        True when threshold embedder_th not exceeded, False otherwise"""

    evaluator_2 = load_evaluator("pairwise_embedding_distance", embeddings=embedder)
    score = evaluator_2.evaluate_string_pairs(prediction=first, prediction_b=second)[
        "score"
    ]
    # print("SCORE: ", score)
    return score <= embedder_th


def get_similarity(
    generated: list[str],
    golden: list[str],
    model_name: str = "BAAI/bge-m3",
    device="cuda:0",
):
    """Calculate similarity matrix between generated and golden using model model_name"""

    if model_name not in preloaded_models:
        preloaded_models[model_name] = SentenceTransformer(model_name, device=device)

    golden_vectors = preloaded_models[model_name].encode(
        golden, normalize_embeddings=True
    )
    generated_vectors = preloaded_models[model_name].encode(
        generated, normalize_embeddings=True
    )
    similarities = generated_vectors @ golden_vectors.T
    return similarities

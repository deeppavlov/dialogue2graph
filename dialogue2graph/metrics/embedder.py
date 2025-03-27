import os
from typing import Optional
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.evaluation import load_evaluator
from pydantic_settings import BaseSettings, SettingsConfigDict


class EnvSettings(BaseSettings, case_sensitive=True):
    """Pydantic settings to get env variables"""

    model_config = SettingsConfigDict(
        env_file=os.environ.get("PATH_TO_ENV", ".env"), env_file_encoding="utf-8", env_file_exists_ok=False  # Makes .env file optional
    )
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_BASE_URL: Optional[str] = None
    HUGGINGFACE_TOKEN: Optional[str] = None
    SAMPLING_MAX: Optional[int] = 1000000  # Default value
    DEVICE: Optional[str] = "cpu"  # Default value


# Try to load settings, fall back to defaults if fails
try:
    env_settings = EnvSettings()
except Exception:
    env_settings = EnvSettings(_env_file=None)

preloaded_models = {}


def compare_strings(first: str, second: str, embeddings: HuggingFaceEmbeddings, embedder_th: float = 0.001) -> bool:
    """Calculate pairwise_embedding_distance between two strings based on embeddings
    and return True when threshold embedder_th not exceeded
    Return False othetwise"""

    evaluator_2 = load_evaluator("pairwise_embedding_distance", embeddings=embeddings)
    score = evaluator_2.evaluate_string_pairs(prediction=first, prediction_b=second)["score"]
    # print("SCORE: ", score)
    return score <= embedder_th


def get_similarity(generated: list[str], golden: list[str], model_name: str = "BAAI/bge-m3"):
    """ "Calculate similarity matrix between generated and golden using model model_name"""

    if model_name not in preloaded_models:
        preloaded_models[model_name] = SentenceTransformer(model_name, device=env_settings.DEVICE)

    golden_vectors = preloaded_models[model_name].encode(golden, normalize_embeddings=True)
    generated_vectors = preloaded_models[model_name].encode(generated, normalize_embeddings=True)
    similarities = generated_vectors @ golden_vectors.T
    return similarities

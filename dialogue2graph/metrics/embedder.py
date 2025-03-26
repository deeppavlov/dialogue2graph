import os
from typing import Optional
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.evaluation import load_evaluator
from pydantic_settings import BaseSettings, SettingsConfigDict


class EnvSettings(BaseSettings, case_sensitive=True):

    model_config = SettingsConfigDict(env_file=os.environ["PATH_TO_ENV"], env_file_encoding="utf-8")

    OPENAI_API_KEY: Optional[str]
    OPENAI_BASE_URL: Optional[str]
    HUGGINGFACE_TOKEN: Optional[str]
    SAMPLING_MAX: Optional[int]
    DEVICE: Optional[str]


env_settings = EnvSettings()
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

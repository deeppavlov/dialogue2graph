import numpy as np
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from sentence_transformers import SentenceTransformer
from langchain_community.cross_encoders.huggingface import HuggingFaceCrossEncoder
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.evaluation import load_evaluator


class EnvSettings(BaseSettings, case_sensitive=True):

    model_config = SettingsConfigDict(env_file="./.env", env_file_encoding="utf-8")
    OPENAI_API_KEY: Optional[str]
    OPENAI_BASE_URL: Optional[str]
    HUGGINGFACE_TOKEN: Optional[str]
    EMBEDDER_DEVICE: Optional[str]


embedding = {}
env_settings = EnvSettings()
evaluator = {}


def compare_strings(first: str, second: str, embeddings: HuggingFaceEmbeddings, embedder_th: float = 0.001) -> bool:

    evaluator_2 = load_evaluator("pairwise_embedding_distance", embeddings=embeddings)
    score = evaluator_2.evaluate_string_pairs(prediction=first, prediction_b=second)["score"]
    # print("SCORE: ", score)
    return score <= embedder_th


def get_embedding(generated: list[str], golden: list[str], emb_name: str = "BAAI/bge-m3"):

    if emb_name not in embedding:
        embedding[emb_name] = SentenceTransformer(emb_name, device=env_settings.EMBEDDER_DEVICE)

    golden_vectors = embedding[emb_name].encode(golden, normalize_embeddings=True)
    generated_vectors = embedding[emb_name].encode(generated, normalize_embeddings=True)
    similarities = generated_vectors @ golden_vectors.T
    return similarities


def get_reranking(generated: list[str], golden: list[str], evaluator_name: str = "BAAI/bge-reranker-v2-m3") -> np.ndarray:

    sz = len(generated)
    to_score = []
    for gen in generated:
        for gol in golden:
            to_score.append((gen, gol))

    if evaluator_name not in evaluator:
        evaluator[evaluator_name] = HuggingFaceCrossEncoder(model_name=evaluator_name, model_kwargs={"device": env_settings.EMBEDDER_DEVICE})
    score = np.array(evaluator[evaluator_name].score(to_score))

    return score.reshape(sz, sz)


def get_2_rerankings(
    generated1: list[str], golden1: list[str], generated2: list[str], golden2: list[str], evaluator_name: str = "BAAI/bge-reranker-v2-m3"
) -> tuple[np.ndarray]:

    sz = len(generated1)
    to_score = []
    for gen in generated1:
        for gol in golden1:
            to_score.append((gen, gol))
    for gen in generated2:
        for gol in golden2:
            to_score.append((gen, gol))

    if evaluator_name not in evaluator:
        evaluator[evaluator_name] = HuggingFaceCrossEncoder(model_name=evaluator_name, model_kwargs={"device": env_settings.EMBEDDER_DEVICE})
    scores = np.array(evaluator[evaluator_name].score(to_score))

    return scores[: sz * sz].reshape(sz, sz), scores[sz * sz :].reshape(sz, sz)

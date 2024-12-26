import numpy as np
from sentence_transformers import SentenceTransformer

from langchain_community.cross_encoders.huggingface import HuggingFaceCrossEncoder
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.evaluation import load_evaluator

from chatsky_llm_autoconfig.settings import EnvSettings
env_settings = EnvSettings()
embedding = {}





evaluator = HuggingFaceCrossEncoder(model_name=env_settings.RERANKER_MODEL, model_kwargs={"device": env_settings.EMBEDDER_DEVICE})

# def compare_strings(first: str, second: str):

#     # score = evaluator.evaluate_string_pairs(prediction=first, prediction_b=second)['score']
#     score = evaluator.score([(first, second)])[0]
#     # if 0.99 > score > 0.94:
#     #     print("SCORE: ", score, first, second)
#     # return score <= env_settings.EMBEDDER_THRESHOLD
#     return score >= env_settings.RERANKER_THRESHOLD

def compare_strings(first: str, second: str, embeddings: HuggingFaceEmbeddings):

    evaluator_2 = load_evaluator("pairwise_embedding_distance", embeddings=embeddings)
    score = evaluator_2.evaluate_string_pairs(prediction=first, prediction_b=second)['score']
    return score <= env_settings.EMBEDDER_THRESHOLD


class EmbeddableString:
    def __init__(self, element: str):
        self.element = element
    def __eq__(self, other):
        return compare_strings(self.element,other.element)
    def __hash__(self):
        return hash("")
    def __str__(self):
        return self.element

def emb_list(x):
    #print("EMB_LIST: ", x)
    return [EmbeddableString(el) for el in x]

def get_embedding(generated: list[str], golden: list[str], emb_name: str, device: str):

    if emb_name not in embedding:
        embedding[emb_name] = SentenceTransformer(emb_name,device=device)
 
    golden_vectors = embedding[emb_name].encode(golden, normalize_embeddings=True)
    generated_vectors = embedding[emb_name].encode(generated, normalize_embeddings=True)
    similarities = generated_vectors @ golden_vectors.T
    return similarities

def get_reranking(generated: list[str], golden: list[str]):
    
    sz = len(generated)
    to_score = []
    for gen in generated:
        for gol in golden:
            to_score.append((gen,gol))
    print("SCORING...")
    # print(to_score)
    score = np.array(evaluator.score(to_score))
    print("finished")

    return score.reshape(sz,sz)

def get_2_rerankings(generated1: list[str], golden1: list[str], generated2: list[str], golden2: list[str]):
    
    sz = len(generated1)
    to_score = []
    for gen in generated1:
        for gol in golden1:
            to_score.append((gen,gol))
    for gen in generated2:
        for gol in golden2:
            to_score.append((gen,gol))
    print("SCORING...")
    # print(to_score)
    scores = np.array(evaluator.score(to_score))
    print("finished")

    return scores[:sz*sz].reshape(sz,sz), scores[sz*sz:].reshape(sz,sz)

# experiments/exp2025_03_27_create_graphs_by_keys/keys2graph/graph_triplet_comparison.py
# ------------------------------------------------------------------------------------
#         ── версия с поддержкой нового client.embeddings.create(..) синтаксиса ──
# ------------------------------------------------------------------------------------
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment

# from openai import OpenAI                              # NEW
import dotenv
from langchain_huggingface import HuggingFaceEmbeddings

# from .config import get_embedding_model
# для имени модели
dotenv.load_dotenv()
# ---------- 1. инициализируем OpenAI‑клиент один раз ----------
# api_key=os.environ["OPENAI_API_KEY"]
# base_url=os.environ["OPENAI_API_BASE"]


def _embed(model: HuggingFaceEmbeddings, text: str) -> np.ndarray:
    document_embeddings = model.embed_documents([text])[0]
    return np.array(document_embeddings, dtype=np.float32)


# _client = OpenAI(api_key=api_key, base_url=base_url)
# _client = OpenAI()

# def _embed(text: str, model: str) -> np.ndarray:       # NEW
#     """
#     Нормализует перевод строки → пробел и возвращает вектор‑эмбеддинг
#     по новому синтаксису client.embeddings.create(...)
#     """
#     text = text.replace("\n", " ")
#     vec = _client.embeddings.create(input=[text], model=model).data[0].embedding
#     return np.array(vec, dtype=np.float32)


# ---------- 2. основная публичная функция ----------
def compare_two_graphs(model: HuggingFaceEmbeddings, original_graph, generated_graph):
    """
    Возвращает:
      {
        "similarity_avg": float,
        "matched_triplets": [ { original_triplet, generated_triplet, similarity_score } ]
      }
    """
    original_graph = original_graph.graph_dict
    generated_graph = generated_graph.graph_dict

    orig_triplets = _get_triplets_from_graph(original_graph)
    gen_triplets = _get_triplets_from_graph(generated_graph)

    emb_o, utt_o = _build_maps(model, original_graph)  # CHANGED (см. ниже)
    emb_g, utt_g = _build_maps(model, generated_graph)  # CHANGED

    N, M = len(orig_triplets), len(gen_triplets)
    sim_matrix = np.zeros((N, M), dtype=np.float32)
    for i, o_tr in enumerate(orig_triplets):
        for j, g_tr in enumerate(gen_triplets):
            sim_matrix[i, j] = _triplet_similarity(o_tr, g_tr, emb_o, emb_g)

    size = max(N, M)
    padded = np.zeros((size, size), dtype=np.float32)
    padded[:N, :M] = sim_matrix
    rows, cols = linear_sum_assignment(padded, maximize=True)  # maximise=True

    pairs = [(r, c) for r, c in zip(rows, cols) if r < N and c < M]
    scores = [sim_matrix[r, c] for r, c in pairs]
    similarity_avg = float(np.mean(scores)) if scores else 0.0

    matched = []
    for r, c in pairs:
        matched.append(
            {
                "original_triplet": _pretty_triplet(orig_triplets[r], utt_o),
                "generated_triplet": _pretty_triplet(gen_triplets[c], utt_g),
                "similarity_score": float(sim_matrix[r, c]),
            }
        )

    return {"similarity_avg": similarity_avg, "matched_triplets": matched}


# ---------- 3. утилиты для триплетов ----------
def _get_triplets_from_graph(g):
    """
    Формирует список триплетов node→edge→node и edge→node→edge.
    Работает корректно даже если в edges встречаются node‑id, отсутствующие
    в разделе "nodes" (создаёт для них пустые списки).
    """
    nodes, edges = g.get("nodes", []), g.get("edges", [])

    # карты «входящие» и «исходящие» рёбра
    inc, out = {}, {}
    for n in nodes:
        inc[n["id"]] = []
        out[n["id"]] = []

    # дополняем словари при встрече «незнакомых» id
    def _ensure(node_id):
        if node_id not in inc:  # значит узла нет в nodes[]
            print("Ошибка - нет такой ноды")
            inc[node_id] = []
            out[node_id] = []

    for e in edges:
        src, tgt = e["source"], e["target"]
        _ensure(src)
        _ensure(tgt)  # ← NEW
        out[src].append(tgt)
        inc[tgt].append(src)

    triplets = []
    # node → edge → node
    for e in edges:
        triplets.append(
            {
                "start_type": "node",
                "start_id": e["source"],
                "middle_type": "edge",
                "middle_id": (e["source"], e["target"]),
                "end_type": "node",
                "end_id": e["target"],
            }
        )
    # edge → node → edge
    for nid in inc.keys():  # все известные id (вкл. «плавающие»)
        for s in inc[nid]:
            for t in out[nid]:
                triplets.append(
                    {
                        "start_type": "edge",
                        "start_id": (s, nid),
                        "middle_type": "node",
                        "middle_id": nid,
                        "end_type": "edge",
                        "end_id": (nid, t),
                    }
                )
    return triplets


def _build_maps(model: HuggingFaceEmbeddings, g):
    """
    Возвращает:
      embeddings_map : ключ -> [векторы]
      utterances_map : ключ -> [строки]
    Использует _embed(...) (новый синтаксис OpenAI).
    """
    # model = get_embedding_model()
    e_map, u_map = {}, {}

    # nodes
    for n in g.get("nodes", []):
        key = ("node", n["id"])
        u_map[key] = n.get("utterances", [])
        e_map[key] = [_embed(model, u) for u in u_map[key]]
    # edges
    for e in g.get("edges", []):
        key = ("edge", (e["source"], e["target"]))
        u_map[key] = e.get("utterances", [])
        e_map[key] = [_embed(model, u) for u in u_map[key]]

    return e_map, u_map


def _max_sim(A, B):
    if not A or not B:
        return 0.0
    return float(cosine_similarity(np.vstack(A), np.vstack(B)).max())


def _triplet_similarity(o, g, emb_o, emb_g):
    key_o_start = (o["start_type"], o["start_id"])
    key_g_start = (g["start_type"], g["start_id"])
    key_o_mid = (o["middle_type"], o["middle_id"])
    key_g_mid = (g["middle_type"], g["middle_id"])
    key_o_end = (o["end_type"], o["end_id"])
    key_g_end = (g["end_type"], g["end_id"])

    start = (
        _max_sim(emb_o.get(key_o_start, []), emb_g.get(key_g_start, []))
        if o["start_type"] == g["start_type"]
        else 0.0
    )

    middle = (
        _max_sim(emb_o.get(key_o_mid, []), emb_g.get(key_g_mid, []))
        if o["middle_type"] == g["middle_type"]
        else 0.0
    )

    end = (
        _max_sim(emb_o.get(key_o_end, []), emb_g.get(key_g_end, []))
        if o["end_type"] == g["end_type"]
        else 0.0
    )

    return (start + middle + end) / 3.0


def _pretty_triplet(tr, utt_map):
    def pack(tp, idx):
        return {"type": tp, "id": idx, "utterances": utt_map.get((tp, idx), [])}

    return {
        "start": pack(tr["start_type"], tr["start_id"]),
        "middle": pack(tr["middle_type"], tr["middle_id"]),
        "end": pack(tr["end_type"], tr["end_id"]),
    }

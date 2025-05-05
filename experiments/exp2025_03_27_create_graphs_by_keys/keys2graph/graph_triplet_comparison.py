# experiments/exp2025_03_27_create_graphs_by_keys/keys2graph/graph_triplet_comparison.py

"""
Compare an *original* dialogue graph to a *generated* one by aligning their
node/edge triplets and computing an average cosine similarity of embeddings.

A *triplet* is either::

    node → edge → node        or        edge → node → edge

The similarity metric embeds every utterance with the selected embedding model,
computes cosine similarity between all utterance pairs belonging to
corresponding elements of two triplets, and keeps the maximum score.
The Hungarian algorithm (``linear_sum_assignment``) is then used to find an
optimal 1-to-1 matching between triplets.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
from openai import OpenAI 
import os

from .config import get_embedding_model

api_key=os.environ["OPENAI_API_KEY"]
base_url=os.environ["OPENAI_API_BASE"]

_client = OpenAI(api_key=api_key, base_url=base_url)

def _embed(text: str, model: str) -> np.ndarray:
    """
    Embed *text* with the specified *model* and return a NumPy vector.

    All newlines are replaced with spaces to avoid misleading the embedding
    model with artificial line breaks.
    """
    text = text.replace("\n", " ")
    vec = _client.embeddings.create(input=[text], model=model).data[0].embedding
    return np.array(vec, dtype=np.float32)

def compare_two_graphs(original_graph, generated_graph):
    """
    Compare two graphs and return average similarity plus matched triplets.

    :returns: ``{"similarity_avg": float,
                 "matched_triplets": [ {...}, ... ]}``
    """
    orig_triplets = _get_triplets_from_graph(original_graph)
    gen_triplets  = _get_triplets_from_graph(generated_graph)

    emb_o, utt_o = _build_maps(original_graph)         # CHANGED (см. ниже)
    emb_g, utt_g = _build_maps(generated_graph)        # CHANGED

    N, M = len(orig_triplets), len(gen_triplets)
    sim_matrix = np.zeros((N, M), dtype=np.float32)

    # pair-wise triplet similarity matrix
    for i, o_tr in enumerate(orig_triplets):
        for j, g_tr in enumerate(gen_triplets):
            sim_matrix[i, j] = _triplet_similarity(o_tr, g_tr, emb_o, emb_g)

    # pad to square matrix for Hungarian algorithm
    size = max(N, M)
    padded = np.zeros((size, size), dtype=np.float32)
    padded[:N, :M] = sim_matrix
    rows, cols = linear_sum_assignment(padded, maximize=True)   # maximise=True

    pairs = [(r, c) for r, c in zip(rows, cols) if r < N and c < M]
    scores = [sim_matrix[r, c] for r, c in pairs]
    similarity_avg = float(np.mean(scores)) if scores else 0.0

    matched = []
    for r, c in pairs:
        matched.append({
            "original_triplet": _pretty_triplet(orig_triplets[r], utt_o),
            "generated_triplet": _pretty_triplet(gen_triplets[c], utt_g),
            "similarity_score": float(sim_matrix[r, c])
        })

    return {"similarity_avg": similarity_avg, "matched_triplets": matched}

def _get_triplets_from_graph(g):
    """
    Extract **all** triplets from graph *g*.

    The function is robust to "dangling" node IDs that are referenced by edges
    but absent from ``g["nodes"]`` - it creates empty placeholders for them.
    """
    nodes, edges = g.get("nodes", []), g.get("edges", [])

    # maps of incoming / outgoing edges
    inc, out = {}, {}
    for n in nodes:
        inc[n["id"]] = []
        out[n["id"]] = []

    def _ensure(node_id):
        """Create empty slots for nodes missing in the *nodes* list."""
        if node_id not in inc:   # значит узла нет в nodes[]
            print("Warning: node id referenced in edges but absent in nodes.")
            inc[node_id] = []
            out[node_id] = []

    for e in edges:
        src, tgt = e["source"], e["target"]
        _ensure(src); _ensure(tgt)          # ← NEW
        out[src].append(tgt)
        inc[tgt].append(src)

    triplets = []
    # node → edge → node
    for e in edges:
        triplets.append({
            "start_type": "node",  "start_id": e["source"],
            "middle_type": "edge", "middle_id": (e["source"], e["target"]),
            "end_type": "node",    "end_id": e["target"]
        })
    # edge → node → edge
    for nid in inc.keys():
        for s in inc[nid]:
            for t in out[nid]:
                triplets.append({
                    "start_type": "edge",  "start_id": (s, nid),
                    "middle_type": "node", "middle_id": nid,
                    "end_type": "edge",    "end_id": (nid, t)
                })
    return triplets


def _build_maps(g):
    """
    Build two dictionaries for quick lookup:

    * ``embeddings_map`` – key → list of embedding vectors
    * ``utterances_map`` – key → list of raw utterance strings

    A *key* is the tuple ``("node" | "edge", identifier)`` where *identifier*
    is either the numeric node id or ``(source, target)`` for edges.
    """
    model = get_embedding_model()
    e_map, u_map = {}, {}

    # nodes
    for n in g.get("nodes", []):
        key = ("node", n["id"])
        u_map[key] = n.get("utterances", [])
        e_map[key] = [_embed(u, model) for u in u_map[key]]
    # edges
    for e in g.get("edges", []):
        key = ("edge", (e["source"], e["target"]))
        u_map[key] = e.get("utterances", [])
        e_map[key] = [_embed(u, model) for u in u_map[key]]

    return e_map, u_map


def _max_sim(A, B):
    """Return maximum cosine similarity between any vector in *A* and *B*."""
    if not A or not B:
        return 0.0
    return float(cosine_similarity(np.vstack(A), np.vstack(B)).max())


def _triplet_similarity(o, g, emb_o, emb_g):
    """
    Compute mean similarity of corresponding *start*, *middle*, and *end*
    elements of two triplets *o* and *g*.
    """
    key_o_start = (o["start_type"],  o["start_id"])
    key_g_start = (g["start_type"],  g["start_id"])
    key_o_mid   = (o["middle_type"], o["middle_id"])
    key_g_mid   = (g["middle_type"], g["middle_id"])
    key_o_end   = (o["end_type"],    o["end_id"])
    key_g_end   = (g["end_type"],    g["end_id"])

    start = _max_sim(
        emb_o.get(key_o_start, []),
        emb_g.get(key_g_start, [])
    ) if o["start_type"] == g["start_type"] else 0.0

    middle = _max_sim(
        emb_o.get(key_o_mid, []),
        emb_g.get(key_g_mid, [])
    ) if o["middle_type"] == g["middle_type"] else 0.0

    end = _max_sim(
        emb_o.get(key_o_end, []),
        emb_g.get(key_g_end, [])
    ) if o["end_type"] == g["end_type"] else 0.0

    return (start + middle + end) / 3.0


def _pretty_triplet(tr, utt_map):
    """Return a human-readable representation of *tr* with utterances."""
    def pack(tp, idx):
        return {"type": tp, "id": idx, "utterances": utt_map.get((tp, idx), [])}
    return {"start": pack(tr["start_type"], tr["start_id"]),
            "middle": pack(tr["middle_type"], tr["middle_id"]),
            "end": pack(tr["end_type"], tr["end_id"])}

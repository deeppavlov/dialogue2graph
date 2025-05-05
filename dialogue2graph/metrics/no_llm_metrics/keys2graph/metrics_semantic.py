# experiments/exp2025_03_27_create_graphs_by_keys/keys2graph/metrics_semantic.py

"""
Advanced semantic-based metrics for comparing graphs
by embedding their utterances and matching them if similarity >= threshold.
"""

import os
import math
import openai
from typing import List, Dict, Any

from . import config


def _init_openai():
    """
    Init openai keys from environment or fallback.
    """
    openai.api_key = os.getenv("OPENAI_API_KEY", "")
    openai.api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")


def get_text_embedding_openai(texts: List[str], model: str) -> List[List[float]]:
    """
    Get embeddings from OpenAI for a list of text strings, using a single request if possible.
    """
    _init_openai()
    response = openai.Embedding.create(input=texts, model=model)
    embeddings = [r["embedding"] for r in response["data"]]
    return embeddings


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(a * a for a in vec2))
    if norm1 < 1e-9 or norm2 < 1e-9:
        return 0.0
    return dot / (norm1 * norm2)


def _semantic_match_utterances(
    utter1: List[str], utter2: List[str], threshold: float, embed_model: str
) -> bool:
    """
    Returns True if there's at least one pair (u1 in utter1, u2 in utter2)
    with cos sim >= threshold.
    """
    if not utter1 or not utter2:
        return False

    emb1_list = get_text_embedding_openai(utter1, model=embed_model)
    emb2_list = get_text_embedding_openai(utter2, model=embed_model)

    for e1 in emb1_list:
        for e2 in emb2_list:
            sim = cosine_similarity(e1, e2)
            if sim >= threshold:
                return True
    return False


def match_graph_triplets_semantic(G1: Any, G2: Any, threshold: float, embed_model: str):
    """
    Similar to match_graph_triplets, but uses semantic matching
    for edges and nodes. Return node_mapping, edge_mapping.
    """
    g1 = G1.graph
    g2 = G2.graph
    node_mapping = {node: None for node in g1.nodes}
    node_mapping.update({node: None for node in g2.nodes})

    edges1 = list(g1.edges(data=True))
    edges2 = list(g2.edges(data=True))
    edge_mapping = {}

    # Match edges
    for s1, t1, data1 in edges1:
        best_match = None
        utt1 = (
            data1["utterances"]
            if isinstance(data1["utterances"], list)
            else [data1["utterances"]]
        )
        for s2, t2, data2 in edges2:
            utt2 = (
                data2["utterances"]
                if isinstance(data2["utterances"], list)
                else [data2["utterances"]]
            )
            if _semantic_match_utterances(utt1, utt2, threshold, embed_model):
                best_match = (s2, t2)
                break
        edge_mapping[f"{s1}->{t1}"] = (
            f"{best_match[0]}->{best_match[1]}" if best_match else None
        )

    # Match nodes
    for node in g1.nodes(data=True):
        node_id1 = node[0]
        utt1 = (
            node[1]["utterances"]
            if isinstance(node[1]["utterances"], list)
            else [node[1]["utterances"]]
        )
        matched_node = None
        for node2 in g2.nodes(data=True):
            node_id2 = node2[0]
            utt2 = (
                node2[1]["utterances"]
                if isinstance(node2[1]["utterances"], list)
                else [node2[1]["utterances"]]
            )
            if _semantic_match_utterances(utt1, utt2, threshold, embed_model):
                matched_node = node_id2
                break
        node_mapping[node_id1] = matched_node

    return node_mapping, edge_mapping


def is_same_structure_semantic(
    G1: Any, G2: Any, threshold: float, embed_model: str
) -> bool:
    node_map, edge_map = match_graph_triplets_semantic(G1, G2, threshold, embed_model)
    for n, mapped in node_map.items():
        if mapped is None:
            return False
    for e, mapped_e in edge_map.items():
        if mapped_e is None:
            return False
    return True


def match_triplets_dg_semantic(
    G1: Any, dialogues: List[Any], threshold: float, embed_model: str
) -> Dict[str, Any]:
    """
    Stub function to do semantic check of G1 vs dialogues.
    """
    return {"value": True, "semantic_check": True}


def triplet_match_accuracy_semantic(
    G1: Any, G2: Any, threshold: float, embed_model: str
) -> Dict[str, float]:
    node_map, edge_map = match_graph_triplets_semantic(G1, G2, threshold, embed_model)
    g1_nodes = list(G1.graph.nodes())
    matched_nodes = sum(1 for n in g1_nodes if node_map.get(n) is not None)
    total_nodes = len(g1_nodes)
    node_acc = matched_nodes / total_nodes if total_nodes else 0.0

    g1_edges = list(G1.graph.edges())
    matched_edges = sum(1 for _, v in edge_map.items() if v is not None)
    total_edges = len(g1_edges)
    edge_acc = matched_edges / total_edges if total_edges else 0.0
    return {"node_accuracy": node_acc, "edge_accuracy": edge_acc}


def _semantic_jaccard_for_nodes(
    nodes1: List[Dict[str, Any]],
    nodes2: List[Dict[str, Any]],
    threshold: float,
    embed_model: str,
) -> float:
    matched1 = set()
    matched2 = set()

    for i, n1 in enumerate(nodes1):
        utt1 = (
            n1["utterances"]
            if isinstance(n1["utterances"], list)
            else [n1["utterances"]]
        )
        found_j = None
        for j, n2 in enumerate(nodes2):
            utt2 = (
                n2["utterances"]
                if isinstance(n2["utterances"], list)
                else [n2["utterances"]]
            )
            if _semantic_match_utterances(utt1, utt2, threshold, embed_model):
                found_j = j
                break
        if found_j is not None:
            matched1.add(i)
            matched2.add(found_j)

    intersection = len(matched1)
    union = len(nodes1) + len(nodes2) - intersection
    if union == 0:
        return 0.0
    return intersection / union


def _semantic_jaccard_for_edges(
    edges1: List[Dict[str, Any]],
    edges2: List[Dict[str, Any]],
    threshold: float,
    embed_model: str,
) -> float:
    matched1 = set()
    matched2 = set()

    for i, e1 in enumerate(edges1):
        utt1 = (
            e1["utterances"]
            if isinstance(e1["utterances"], list)
            else [e1["utterances"]]
        )
        found_j = None
        for j, e2 in enumerate(edges2):
            utt2 = (
                e2["utterances"]
                if isinstance(e2["utterances"], list)
                else [e2["utterances"]]
            )
            if _semantic_match_utterances(utt1, utt2, threshold, embed_model):
                found_j = j
                break
        if found_j is not None:
            matched1.add(i)
            matched2.add(found_j)

    intersection = len(matched1)
    union = len(edges1) + len(edges2) - intersection
    if union == 0:
        return 0.0
    return intersection / union


def compare_two_graphs_semantically(
    graph1: Dict[str, Any], graph2: Dict[str, Any], threshold: float
) -> Dict[str, float]:
    """
    Compare two dialogue graphs by semantic jaccard for nodes & edges.
    Embedding model is taken from config.get_embedding_model().
    """
    embed_model = config.get_embedding_model()

    nodes1 = graph1.get("nodes", [])
    nodes2 = graph2.get("nodes", [])
    edges1 = graph1.get("edges", [])
    edges2 = graph2.get("edges", [])

    sem_j_nodes = _semantic_jaccard_for_nodes(nodes1, nodes2, threshold, embed_model)
    sem_j_edges = _semantic_jaccard_for_edges(edges1, edges2, threshold, embed_model)

    return {
        "semantic_jaccard_nodes": sem_j_nodes,
        "semantic_jaccard_edges": sem_j_edges,
    }

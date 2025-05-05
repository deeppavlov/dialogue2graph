"""
metrics_semantic_jaccard.py
---------------------------
Semantic‑Jaccard‑метрики для пар диалоговых графов.

Главная точка входа:  evaluate_graph_lists(...)
или класс‑обёртка GraphSimilarityEvaluator.

• Для каждой пары графов вычисляется:
    - semantic_jaccard_nodes
    - semantic_jaccard_edges
• Узлы и рёбра считаются «совпавшими», если
  косинусная близость их эмбеддингов ≥ threshold.
"""

from __future__ import annotations
import json
import os
from typing import List, Dict, Tuple, Any

import numpy as np
from openai import OpenAI


# -------------------- НАСТРОЙКИ ПО УМОЛЧАНИЮ -------------------- #
DEFAULT_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_THRESHOLD = 0.83
BATCH = 96
# ---------------------------------------------------------------- #


# ===============  ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ  =============== #
def _chunk(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


_client: OpenAI | None = None  # единый клиент на всё время работы


def _get_client() -> OpenAI:
    """Создаём и кэшируем экземпляр клиента OpenAI."""
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
        )
    return _client


def embed_texts(texts: List[str], model: str) -> List[np.ndarray]:
    """
    Запрашиваем OpenAI embeddings батчами и возвращаем list[np.ndarray].
    Совместимо с новой версией SDK: response.data -> List[Embedding]
    """
    vectors: List[np.ndarray] = []
    client = _get_client()

    for batch in _chunk(texts, BATCH):
        response = client.embeddings.create(model=model, input=batch)
        # response.data — список Embedding, у каждого есть .embedding
        vectors.extend(np.asarray(e.embedding, dtype=np.float32) for e in response.data)

    return vectors


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class GraphSimilarityEvaluator:
    def __init__(
        self,
        embed_model: str = DEFAULT_EMBED_MODEL,
        threshold: float = DEFAULT_THRESHOLD,
    ):
        self.embed_model = embed_model
        self.threshold = threshold

    def evaluate_pairs(
        self,
        original_graphs: List[Dict[str, Any]],
        generated_graphs: List[Dict[str, Any]],
        json_out_path: str,
    ) -> Tuple[List[Dict[str, Any]], float, float]:
        if len(original_graphs) != len(generated_graphs):
            raise ValueError("Списки графов должны быть одинаковой длины")

        results = []
        j_nodes, j_edges = [], []

        for idx, (g1_obj, g2_obj) in enumerate(zip(original_graphs, generated_graphs)):
            g1, g2 = g1_obj["graph"], g2_obj["graph"]
            res = self._compare_two_graphs(g1, g2, idx)  # ← idx = индекс пары
            results.append(res)
            j_nodes.append(res["semantic_jaccard_nodes"])
            j_edges.append(res["semantic_jaccard_edges"])

            # Вывод итоговой сводки по паре
            print(f"\n=== Pair {idx}: final Jaccard ===")
            print(f"Nodes jaccard = {res['semantic_jaccard_nodes']:.3f}")
            print(f"Edges jaccard = {res['semantic_jaccard_edges']:.3f}")

        # Считаем среднее
        summary = {
            "avg_semantic_jaccard_nodes": float(np.mean(j_nodes)) if j_nodes else 0.0,
            "avg_semantic_jaccard_edges": float(np.mean(j_edges)) if j_edges else 0.0,
            "pairs_count": len(results),
        }

        # Пишем в JSON
        with open(json_out_path, "w", encoding="utf-8") as f:
            json.dump(
                {"per_pair": results, "summary": summary},
                f,
                ensure_ascii=False,
                indent=2,
            )

        return (
            results,
            summary["avg_semantic_jaccard_nodes"],
            summary["avg_semantic_jaccard_edges"],
        )

    def _compare_two_graphs(
        self, g1: Dict[str, Any], g2: Dict[str, Any], pair_index: int
    ):
        # Узлы
        node_vecs1, node_vecs2 = self._embed_nodes(g1), self._embed_nodes(g2)
        matched_nodes = self._greedy_match(
            vecs1=node_vecs1,
            vecs2=node_vecs2,
            objs1=g1["nodes"],
            objs2=g2["nodes"],
            pair_index=pair_index,
            obj_type="node",
        )
        j_nodes = self._jaccard(len(matched_nodes), len(g1["nodes"]), len(g2["nodes"]))

        # Рёбра
        edge_vecs1, edge_vecs2 = self._embed_edges(g1), self._embed_edges(g2)
        matched_edges = self._greedy_match(
            vecs1=edge_vecs1,
            vecs2=edge_vecs2,
            objs1=g1["edges"],
            objs2=g2["edges"],
            pair_index=pair_index,
            obj_type="edge",
        )
        j_edges = self._jaccard(len(matched_edges), len(g1["edges"]), len(g2["edges"]))

        return {
            "pair_index": pair_index,
            "semantic_jaccard_nodes": j_nodes,
            "semantic_jaccard_edges": j_edges,
            "matched_nodes": matched_nodes,  # [(node1, node2, cos)]
            "matched_edges": matched_edges,  # [(edge1, edge2, cos)]
        }

    def _embed_nodes(self, g):
        texts = [" ".join(n.get("utterances", [])) for n in g["nodes"]]
        return embed_texts(texts, self.embed_model)

    def _edge_text(self, edge, id2node):
        src_lbl = id2node[edge["source"]]["label"]
        tgt_lbl = id2node[edge["target"]]["label"]
        utt = " ".join(edge.get("utterances", []))
        return f"{src_lbl} -> {tgt_lbl} {utt}"

    def _embed_edges(self, g):
        id2node = {n["id"]: n for n in g["nodes"]}
        texts = []
        for e in g["edges"]:
            txt = self._edge_text(e, id2node)
            texts.append(txt)
        return embed_texts(texts, self.embed_model)

    def _greedy_match(
        self,
        vecs1: List[np.ndarray],
        vecs2: List[np.ndarray],
        objs1: List[Dict[str, Any]],
        objs2: List[Dict[str, Any]],
        pair_index: int,
        obj_type: str = "node",
    ) -> List[Tuple[Any, Any, float]]:
        """
        Для всех пар (i,j) печатаем диагностику:
         - pair_index, id original, id generated,
         - тексты (либо node.utterances, либо edge.utterances),
         - cos similarity,
         - 'слеиваются' или 'не сливаются' (по threshold).
        Затем оставляем только пары, у которых cos >= threshold.
        Делаем жадное сопоставление 1‑к‑1 (remove already used).
        """

        # соберём все пары (i, j) с их косинусом (без фильтра)
        all_pairs = []
        for i, v1 in enumerate(vecs1):
            for j, v2 in enumerate(vecs2):
                c = cosine(v1, v2)
                all_pairs.append((i, j, c))

        # Печатаем инфу о каждой паре
        for i, j, c in all_pairs:
            # original / generated
            id_str_1 = (
                str(objs1[i].get("id"))
                if obj_type == "node"
                else f"{objs1[i]['source']}→{objs1[i]['target']}"
            )
            id_str_2 = (
                str(objs2[j].get("id"))
                if obj_type == "node"
                else f"{objs2[j]['source']}→{objs2[j]['target']}"
            )

            # тексты
            if obj_type == "node":
                text1 = " ".join(objs1[i].get("utterances", []))
                text2 = " ".join(objs2[j].get("utterances", []))
            else:
                text1 = " ".join(objs1[i].get("utterances", []))
                text2 = " ".join(objs2[j].get("utterances", []))

            # решение
            is_merge = c >= self.threshold
            result_str = "слеиваются" if is_merge else "не сливаются"

            print("\n--- COMPARISON ---")
            print(f"Pair index (graph) : {pair_index}")
            print(f"Object type        : {obj_type}")
            print(f"Original {obj_type} id: {id_str_1}")
            print(f"Generated {obj_type} id: {id_str_2}")
            print(f"Original text      : {text1}")
            print(f"Generated text     : {text2}")
            print(f"Cosine similarity  : {c:.3f}")
            print(f"Result (threshold={self.threshold}): {result_str}")

        # Теперь отфильтруем по threshold + сортировка убывания
        pairs_above_thr = [(i, j, c) for (i, j, c) in all_pairs if c >= self.threshold]
        pairs_above_thr.sort(key=lambda x: x[2], reverse=True)

        # Greedy match
        used1, used2 = set(), set()
        matched = []
        for i, j, c in pairs_above_thr:
            if i not in used1 and j not in used2:
                matched.append((objs1[i], objs2[j], c))
                used1.add(i)
                used2.add(j)

        return matched

    @staticmethod
    def _jaccard(intersection: int, len1: int, len2: int) -> float:
        union = len1 + len2 - intersection
        return intersection / union if union else 0.0


def evaluate_graph_lists(
    original_graphs: List[Dict[str, Any]],
    generated_graphs: List[Dict[str, Any]],
    output_path: str,
    model_name: str = DEFAULT_EMBED_MODEL,
    threshold: float = DEFAULT_THRESHOLD,
):
    evaluator = GraphSimilarityEvaluator(embed_model=model_name, threshold=threshold)
    return evaluator.evaluate_pairs(original_graphs, generated_graphs, output_path)

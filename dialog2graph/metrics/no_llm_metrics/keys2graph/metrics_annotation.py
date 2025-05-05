# experiments/exp2025_03_27_create_graphs_by_keys/keys2graph/metrics_annotation.py

"""
Module to compare annotation-level similarity between original and generated graphs.

For each key in the annotation:
- If both are numeric, compute absolute difference.
- If both are text (not unknown), compute semantic similarity (cosine).
- If either is "unknown", skip from accuracy. We'll track them separately.

We return per-key stats and then aggregated stats.
"""

from typing import Dict, Any

from . import config
from .metrics_semantic import get_text_embedding_openai, cosine_similarity


def is_numeric(val) -> bool:
    if isinstance(val, (int, float)):
        return True
    if isinstance(val, str) and val.isdigit():
        return True
    return False


def compare_annotation_differences(
    ann_original: Dict[str, Any], ann_generated: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compare two annotation dictionaries (like ann_original, ann_generated).
    Return detailed and summary stats:

    {
      "per_key": {
         "topic": {"orig": "...", "gen": "...", "score": 1.0, "difference": 0.0, "is_unknown": False, ...},
         ...
      },
      "summary": {
         "avg_score": 0.XX,
         "avg_difference": 0.XX,
         "count_keys_compared": X,
         "count_unknowns": Y,
         "unknown_keys": [...]
      }
    }
    """

    all_keys = set(ann_original.keys()).union(set(ann_generated.keys()))
    per_key = {}
    embedding_model = config.get_embedding_model()  # Модель эмбеддингов

    # collect stats
    scores = []  # косинусная похожесть (для текстовых)
    differences = []  # для числовых
    unknown_count = 0
    unknown_keys = []

    for k in all_keys:
        orig_val = ann_original.get(k, "unknown")
        gen_val = ann_generated.get(k, "unknown")

        # Если хоть одно == "unknown", пропускаем из accuracy, но учитываем в статистике unknown
        if orig_val == "unknown" or gen_val == "unknown":
            per_key[k] = {
                "orig": orig_val,
                "gen": gen_val,
                "score": None,
                "difference": None,
                "is_unknown": True,
            }
            unknown_count += 1
            unknown_keys.append(k)
            continue

        # Оба не unknown => сравниваем
        # Если оба числовые -> разница
        if is_numeric(orig_val) and is_numeric(gen_val):
            # float cast
            f1 = float(orig_val)
            f2 = float(gen_val)
            diff = abs(f1 - f2)
            differences.append(diff)
            per_key[k] = {
                "orig": orig_val,
                "gen": gen_val,
                "score": None,  # не применяем cos similarity
                "difference": diff,
                "is_unknown": False,
            }
        else:
            # считаем это текст
            str1 = str(orig_val)
            str2 = str(gen_val)
            emb1 = get_text_embedding_openai([str1], model=embedding_model)[0]
            emb2 = get_text_embedding_openai([str2], model=embedding_model)[0]
            sim = cosine_similarity(emb1, emb2)
            scores.append(sim)
            per_key[k] = {
                "orig": orig_val,
                "gen": gen_val,
                "score": sim,
                "difference": None,
                "is_unknown": False,
            }

    # Подсчитываем summary
    avg_score = sum(scores) / len(scores) if scores else 0.0
    avg_diff = sum(differences) / len(differences) if differences else 0.0
    total_compared = (
        len(all_keys) - unknown_count
    )  # кол-во ключей, которые реально сравнили

    summary = {
        "avg_score": avg_score,
        "avg_difference": avg_diff,
        "count_keys_compared": total_compared,
        "count_unknowns": unknown_count,
        "unknown_keys": unknown_keys,
    }

    return {"per_key": per_key, "summary": summary}

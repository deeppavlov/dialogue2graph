# experiments/exp2025_03_27_create_graphs_by_keys/keys2graph/annotation_pipeline.py

from typing import List, Dict, Any, Optional
from .config import FIX_ATTEMPTS, REGENERATION_ATTEMPTS
from .model_interactions import (
    generate_annotation,
    fix_json_with_gpt35,
    regenerate_annotation,
)
from .json_validator import (
    is_json_string_valid,
    parse_json_string,
    check_annotation_structure,
)


def annotate_single_graph(
    dialog_graph: Dict[str, Any],
    prompt_instructions: str,
    model_name: str,
    temperature: float,
    api_key: str,
    base_url: str,
    supports_user_role: bool = True,
) -> Optional[Dict[str, Any]]:
    annotation_str = generate_annotation(
        dialog_graph=dialog_graph,
        prompt_instructions=prompt_instructions,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
        supports_user_role=supports_user_role,
    )

    # Validate
    if is_json_string_valid(annotation_str):
        annotation_obj = parse_json_string(annotation_str)
        if check_annotation_structure(annotation_obj):
            return annotation_obj

    # Try fixes
    for _ in range(FIX_ATTEMPTS):
        annotation_str = fix_json_with_gpt35(
            broken_json=annotation_str, api_key=api_key, base_url=base_url
        )
        if is_json_string_valid(annotation_str):
            annotation_obj = parse_json_string(annotation_str)
            if check_annotation_structure(annotation_obj):
                return annotation_obj

    # Try regeneration
    for _ in range(REGENERATION_ATTEMPTS):
        annotation_str = regenerate_annotation(
            dialog_graph=dialog_graph,
            prompt_instructions=prompt_instructions,
            model_name=model_name,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url,
            supports_user_role=supports_user_role,
        )
        if is_json_string_valid(annotation_str):
            annotation_obj = parse_json_string(annotation_str)
            if check_annotation_structure(annotation_obj):
                return annotation_obj

    return None


def annotate_multiple_graphs(
    data_list: List[Dict[str, Any]],
    prompt_instructions: str,
    model_name: str,
    temperature: float,
    api_key: str,
    base_url: str,
    supports_user_role: bool = True,
) -> List[Dict[str, Any]]:
    """
    data_list is expected to have elements like:
    {
      "graph": {...},
      "annotation": {...} (optional),
      "generated_graph": {...} (optional)
    }
    We'll annotate each "graph" if annotation is missing or if user wants to refresh it.
    """
    results = []
    for item in data_list:
        if "graph" not in item:
            # skip if no graph
            results.append(item)
            continue

        # Perform annotation
        annotation_obj = annotate_single_graph(
            dialog_graph=item["graph"],
            prompt_instructions=prompt_instructions,
            model_name=model_name,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url,
            supports_user_role=supports_user_role,
        )
        # Store it if not None
        if annotation_obj is not None:
            item["annotation"] = annotation_obj
        results.append(item)

    return results


def compute_dialog_metrics(dialog_graphs):
    """
    Вычисляет метрики для списка диалоговых графов:
    - max_dialog_depth: максимальное число переходов (ребер) в самом длинном пути
    - max_branching_factor: максимальное число исходящих переходов из одного узла
    - max_dialog_length: максимальное число узлов (ходов) в самом длинном пути

    :param dialog_graphs: список графов, каждый граф – словарь с ключом "graph",
                          содержащим "nodes" и "edges"
    :return: словарь с тремя ключами: 'max_dialog_depth', 'max_branching_factor', 'max_dialog_length'
    """
    overall_max_depth = 0  # максимальное число переходов среди всех графов
    overall_max_length = 0  # максимальное число узлов (ходов) среди всех графов
    overall_max_branch = 0  # максимальный branching factor среди всех графов

    for graph_obj in dialog_graphs:
        graph = graph_obj.get("graph", {})
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])

        # Создадим отображение: id узла -> список id соседей (исходящие ребра)
        adj = {}
        for node in nodes:
            node_id = node.get("id")
            adj[node_id] = []
        for edge in edges:
            src = edge.get("source")
            tgt = edge.get("target")
            if src in adj:
                adj[src].append(tgt)

        # Вычисляем max branching factor для текущего графа
        graph_max_branch = max(
            (len(neighbors) for neighbors in adj.values()), default=0
        )
        overall_max_branch = max(overall_max_branch, graph_max_branch)

        # Находим стартовые узлы (is_start == True)
        start_nodes = [node.get("id") for node in nodes if node.get("is_start")]

        graph_max_depth = 0  # максимальное число ребер в данном графе
        graph_max_length = 0  # максимальное число узлов в данном графе

        # Обходим граф DFS, избегая циклов
        def dfs(node_id, depth, visited):
            nonlocal graph_max_depth, graph_max_length
            if depth > graph_max_depth:
                graph_max_depth = depth
                graph_max_length = depth + 1  # число узлов = число ребер + 1
            for neighbor in adj.get(node_id, []):
                if neighbor not in visited:
                    dfs(neighbor, depth + 1, visited | {neighbor})

        for start in start_nodes:
            dfs(start, 0, {start})

        overall_max_depth = max(overall_max_depth, graph_max_depth)
        overall_max_length = max(overall_max_length, graph_max_length)

    return {
        "max_dialog_depth": overall_max_depth,
        "max_branching_factor": overall_max_branch,
        "max_dialog_length": overall_max_length,
    }


def construct_graph_from_annotation(annotation):
    """
    На основе аннотации строит упрощённое представление графа диалога.
    Предполагается, что диалог линейный и узлы берутся из mandatory_nodes,
    а для каждого узла берутся шаблоны из bot_utterances_templates.
    """
    mandatory_nodes = annotation.get("mandatory_nodes", [])
    bot_templates = annotation.get("bot_utterances_templates", {})

    nodes = []
    for i, node_label in enumerate(mandatory_nodes, start=1):
        nodes.append(
            {
                "id": i,
                "label": node_label,
                "is_start": (i == 1),
                "utterances": bot_templates.get(node_label, []),
            }
        )
    edges = []
    # Линейное соединение: от первого к последнему узлу по порядку
    for i in range(len(nodes) - 1):
        edges.append(
            {
                "source": nodes[i]["id"],
                "target": nodes[i + 1]["id"],
                "utterances": [],  # Переходы можно оставить без примеров
            }
        )

    return {"nodes": nodes, "edges": edges}


def update_annotation_with_metrics(input_dict):
    """
    Функция принимает на вход словарь с ключом "annotation". Если значения для
    max_dialog_depth, max_branching_factor или max_dialog_length равны "unknown",
    они вычисляются на основе структуры диалога (mandatory_nodes) и обновляются.

    :param input_dict: исходный словарь с аннотацией
    :return: обновлённый словарь с заполненными метриками, если они были неизвестны
    """
    annotation = input_dict.get("annotation", {})

    # Строим граф диалога из аннотации (линейное представление)
    graph = construct_graph_from_annotation(annotation)
    # Вычисляем метрики для одного графа
    metrics = compute_dialog_metrics([{"graph": graph}])

    if annotation.get("max_dialog_depth") == "unknown":
        annotation["max_dialog_depth"] = metrics["max_dialog_depth"]
    if annotation.get("max_branching_factor") == "unknown":
        annotation["max_branching_factor"] = metrics["max_branching_factor"]
    if annotation.get("max_dialog_length") == "unknown":
        annotation["max_dialog_length"] = metrics["max_dialog_length"]

    input_dict["annotation"] = annotation
    return input_dict

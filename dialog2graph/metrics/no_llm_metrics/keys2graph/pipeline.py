# experiments/exp2025_03_27_create_graphs_by_keys/keys2graph/pipeline.py

import json
from typing import List, Dict, Any, Optional, Union

import tiktoken

from .config import (
    FIX_ATTEMPTS,
    REGENERATION_ATTEMPTS,
    COST_TABLE,
    SEMANTIC_THRESHOLD,
    get_embedding_model,
)
from .prompts import ANNOTATION_INSTRUCTIONS
from .model_interactions import (
    generate_annotation,
    fix_json_with_gpt35,
    regenerate_annotation,
)
from .graph_generation import (
    generate_new_dialog_graph_from_annotation,
    generate_new_dialog_graph_from_dict,
)
from .json_validator import (
    is_json_string_valid,
    parse_json_string,
    check_annotation_structure,
)

from .metrics_semantic_jaccard import evaluate_graph_lists

# Импортируем модуль без LLM-метрик из dialog2graph
from dialog2graph.metrics.no_llm_metrics import metrics as no_llm_metrics

# Локальные semantic metrics и annotation metrics
from . import metrics_semantic
from . import metrics_annotation

from .graph_triplet_comparison import compare_two_graphs


def _load_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load a list of items from JSON file.
    If 'original_graph' has 'summary', remove it.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)
    return data_list


def _save_data(data_list: List[Dict[str, Any]], file_path: str) -> None:
    """
    Save list of items into a JSON file, overwriting.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data_list, f, ensure_ascii=False, indent=2)


def _estimate_tokens_in_chat_message(
    messages: List[Dict[str, str]], model_name: str
) -> int:
    """
    Example token counting with tiktoken.
    Concatenate role+content line by line, then encode and get length.
    """
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    combined_text = ""
    for msg in messages:
        combined_text += f"{msg.get('role', '')}:{msg.get('content', '')}\n"
    return len(encoding.encode(combined_text))


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


class Pipeline:
    """
    Full pipeline class.
    Stores loaded data (self._all_data), selected subset (self._selected_data),
    and provides methods for annotation, generation, comparison, etc.
    """

    def __init__(self):
        self._all_data: List[Dict[str, Any]] = []
        self._selected_indices: List[int] = []
        self._original_graphs: List[Dict[str, Any]] = []
        self._original_graphs_annotation: List[Dict[str, Any]] = []
        self._generated_graphs: List[Dict[str, Any]] = []
        self._generated_graphs_annotation: List[Dict[str, Any]] = []
        self._keys_generation: List[Dict[str, Any]] = []

    # 1) LOAD
    def load_dialog_graphs(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load a list of items from a JSON file, store in self._all_data, return them.
        """
        data_list = _load_data(file_path)
        self._all_data = data_list
        return data_list

    def load_graphs_for_comparison(
        self, original_file: str, generated_file: str
    ) -> None:
        """
        Загружает списки графов из двух JSON-файлов:
         - original_file -> self._original_graphs
         - generated_file -> self._generated_graphs
        """
        with open(original_file, "r", encoding="utf-8") as f_orig:
            self._original_graphs = json.load(f_orig)
        with open(generated_file, "r", encoding="utf-8") as f_gen:
            self._generated_graphs = json.load(f_gen)

        print(
            f"Loaded {len(self._original_graphs)} original graphs from {original_file}."
        )
        print(
            f"Loaded {len(self._generated_graphs)} generated graphs from {generated_file}."
        )

    # 2) SELECT
    def add_graphs_to_test(
        self, selection: Union[str, List[int]], output_file: str
    ) -> List[Dict[str, Any]]:
        """
        Choose a subset of loaded data by index or 'all',
        store in self._selected_data, return them.
        """
        if not self._all_data:
            print("No data loaded. Please call load_dialog_graphs first.")
            self._original_graphs = []
            return []

        if selection == "all":
            self._selected_indices = list(range(len(self._all_data)))
        else:
            self._selected_indices = [
                idx for idx in selection if 0 <= idx < len(self._all_data)
            ]

        self._original_graphs = [self._all_data[idx] for idx in self._selected_indices]

        # Удаляем summary, чтобы эта информация не "сбивала" систему при аннотации
        for item in self._original_graphs:
            if "summary" in item.keys():
                del item["summary"]

        print(f"Selected {len(self._original_graphs)} graphs for testing.")

        _save_data(self._original_graphs, output_file)

        return self._original_graphs

    # 3) ANNOTATE
    def annotate_graphs(
        self,
        model_name: str,
        temperature: float,
        output_file: str,
        graph_type: str,
        supports_user_role: bool = True,
    ) -> None:
        """
        Annotate only items that have either 'original_graph' or 'graph'.
        If none found, skip the item (silently).
        Save *only* annotated items to output_file.
        """

        selected_graphs = []

        if graph_type == "original":
            selected_graphs = self._original_graphs
        elif graph_type == "generated":
            selected_graphs = self._generated_graphs
        else:
            print("Unexpectable type of graph")

        for idx in range(len(selected_graphs)):
            graph_data = selected_graphs[idx]["graph"]

            # Generate annotation
            annotation_str = generate_annotation(
                dialog_graph=graph_data,
                prompt_instructions=ANNOTATION_INSTRUCTIONS,
                model_name=model_name,
                temperature=temperature,
                supports_user_role=supports_user_role,
            )

            # Validate/fix
            ann_obj = self._validate_annotation_string(annotation_str)

            if not ann_obj:
                for _ in range(FIX_ATTEMPTS):
                    annotation_str = fix_json_with_gpt35(
                        annotation_str, model_name="gpt-3.5-turbo"
                    )
                    ann_obj = self._validate_annotation_string(annotation_str)
                    if ann_obj:
                        break
            if not ann_obj:
                for _ in range(REGENERATION_ATTEMPTS):
                    annotation_str = regenerate_annotation(
                        dialog_graph=graph_data,
                        prompt_instructions=ANNOTATION_INSTRUCTIONS,
                        model_name=model_name,
                        temperature=temperature,
                        supports_user_role=supports_user_role,
                    )
                    ann_obj = self._validate_annotation_string(annotation_str)
                    if ann_obj:
                        break
            if ann_obj:
                ann_obj = update_annotation_with_metrics(ann_obj)
                if graph_type == "original":
                    self._original_graphs_annotation.append(ann_obj)
                else:
                    self._generated_graphs_annotation.append(ann_obj)
            else:
                print("Annotation failed")

        if graph_type == "original":
            _save_data(self._original_graphs_annotation, output_file)
            print(
                f"Annotation finished. {len(self._original_graphs_annotation)} items annotated."
            )
        else:
            _save_data(self._generated_graphs_annotation, output_file)
            print(
                f"Annotation finished. {len(self._generated_graphs_annotation)} items annotated."
            )
        print(f"Saved annotated items to {output_file}.")

    def _validate_annotation_string(
        self, annotation_str: str
    ) -> Optional[Dict[str, Any]]:
        """
        Check if annotation_str is valid JSON with the required structure.
        """
        if is_json_string_valid(annotation_str):
            obj = parse_json_string(annotation_str)
            if check_annotation_structure(obj):
                return obj
        return None

    # 4) GENERATE GRAPHS (BY KEYS)
    def generate_graphs_by_keys(
        self,
        keys: Union[List[str], str],
        model_name: str,
        temperature: float,
        output_file: str,
    ) -> None:
        """
        Generate new graphs from the annotation keys. Only applies to items
        that actually have item["annotation"] with a valid annotation dict.
        Save only changed items to output_file.
        """
        self._keys_generation = keys
        for item in self._original_graphs_annotation:
            if "annotation" not in item or not item["annotation"]:
                continue
            ann_dict = item["annotation"]
            if not ann_dict:
                continue

            # If keys == 'all', we take all keys from the annotation
            if isinstance(keys, str) and keys == "all":
                selected_keys = list(ann_dict.keys())
            else:
                selected_keys = keys

            gen_res = generate_new_dialog_graph_from_annotation(
                item=item,
                selected_keys=selected_keys,
                model_name=model_name,
                temperature=temperature,
            )
            if gen_res:
                self._generated_graphs.append(gen_res)
            else:
                print("Failed to generate new graph")

        _save_data(self._generated_graphs, output_file)
        print(
            f"Generation by keys completed. {len(self._generated_graphs)} items updated."
        )
        print(f"Saved to {output_file}.")

    # 5) GENERATE GRAPHS (FROM DICTS)
    def generate_graphs_from_dicts(
        self,
        list_of_key_dicts: List[Dict[str, Any]],
        model_name: str,
        temperature: float,
        output_file: str,
    ) -> None:
        """
        Create entirely new items in self._all_data for each dict in list_of_key_dicts,
        each with "generated_graph" as result.
        Save those newly created items to output_file.
        """
        new_items = []

        for key_dict in list_of_key_dicts:
            new_graph = generate_new_dialog_graph_from_dict(
                keys_dict=key_dict, model_name=model_name, temperature=temperature
            )
            if new_graph:
                new_item = {
                    "original_graph": {},
                    "annotation": None,
                    "generated_graph": new_graph,
                }
                self._all_data.append(new_item)
                new_items.append(new_item)
            else:
                print("Failed to generate graph from one of the dictionaries.")

        _save_data(new_items, output_file)
        print(
            f"Generated {len(new_items)} new graph(s) from dicts. Saved to {output_file}."
        )

    # 6) ESTIMATE COST
    def estimate_costs_for_selected_graphs(
        self, model_name: str, approximate_completion_ratio: float = 1.0
    ) -> None:
        """
        Roughly estimate token usage for annotation of selected items.
        Just sums tokens for system+user messages with the original_graph/graph.
        """
        prompt_rate = COST_TABLE.get(model_name, {}).get("prompt", 0.0)
        completion_rate = COST_TABLE.get(model_name, {}).get("completion", 0.0)

        total_prompt_tokens = 0
        for idx in self._selected_indices:
            item = self._all_data[idx]
            graph_data = None
            if "original_graph" in item and item["original_graph"]:
                graph_data = item["original_graph"]
            elif "graph" in item and item["graph"]:
                graph_data = item["graph"]

            if not graph_data:
                continue

            graph_str = json.dumps(graph_data, ensure_ascii=False, indent=2)
            messages = [
                {"role": "system", "content": ANNOTATION_INSTRUCTIONS},
                {"role": "user", "content": f"DIALOG GRAPHS:\n{graph_str}"},
            ]
            tokens = _estimate_tokens_in_chat_message(messages, model_name)
            total_prompt_tokens += tokens

        completion_tokens = int(total_prompt_tokens * approximate_completion_ratio)
        cost_prompt = (total_prompt_tokens / 1000.0) * prompt_rate
        cost_completion = (completion_tokens / 1000.0) * completion_rate
        total_cost = cost_prompt + cost_completion

        print("=== ESTIMATED COST FOR SELECTED GRAPHS ===")
        print(f"Model: {model_name}")
        print(f"Selected graphs count: {len(self._selected_data)}")
        print(f"Approx prompt tokens: {total_prompt_tokens}")
        print(f"Prompt cost: ${cost_prompt:.4f}")
        print(f"Completion cost (approx): ${cost_completion:.4f}")
        print(f"Total approx cost: ${total_cost:.4f}")
        print("==========================================")

    # 7) COMPARE GENERATED (DIRECT + SEMANTIC)
    def compare_generated_graphs(
        self, output_metrics_file: str, threshold: float = SEMANTIC_THRESHOLD
    ) -> None:
        """
        Compare original_graph vs generated_graph using:
         - direct metrics (metrics.triplet_match_accuracy)
         - semantic metrics (metrics_semantic.triplet_match_accuracy_semantic, compare_two_graphs_semantically)

        Only on items that have both original_graph (or graph) and generated_graph.
        Save results in output_metrics_file.
        """
        results = []
        direct_node_accs = []
        direct_edge_accs = []
        semantic_node_accs = []
        semantic_edge_accs = []
        sem_j_nodes_list = []
        sem_j_edges_list = []

        from dialog2graph.pipelines.core.graph import BaseGraph

        for idx in self._selected_indices:
            item = self._all_data[idx]
            # Find the original or graph
            orig_data = None
            if "original_graph" in item and item["original_graph"]:
                orig_data = item["original_graph"]
            elif "graph" in item and item["graph"]:
                orig_data = item["graph"]
            if not orig_data:
                continue

            # Must also have generated_graph
            if "generated_graph" not in item or not item["generated_graph"]:
                continue

            gen_data = item["generated_graph"]["graph"]

            g1 = BaseGraph(graph_dict=orig_data)
            g2 = BaseGraph(graph_dict=gen_data)

            # Direct
            direct_metrics = no_llm_metrics.triplet_match_accuracy(g1, g2)
            d_node = direct_metrics["node_accuracy"]
            d_edge = direct_metrics["edge_accuracy"]
            direct_node_accs.append(d_node)
            direct_edge_accs.append(d_edge)

            # Semantic
            sem_metrics = metrics_semantic.triplet_match_accuracy_semantic(
                g1, g2, threshold=threshold, embed_model=get_embedding_model()
            )
            s_node = sem_metrics["node_accuracy"]
            s_edge = sem_metrics["edge_accuracy"]
            semantic_node_accs.append(s_node)
            semantic_edge_accs.append(s_edge)

            # Semantic jaccard
            jaccard_dict = metrics_semantic.compare_two_graphs_semantically(
                graph1=orig_data, graph2=gen_data, threshold=threshold
            )
            sem_j_nodes_list.append(jaccard_dict["semantic_jaccard_nodes"])
            sem_j_edges_list.append(jaccard_dict["semantic_jaccard_edges"])

            results.append(
                {
                    "index": idx,
                    "direct_node_accuracy": d_node,
                    "direct_edge_accuracy": d_edge,
                    "semantic_node_accuracy": s_node,
                    "semantic_edge_accuracy": s_edge,
                    "semantic_jaccard_nodes": jaccard_dict["semantic_jaccard_nodes"],
                    "semantic_jaccard_edges": jaccard_dict["semantic_jaccard_edges"],
                }
            )

        def avg(lst):
            return sum(lst) / len(lst) if lst else 0.0

        summary = {
            "average_direct_node_accuracy": avg(direct_node_accs),
            "average_direct_edge_accuracy": avg(direct_edge_accs),
            "average_semantic_node_accuracy": avg(semantic_node_accs),
            "average_semantic_edge_accuracy": avg(semantic_edge_accs),
            "average_semantic_jaccard_nodes": avg(sem_j_nodes_list),
            "average_semantic_jaccard_edges": avg(sem_j_edges_list),
            "count_items": len(results),
        }

        final_data = {"per_graph": results, "summary": summary}
        _save_data(final_data, output_metrics_file)
        print(f"Comparison results saved to {output_metrics_file}.")

    # 9) COMPARE ANNOTATIONS (ORIGINAL VS GENERATED)
    def compare_annotations_of_original_and_generated(self, output_file: str) -> None:
        """
        Compare annotation["annotation"] of original vs generated_graph_annotation["annotation"].
        Use metrics_annotation.compare_annotation_differences to measure numeric diffs and semantic sim.
        Save results array to output_file.
        """
        comparison_results = []

        for idx in range(len(self._generated_graphs)):
            gen_anno = self._generated_graphs_annotation[idx]["annotation"]
            orig_anno = self._original_graphs_annotation[idx]["annotation"]
            diff_res = metrics_annotation.compare_annotation_differences(
                orig_anno, gen_anno
            )

        for idx in self._selected_indices:
            item = self._all_data[idx]
            if "annotation" not in item or not item["annotation"]:
                # No original annotation
                continue
            if (
                "generated_graph_annotation" not in item
                or not item["generated_graph_annotation"]
            ):
                # Not validated or no annotation for generated
                continue

            orig_anno = item["annotation"]["annotation"]
            gen_anno = item["generated_graph_annotation"]["annotation"]

            diff_res = metrics_annotation.compare_annotation_differences(
                orig_anno, gen_anno
            )
            comparison_results.append({"index": idx, "details": diff_res})

        _save_data(comparison_results, output_file)
        print(f"Annotation-level comparison saved to {output_file}.")

    # 10) COMPARE VIA SEMANTIC‑JACCARD
    def compare_jaccard_similarity(
        self, model_name: str = "o1-mini", output_file: str = "jacard_metrics.json"
    ):
        """
        Сравнивает self._original_graphs и self._generated_graphs
        по Semantic‑Jaccard метрикам (узлы и рёбра).

        Пишет детальный JSON в `output_file` и
        выводит средние значения в stdout.
        """
        if not self._original_graphs or not self._generated_graphs:
            print("Списки графов пусты. Сначала выполните генерацию/загрузку графов.")
            return
        if len(self._original_graphs) != len(self._generated_graphs):
            print("Длины списков графов не совпадают — сравнение невозможно.")
            return

        _, avg_nodes, avg_edges = evaluate_graph_lists(
            original_graphs=self._original_graphs,
            generated_graphs=self._generated_graphs,
            output_path=output_file,
            model_name=model_name,
        )

        print("\n=== Semantic‑Jaccard similarity (avg over all pairs) ===")
        print(f"Nodes: {avg_nodes:.3f}")
        print(f"Edges: {avg_edges:.3f}")
        print(f"Metrics saved to {output_file}")

    def calculate_graphs_similarity(self, output_file: str) -> None:
        if len(self._original_graphs) != len(self._generated_graphs):
            print("Lengths of original and generated graphs do not match.")
            return

        pair_results, scores = [], []
        for i, (orig, gen) in enumerate(
            zip(self._original_graphs, self._generated_graphs)
        ):
            if "graph" not in orig or "graph" not in gen:
                continue
            comp = compare_two_graphs(orig["graph"], gen["graph"])  # NEW CALL
            pair_results.append(
                {
                    "pair_index": i,
                    "similarity_avg": comp["similarity_avg"],
                    "matched_triplets": comp["matched_triplets"],
                }
            )
            scores.append(comp["similarity_avg"])

        final = {
            "pairs": pair_results,
            "overall_average_similarity": float(sum(scores) / len(scores))
            if scores
            else 0.0,
        }
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final, f, ensure_ascii=False, indent=2)
        print(f"[Triplet‑similarity] saved → {output_file}")

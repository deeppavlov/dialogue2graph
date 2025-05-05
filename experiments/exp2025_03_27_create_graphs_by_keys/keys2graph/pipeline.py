# experiments/exp2025_03_27_create_graphs_by_keys/keys2graph/pipeline.py

"""
High-level orchestration for the *keys-to-graph* workflow.  
The :class:`Pipeline` object lets you

1. **Load** original dialogue graphs from file.
2. **Select** a subset for experiments.
3. **Annotate** graphs with a large language model.
4. **Generate** new graphs from selected annotation keys or from an explicit
   ``dict`` of keys.
5. **Compare** original and generated graphs via triplet-level similarity.
6. **Compute** structural metrics such as maximum depth or branching factor.
"""

import json
from typing import List, Dict, Any, Optional, Union

from .config import (
    FIX_ATTEMPTS,
    REGENERATION_ATTEMPTS
)
from .prompts import ANNOTATION_INSTRUCTIONS
from .model_interactions import (
    generate_annotation,
    fix_json_with_gpt35,
    regenerate_annotation
)
from .graph_generation import (
    generate_new_dialog_graph_from_annotation,
    generate_new_dialog_graph_from_dict
)
from .json_validator import (
    is_json_string_valid,
    parse_json_string,
    check_annotation_structure
)

from .graph_triplet_comparison import compare_two_graphs


def _load_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load a list of graph objects from *file_path*.

    Each element **must** contain a top-level key ``"graph"`` that follows the
    canonical structure (``nodes`` / ``edges``).  If an object also contains
    ``"summary"``, that field is preserved for now – it can be removed later
    when selecting graphs for annotation to avoid leaking high-level
    information to the model.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
    return data_list


def _save_data(data_list: List[Dict[str, Any]], file_path: str) -> None:
    """
    Overwrite *file_path* with *data_list* encoded as pretty-printed JSON.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=2)

def compute_dialog_metrics(dialog_graphs):
    """
    Calculate simple structural metrics for a list of graphs.

    Returned keys
    -------------
    max_dialog_depth
        Maximum **number of edges** on the deepest path across all graphs.
    max_branching_factor
        Maximum out-degree (number of outgoing transitions) for any node.
    max_dialog_length
        Maximum **number of nodes** on the deepest path (depth + 1).

    Parameters
    ----------
    dialog_graphs
        Iterable with objects that wrap a ``"graph"`` dict.

    Notes
    -----
    Loops are ignored by keeping a ``visited`` set during DFS.
    """
    overall_max_depth = 0       
    overall_max_length = 0      
    overall_max_branch = 0     

    for graph_obj in dialog_graphs:
        graph = graph_obj.get("graph", {})
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])

        # Build adjacency list: node_id → [neighbour ids]
        adj = {}
        for node in nodes:
            node_id = node.get("id")
            adj[node_id] = []
        for edge in edges:
            src = edge.get("source")
            tgt = edge.get("target")
            if src in adj:
                adj[src].append(tgt)

        # Branching factor for this graph
        graph_max_branch = max((len(neighbors) for neighbors in adj.values()), default=0)
        overall_max_branch = max(overall_max_branch, graph_max_branch)

        # Find starting nodes (is_start == True)
        start_nodes = [node.get("id") for node in nodes if node.get("is_start")]

        graph_max_depth = 0   # edges
        graph_max_length = 0  # nodes

        def dfs(node_id, depth, visited):
            nonlocal graph_max_depth, graph_max_length
            if depth > graph_max_depth:
                graph_max_depth = depth
                graph_max_length = depth + 1
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
        "max_dialog_length": overall_max_length
    }


class Pipeline:
    """
    Convenience wrapper that keeps state between steps of the experiment.
    """
    def __init__(self):
        self._all_data: List[Dict[str, Any]] = []
        self._selected_indices: List[int] = []
        self._original_graphs: List[Dict[str, Any]] = []
        self._original_graphs_annotation: List[Dict[str, Any]] = []
        self._generated_graphs: List[Dict[str, Any]] = []
        self._generated_graphs_annotation: List[Dict[str, Any]] = []
        self._keys_generation: List[Dict[str, Any]] = []

    def load_dialog_graphs(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Read a *single* JSON file of graph items and keep it in memory.

        :returns: The parsed list (also stored in ``self._all_data``).
        """
        data_list = _load_data(file_path)
        self._all_data = data_list
        return data_list
    
    def load_graphs_for_comparison(self, original_file: str, generated_file: str) -> None:
        """
        Load two lists of graphs – one *original*, one *generated* – that
        are expected to have the same length and positional alignment.
        """
        with open(original_file, "r", encoding="utf-8") as f_orig:
            self._original_graphs = json.load(f_orig)
        with open(generated_file, "r", encoding="utf-8") as f_gen:
            self._generated_graphs = json.load(f_gen)

        print(f"Loaded {len(self._original_graphs)} original graphs from {original_file}.")
        print(f"Loaded {len(self._generated_graphs)} generated graphs from {generated_file}.")

    def add_graphs_to_test(self, selection: Union[str, List[int]], output_file: str) -> List[Dict[str, Any]]:
        """
        Select a subset of ``self._all_data`` for downstream steps.

        * ``selection='all'`` – take everything.
        * ``selection=[0, 4, 7]`` – take explicit indices.

        The resulting list is saved to *output_file* (with ``summary`` fields
        removed so that the LLM does not see condensed context).
        """
        if not self._all_data:
            print("No data loaded. Please call load_dialog_graphs first.")
            self._original_graphs = []
            return []
        
        if selection == "all":
            self._selected_indices = list(range(len(self._all_data)))
        else:
            self._selected_indices = [idx for idx in selection if 0 <= idx < len(self._all_data)]
        
        self._original_graphs = [self._all_data[idx] for idx in self._selected_indices]

        # Remove summaries to avoid leaking answers to the model
        for item in self._original_graphs:
            if "summary" in item.keys():
                del item["summary"]

        print(f"Selected {len(self._original_graphs)} graphs for testing.")

        _save_data(self._original_graphs, output_file)

        return self._original_graphs

    def annotate_graphs(
        self,
        model_name: str,
        temperature: float,
        output_file: str,
        graph_type: str,
        supports_user_role: bool = True
    ) -> None:
        """
        Annotate either the *original* or the *generated* graphs.

        Parameters
        ----------
        graph_type
            ``"original"`` or ``"generated"``.
        """
        selected_graphs = []

        if graph_type == "original":
            selected_graphs = self._original_graphs
        elif graph_type == "generated":
            selected_graphs = self._generated_graphs
        else:
            print('Unexpectable type of graph')

        for idx in range(len(selected_graphs)):
            graph_data = selected_graphs[idx]["graph"]

            # 1. Ask the LLM for an annotation
            annotation_str = generate_annotation(
                dialog_graph=graph_data,
                prompt_instructions=ANNOTATION_INSTRUCTIONS,
                model_name=model_name,
                temperature=temperature,
                supports_user_role=supports_user_role
            )

            # 2. Validate ⇒ attempt quick fixes ⇒ full regeneration
            ann_obj = self._validate_annotation_string(annotation_str)

            if not ann_obj:
                for _ in range(FIX_ATTEMPTS):
                    annotation_str = fix_json_with_gpt35(annotation_str, model_name="gpt-3.5-turbo")
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
                        supports_user_role=supports_user_role
                    )
                    ann_obj = self._validate_annotation_string(annotation_str)
                    if ann_obj:
                        break

            # Persist the result
            if ann_obj:
                if graph_type == "original":
                    self._original_graphs_annotation.append(ann_obj)
                else:
                    self._generated_graphs_annotation.append(ann_obj)
            else:
                print(f"Annotation failed")

        if graph_type == "original":
            _save_data(self._original_graphs_annotation, output_file)
            print(f"Annotation finished. {len(self._original_graphs_annotation)} items annotated.")
        else:
            _save_data(self._generated_graphs_annotation, output_file)
            print(f"Annotation finished. {len(self._generated_graphs_annotation)} items annotated.")
        print(f"Saved annotated items to {output_file}.")

    def _validate_annotation_string(self, annotation_str: str) -> Optional[Dict[str, Any]]:
        """
        Internal helper – check JSON syntax **and** presence of all required keys.
        """
        if is_json_string_valid(annotation_str):
            obj = parse_json_string(annotation_str)
            if check_annotation_structure(obj):
                return obj
        return None

    def generate_graphs_by_keys(
        self,
        keys: Union[List[str], str],
        model_name: str,
        temperature: float,
        output_file: str
    ) -> None:
        """
        For every *annotated* original graph create a new graph using the
        specified *keys* and store the outcome in :pyattr:`_generated_graphs`.

        * ``keys='all'`` – forward **every** annotation key.
        * ``keys=['topic', 'bot_goal']`` – forward only the listed ones.
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
                temperature=temperature
            )
            if gen_res:
                self._generated_graphs.append(gen_res)
            else:
                print(f"Failed to generate new graph")

        _save_data(self._generated_graphs, output_file)
        print(f"Generation by keys completed. {len(self._generated_graphs)} items updated.")
        print(f"Saved to {output_file}.")

    def generate_graphs_from_dicts(
        self,
        list_of_key_dicts: List[Dict[str, Any]],
        model_name: str,
        temperature: float,
        output_file: str
    ) -> None:
        """
        Generate *brand-new* graphs from arbitrary key-value dictionaries.

        Each dict is treated independently; no original graph is required.
        """
        new_items = []

        for key_dict in list_of_key_dicts:
            new_graph = generate_new_dialog_graph_from_dict(
                keys_dict=key_dict,
                model_name=model_name,
                temperature=temperature
            )
            if new_graph:
                new_item = {
                    "original_graph": {},
                    "annotation": None,
                    "generated_graph": new_graph
                }
                self._all_data.append(new_item)
                new_items.append(new_item)
            else:
                print("Failed to generate graph from one of the dictionaries.")

        _save_data(new_items, output_file)
        print(f"Generated {len(new_items)} new graph(s) from dicts. Saved to {output_file}.")

    def calculate_graphs_similarity(self, output_file: str) -> None:
        """
        Compute triplet-level similarity for *aligned* pairs of original and
        generated graphs and write a detailed report to *output_file*.
        """
        if len(self._original_graphs) != len(self._generated_graphs):
            print("Lengths of original and generated graphs do not match.")
            return

        pair_results, scores = [], []
        for i, (orig, gen) in enumerate(zip(self._original_graphs, self._generated_graphs)):
            if "graph" not in orig or "graph" not in gen:
                continue
            comp = compare_two_graphs(orig["graph"], gen["graph"]) 
            pair_results.append({
                "pair_index": i,
                "similarity_avg": comp["similarity_avg"],
                "matched_triplets": comp["matched_triplets"]
            })
            scores.append(comp["similarity_avg"])

        final = {
            "pairs": pair_results,
            "overall_average_similarity": float(sum(scores)/len(scores)) if scores else 0.0
        }
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final, f, ensure_ascii=False, indent=2)
        print(f"[Triplet‑similarity] saved → {output_file}")

    

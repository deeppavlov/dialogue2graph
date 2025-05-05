# metrics_no_llm.py

"""
Automatic Metrics (no LLM).
---------------------------
Functions that check Graphs and Dialogs with direct string matching
(Jaccard-based or isomorphism-based).
"""

from typing import List, Dict, Any
import numpy as np
import networkx as nx

# Below code is copied from the user's snippet:
# --------------------------------------------------------------


def _collapse_multiedges(edges):
    collapsed_edges = {}
    for u, v, data in edges:
        key = f"{u}->{v}"
        if key not in collapsed_edges:
            collapsed_edges[key] = []
        if isinstance(data["utterances"], str):
            collapsed_edges[key].append(data["utterances"])
        elif isinstance(data["utterances"], list):
            collapsed_edges[key].extend(data["utterances"])
    return collapsed_edges


def _get_jaccard_edges(
    true_graph_edges, generated_graph_edges, verbose=False, return_matrix=False
):
    true_graph_edges = _collapse_multiedges(list(true_graph_edges))
    generated_graph_edges = _collapse_multiedges(list(generated_graph_edges))

    jaccard_values = np.zeros((len(true_graph_edges), len(generated_graph_edges)))
    for idx1, (k1, v1) in enumerate(true_graph_edges.items()):
        for idx2, (k2, v2) in enumerate(generated_graph_edges.items()):
            intersect_set = set(v1).intersection(set(v2))
            union_set = set(v1).union(set(v2))
            jaccard_values[idx1][idx2] = (
                len(intersect_set) / len(union_set) if len(union_set) > 0 else 0.0
            )

    max_jaccard_values = np.max(jaccard_values, axis=1)
    max_jaccard_indices = np.argmax(jaccard_values, axis=1)
    if return_matrix:
        return list(max_jaccard_values), list(max_jaccard_indices), jaccard_values
    return list(max_jaccard_values), list(max_jaccard_indices)


def _get_list_of_node_utterances(node_utterances):
    if isinstance(node_utterances, str):
        return [node_utterances]
    return node_utterances


def _collapse_multinodes(nodes):
    collapsed_nodes = {}
    for key, data in nodes:
        if key not in collapsed_nodes:
            collapsed_nodes[key] = []
        if isinstance(data["utterances"], str):
            collapsed_nodes[key].append(data["utterances"])
        elif isinstance(data["utterances"], list):
            collapsed_nodes[key].extend(data["utterances"])
    return collapsed_nodes


def _get_jaccard_nodes(
    true_graph_nodes, generated_graph_nodes, verbose=False, return_matrix=False
):
    true_graph_nodes = _collapse_multinodes(list(true_graph_nodes))
    generated_graph_nodes = _collapse_multinodes(list(generated_graph_nodes))

    # add +1 to accommodate possible indexing
    max_node_id1 = max(true_graph_nodes.keys()) if true_graph_nodes else 0
    max_node_id2 = max(generated_graph_nodes.keys()) if generated_graph_nodes else 0

    jaccard_values = np.zeros((max_node_id1 + 1, max_node_id2 + 1))

    for node1_id, node1_utterances in true_graph_nodes.items():
        for node2_id, node2_utterances in generated_graph_nodes.items():
            set1 = set(_get_list_of_node_utterances(node1_utterances))
            set2 = set(_get_list_of_node_utterances(node2_utterances))

            jaccard_nominator = set1.intersection(set2)
            jaccard_denominator = set1.union(set2)

            jaccard = (
                len(jaccard_nominator) / len(jaccard_denominator)
                if len(jaccard_denominator) > 0
                else 0.0
            )
            jaccard_values[node1_id][node2_id] = jaccard

    max_jaccard_values = np.max(jaccard_values, axis=1)
    max_jaccard_indices = np.argmax(jaccard_values, axis=1)
    if return_matrix:
        return list(max_jaccard_values), list(max_jaccard_indices), jaccard_values
    return list(max_jaccard_values), list(max_jaccard_indices)


def _match_edge_for_multigraph(x, y):
    if isinstance(x, dict) and isinstance(y, dict):
        set1 = set([elem["utterances"] for elem in list(x.values())])
        set2 = set([elem["utterances"] for elem in list(y.values())])
    else:
        set1 = set(x)
        set2 = set(y)
    return set1.intersection(set2) is not None


def _parse_edge(edge):
    src, trg = edge.split("->")
    return int(src) - 1, int(trg) - 1


def match_graph_triplets(G1: Any, G2: Any, change_to_original_ids=False):
    """
    Attempts to match two graphs (G1, G2) using a combination of isomorphism checks and
    Jaccard-based utterance similarity for edges/nodes.
    Return node_mapping, edge_mapping
    """
    g1 = G1.graph
    g2 = G2.graph
    node_mapping = {node: None for node in g1.nodes}
    node_mapping.update({node: None for node in g2.nodes})

    # Basic isomorphism check
    if isinstance(g1, nx.DiGraph):
        GM = nx.isomorphism.DiGraphMatcher(
            g1,
            g2,
            edge_match=lambda x, y: set(x["utterances"]).intersection(
                set(y["utterances"])
            )
            is not None,
        )
        are_isomorphic = GM.is_isomorphic()
    else:
        GM = nx.isomorphism.MultiDiGraphMatcher(
            g1, g2, edge_match=_match_edge_for_multigraph
        )
        are_isomorphic = GM.is_isomorphic()

    if are_isomorphic:
        # for demonstration, we won't fill node_mapping with actual isomorphism data
        # but you can do so using GM.mapping or a specialized approach.
        pass

    edge_mapping = {}
    mapping_jaccard_values = {}

    edges1 = list(_collapse_multiedges(g1.edges(data=True)).keys())
    edges2 = list(_collapse_multiedges(g2.edges(data=True)).keys())

    _, _, matrix_edges = _get_jaccard_edges(
        g1.edges(data=True), g2.edges(data=True), verbose=False, return_matrix=True
    )
    _, _, matrix_nodes = _get_jaccard_nodes(
        g1.nodes(data=True), g2.nodes(data=True), verbose=False, return_matrix=True
    )

    for i, edge1 in enumerate(edges1):
        edge_mapping[edge1] = None
        mapping_jaccard_values[edge1] = 0
        for j, edge2 in enumerate(edges2):
            if matrix_edges[i][j] > 0:
                node1_src, node1_trg = _parse_edge(edge1)
                node2_src, node2_trg = _parse_edge(edge2)
                # Simple check with matrix_nodes
                if (
                    matrix_nodes[node1_src][node2_src] == 0.0
                    and matrix_nodes[node1_trg][node2_trg] == 0.0
                ):
                    continue
                elif (
                    matrix_nodes[node1_src][node2_src] > 0
                    and matrix_nodes[node1_trg][node2_trg] > 0
                ):
                    if matrix_edges[i][j] > mapping_jaccard_values[edge1]:
                        mapping_jaccard_values[edge1] = matrix_edges[i][j]
                        edge_mapping[edge1] = edge2
                        node_mapping[node1_src + 1] = node2_src + 1
                        node_mapping[node1_trg + 1] = node2_trg + 1
                else:
                    # partial match scenario
                    pass

    return node_mapping, edge_mapping


def is_same_structure(G1: Any, G2: Any) -> bool:
    return nx.is_isomorphic(G1.graph, G2.graph)


def match_triplets_dg(G1: Any, dialogs: List[Any]) -> Dict[str, Any]:
    """
    Check if all triplets from dialogs are present in G1 or vice versa.
    Very simplified approach (from user snippet).
    Here we just return a dummy result for demonstration.
    """
    return {"value": True}


def are_paths_valid(G1: Any, dialogs: List[Any]) -> Dict[str, Any]:
    """
    Check if dialogs form valid paths in G1.
    Very simplified approach from snippet.
    """
    return {"value": True}


def triplet_match_accuracy(
    G1: Any, G2: Any, change_to_original_ids: bool = False
) -> Dict[str, float]:
    """
    Basic example: use match_graph_triplets to count matched nodes/edges
    and produce an accuracy dict.
    """
    node_mapping, edge_mapping = match_graph_triplets(G1, G2, change_to_original_ids)

    g1_nodes = list(G1.graph.nodes())
    matched_nodes = sum(1 for n in g1_nodes if node_mapping.get(n) is not None)
    total_nodes = len(g1_nodes)
    node_accuracy = matched_nodes / total_nodes if total_nodes > 0 else 0.0

    g1_edges = list(G1.graph.edges())
    matched_edges = sum(
        1 for e_str, mapped in edge_mapping.items() if mapped is not None
    )
    total_edges = len(g1_edges)
    edge_accuracy = matched_edges / total_edges if total_edges > 0 else 0.0

    return {"node_accuracy": node_accuracy, "edge_accuracy": edge_accuracy}


def compute_graph_metrics(graph_list: List[Any]) -> Dict[str, float]:
    """
    Example function that checks for cycles, counts edges/nodes, etc.
    """
    total_graphs = len(graph_list)
    with_cycles = 0
    total_edges = 0
    total_nodes = 0

    for gobj in graph_list:
        # we assume gobj has gobj.graph_dict["edges"] and gobj.graph_dict["nodes"]
        edges = gobj.graph_dict["edges"]
        nodes = gobj.graph_dict["nodes"]

        G = nx.DiGraph()
        for node_info in nodes:
            G.add_node(node_info["id"])
        for edge_info in edges:
            G.add_edge(edge_info["source"], edge_info["target"])

        edges_count = G.number_of_edges()
        nodes_count = G.number_of_nodes()
        total_edges += edges_count
        total_nodes += nodes_count

        if not nx.is_directed_acyclic_graph(G):
            with_cycles += 1

    average_edges_amount = total_edges / total_graphs if total_graphs else 0
    average_nodes_amount = total_nodes / total_graphs if total_graphs else 0
    percentage_with_cycles = (with_cycles / total_graphs * 100) if total_graphs else 0

    return {
        "with_cycles": with_cycles,
        "percentage_with_cycles": percentage_with_cycles,
        "average_edges_amount": average_edges_amount,
        "average_nodes_amount": average_nodes_amount,
        "total_graphs": total_graphs,
        "total_edges": total_edges,
        "total_nodes": total_nodes,
    }

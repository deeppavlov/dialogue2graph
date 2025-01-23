"""
Automatic Metrics.
------------------

This module contains functions that automatically (without using LLMs) checks Graphs and Dialogues
for various metrics.
"""

import networkx as nx
from chatsky_llm_autoconfig.metrics.jaccard import jaccard_edges, jaccard_nodes, collapse_multiedges
from chatsky_llm_autoconfig.graph import BaseGraph
from chatsky_llm_autoconfig.dialogue import Dialogue


def edge_match_for_multigraph(x, y):
    if isinstance(x, dict) and isinstance(y, dict):
        set1 = set([elem["utterances"] for elem in list(x.values())])
        set2 = set([elem["utterances"] for elem in list(y.values())])
    else:
        set1 = set(x)
        set2 = set(y)
    return set1.intersection(set2) is not None


def parse_edge(edge):
    src, trg = map(int, edge.split("->"))
    return src - 1, trg - 1


def triplet_match(G1: BaseGraph, G2: BaseGraph, change_to_original_ids=False):
    g1 = G1.graph
    g2 = G2.graph
    node_mapping = {node: None for node in g1.nodes}
    node_mapping.update({node: None for node in g2.nodes})
    if type(g1) is nx.DiGraph():
        GM = nx.isomorphism.DiGraphMatcher(g1, g2, edge_match=lambda x, y: set(x["utterances"]).intersection(set(y["utterances"])) is not None)
        are_isomorphic = GM.is_isomorphic()
    else:
        GM = nx.isomorphism.MultiDiGraphMatcher(g1, g2, edge_match=edge_match_for_multigraph)
        are_isomorphic = GM.is_isomorphic()
    if are_isomorphic:
        print("Graphs are isomorphic")
        node_mapping = nx.vf2pp_isomorphism(g1, g2, node_label=None)

    edge_mapping = {}
    mapping_jaccard_values = {}

    edges1 = list(collapse_multiedges(g1.edges(data=True)).keys())
    edges2 = list(collapse_multiedges(g2.edges(data=True)).keys())

    _, _, matrix_edges = jaccard_edges(g1.edges(data=True), g2.edges(data=True), verbose=False, return_matrix=True)

    _, _, matrix_nodes = jaccard_nodes(g1.nodes(data=True), g2.nodes(data=True), verbose=False, return_matrix=True)

    for i, edge1 in enumerate(edges1):
        edge_mapping[edge1] = None
        mapping_jaccard_values[edge1] = 0
        for j, edge2 in enumerate(edges2):
            if matrix_edges[i][j] > 0:
                node1_src, node1_trg = parse_edge(edge1)
                node2_src, node2_trg = parse_edge(edge2)
                if matrix_nodes[node1_src][node2_src] == 0.0 and matrix_nodes[node1_trg][node2_trg] == 0.0:
                    continue
                elif matrix_nodes[node1_src][node2_src] > 0 and matrix_nodes[node1_trg][node2_trg] > 0:
                    if matrix_edges[i][j] > mapping_jaccard_values[edge1]:
                        mapping_jaccard_values[edge1] = matrix_edges[i][j]
                        edge_mapping[edge1] = edge2
                        node_mapping[node1_src + 1] = node2_src + 1
                        node_mapping[node1_trg + 1] = node2_trg + 1
                else:
                    node1_src_nx = g1.nodes[node1_src + 1]
                    node2_src_nx = g2.nodes[node2_src + 1]
                    if node1_src_nx == node2_src_nx:
                        node_mapping[node1_src + 1] = node2_src + 1

                    node1_trg_nx = g1.nodes[node1_trg + 1]
                    node2_trg_nx = g2.nodes[node2_trg + 1]
                    if node1_trg_nx == node2_trg_nx:
                        node_mapping[node1_trg + 1] = node2_trg + 1
                    print(
                        f"""The nodes of edges {edges1[i]} and {edges2[j]} has something in common, but not complete match: Sources: {
                            node1_src_nx["utterances"]}, {node2_src_nx["utterances"]}"""
                    )
                    print(
                        f"""The nodes of edges {edges1[i]} and {edges2[j]} has something in common, but not complete match: Targets: {
                            node1_trg_nx["utterances"]}, {node2_trg_nx["utterances"]}"""
                    )

    if G1.node_mapping != {} and change_to_original_ids:
        new_node_mapping = {}
        new_edge_mapping = {}

        # какому ключу в старом графе соовтетвует новый ключ в перенумерованном графе
        inverse_mapping = {v: k for k, v in G1.node_mapping.items()}
        # {1: 1, 3: 2} -> {1: 1, 4:2} если в g1 4 перенумеровалась в 3
        for k, v in node_mapping.items():
            if inverse_mapping.get(k) is None and v is None:
                new_node_mapping[k] = v
            elif inverse_mapping.get(k) is None:
                raise ValueError("Invalid renumeration")
            else:
                new_node_mapping[inverse_mapping[k]] = v

        for edge1, edge2 in edge_mapping.items():
            src1, trg1 = edge1.split("->")
            new_edge_mapping[
                f"""{
                inverse_mapping[int(src1)]}->{inverse_mapping[int(trg1)]}"""
            ] = edge2
        return new_node_mapping, new_edge_mapping

    return node_mapping, edge_mapping


def is_same_structure(G1: BaseGraph, G2: BaseGraph) -> bool:
    g1 = G1.graph
    g2 = G2.graph
    return nx.is_isomorphic(g1, g2)


def all_paths_sampled(G: BaseGraph, dialogue: Dialogue) -> bool:
    return True


def all_utterances_present(G: BaseGraph, dialogues: list[Dialogue]) -> bool:
    """
    Check if all graph elements (nodes and edges) appear in at least one dialogue.

    Args:
        G: BaseGraph object containing the dialogue graph
        dialogues: List of Dialogue objects to check against

    Returns:
        bool: True if all graph elements are present in at least one dialogue
    """
    # Get all unique utterances from nodes and edges in the graph
    graph_utterances = set()

    # Add node utterances
    for node_id, node_data in G.graph.nodes(data=True):
        graph_utterances.update(node_data["utterances"])

    # Add edge utterances
    for _, _, edge_data in G.graph.edges(data=True):
        if isinstance(edge_data["utterances"], list):
            graph_utterances.update(edge_data["utterances"])
        else:
            graph_utterances.add(edge_data["utterances"])

    # Collect all utterances from dialogues
    dialogue_utterances = set()
    for dialogue in dialogues:
        dialogue_utterances.update(utt.text for utt in dialogue.messages)

    # Check if all graph utterances are present in dialogues
    if graph_utterances.issubset(dialogue_utterances):
        return True
    else:
        # return False
        return graph_utterances.difference(dialogue_utterances)


def all_roles_correct(D1: Dialogue, D2: Dialogue) -> bool:
    for phrase_1, phrase_2 in zip(D1.messages, D2.messages):
        if phrase_1.participant != phrase_2.participant:
            return False
    return True


def is_correct_lenght(D1: Dialogue, D2: Dialogue) -> bool:
    return len(D1.messages) == len(D2.messages)


def are_answers_similar(D1: Dialogue, D2: Dialogue, model, threshold: float) -> bool:
    raise NotImplementedError

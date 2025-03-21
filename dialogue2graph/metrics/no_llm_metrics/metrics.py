"""
Automatic Metrics.
------------------

This module contains functions that automatically (without using LLMs) checks Graphs and Dialogues
for various metrics.
"""

from typing import List, TypedDict, Optional
import numpy as np
import networkx as nx

from dialogue2graph.pipelines.core.graph import BaseGraph
from dialogue2graph.pipelines.core.dialogue import Dialogue


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


def _get_jaccard_edges(true_graph_edges, generated_graph_edges, verbose=False, return_matrix=False):
    """
    true_graph_edges:Graph.edges - ребра истинного графа
    generated_graph_edges: nx.Graph.edges - ребра сгенерированного графу
    формат ребер:
    (1, 2, {"utterances": ...})
    verbose: bool - печать отладочной информации
    """
    true_graph_edges = _collapse_multiedges(list(true_graph_edges))
    generated_graph_edges = _collapse_multiedges(list(generated_graph_edges))

    jaccard_values = np.zeros((len(true_graph_edges), len(generated_graph_edges)))
    print(jaccard_values.shape)
    for idx1, (k1, v1) in enumerate(true_graph_edges.items()):
        for idx2, (k2, v2) in enumerate(generated_graph_edges.items()):
            value1 = set(v1).intersection(set(v2))
            value2 = set(v1).union(set(v2))
            jaccard_values[idx1][idx2] = len(value1) / len(value2)

            if verbose:
                print(k1, v1)
                print(k2, v2)
                print(value1, value2)
                print("___")
    if verbose:
        print(jaccard_values)
    max_jaccard_values = np.max(jaccard_values, axis=1)
    max_jaccard_indices = np.argmax(jaccard_values, axis=1)
    if return_matrix:
        return max_jaccard_values, max_jaccard_indices, jaccard_values
    return list(max_jaccard_values), list(max_jaccard_indices)


def _get_list_of_node_utterances(node1_utterances):
    if type(node1_utterances) is str:
        return [node1_utterances]
    return node1_utterances


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


def _get_jaccard_nodes(true_graph_nodes, generated_graph_nodes, verbose=False, return_matrix=False):
    """
    true_graph_nodes: Graph.nodes - вершины истинного графа
    generated_graph_nodes: nx.Graph.nodes - вершины сгенерированного графу
    формат вершин:
    (1, {"utterances": ...})
    verbose: bool - печать отладочной информации
    """
    true_graph_nodes = _collapse_multinodes(list(true_graph_nodes))
    generated_graph_nodes = _collapse_multinodes(list(generated_graph_nodes))

    jaccard_values = np.zeros((len(true_graph_nodes) + 1, len(generated_graph_nodes) + 1))
    print(true_graph_nodes)
    for node1_id, node1_utterances in true_graph_nodes.items():
        for node2_id, node2_utterances in generated_graph_nodes.items():
            node1_utterances = set(_get_list_of_node_utterances(node1_utterances))
            node2_utterances = set(_get_list_of_node_utterances(node2_utterances))

            jaccard_nominator = node1_utterances.intersection(node2_utterances)
            jaccard_denominator = node1_utterances.union(node2_utterances)

            jaccard_values[node1_id][node2_id] = len(jaccard_nominator) / len(jaccard_denominator)

            if verbose:
                print(node1_utterances)
                print(node2_utterances)
                print(jaccard_nominator, jaccard_denominator)
                print("_____")
    if verbose:
        print(jaccard_values)
    max_jaccard_values = np.max(jaccard_values[1:], axis=1)
    max_jaccard_indices = np.argmax(jaccard_values[1:], axis=1)
    max_jaccard_indices = max_jaccard_indices - np.ones(max_jaccard_indices.shape)
    np.place(max_jaccard_indices, max_jaccard_indices < 0, 0)
    max_jaccard_indices = max_jaccard_indices.astype(int)
    if return_matrix:
        return max_jaccard_values, max_jaccard_indices, jaccard_values[1:, 1:]
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
    src, trg = map(int, edge.split("->"))
    return src - 1, trg - 1


def match_graph_triplets(G1: BaseGraph, G2: BaseGraph, change_to_original_ids=False):
    g1 = G1.graph
    g2 = G2.graph
    node_mapping = {node: None for node in g1.nodes}
    node_mapping.update({node: None for node in g2.nodes})
    if type(g1) is nx.DiGraph():
        GM = nx.isomorphism.DiGraphMatcher(g1, g2, edge_match=lambda x, y: set(x["utterances"]).intersection(set(y["utterances"])) is not None)
        are_isomorphic = GM.is_isomorphic()
    else:
        GM = nx.isomorphism.MultiDiGraphMatcher(g1, g2, edge_match=_match_edge_for_multigraph)
        are_isomorphic = GM.is_isomorphic()
    if are_isomorphic:
        print("Graphs are isomorphic")
        node_mapping = nx.vf2pp_isomorphism(g1, g2, node_label=None)

    edge_mapping = {}
    mapping_jaccard_values = {}

    edges1 = list(_collapse_multiedges(g1.edges(data=True)).keys())
    edges2 = list(_collapse_multiedges(g2.edges(data=True)).keys())

    _, _, matrix_edges = _get_jaccard_edges(g1.edges(data=True), g2.edges(data=True), verbose=False, return_matrix=True)

    _, _, matrix_nodes = _get_jaccard_nodes(g1.nodes(data=True), g2.nodes(data=True), verbose=False, return_matrix=True)

    for i, edge1 in enumerate(edges1):
        edge_mapping[edge1] = None
        mapping_jaccard_values[edge1] = 0
        for j, edge2 in enumerate(edges2):
            if matrix_edges[i][j] > 0:
                node1_src, node1_trg = _parse_edge(edge1)
                node2_src, node2_trg = _parse_edge(edge2)
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


def _get_dialogue_triplets(seq: list[Dialogue]) -> set[tuple[str]]:
    """Find all dialogue triplets with (source, edge, target) utterances"""
    result = []
    for dialogue in seq:
        assist_texts = [d.text.lower() for d in dialogue.messages if d.participant == "assistant"]
        user_texts = [d.text.lower() for d in dialogue.messages if d.participant == "user"]
        result.extend([(a1, u, a2) for a1, u, a2 in zip(assist_texts[:-1], user_texts[: len(assist_texts) - 1], assist_texts[1:])])
    return set(result)


def _get_graph_triplets(G: BaseGraph):
    """Find all graph triplets with (source, edge, target) utterances"""
    graph = G.graph_dict
    edges = graph["edges"]
    nodes = graph["nodes"]
    result = []
    for node in nodes:
        for edge in [e for e in edges if e["source"] == node["id"]]:
            for utt in edge["utterances"]:
                for utt1 in node["utterances"]:
                    for utt2 in [n for n in nodes if n["id"] == edge["target"]][0]["utterances"]:
                        result.append((utt1.lower(), utt.lower(), utt2.lower()))
    return set(result)


class AbsentTriplet(TypedDict):
    """To return absent triplets in DGTripletsMatchResult"""

    source: str
    edge: str
    target: str

    @classmethod
    def from_tuple(cls, triplet: tuple[str]) -> "AbsentTriplet":
        """Create AbsentTriplet from a tuple"""
        return cls(source=triplet[0], edge=triplet[1], target=triplet[2])


class DGTripletsMatchResult(TypedDict):
    """To return result of matching triplets between graph and dialogues"""

    value: bool
    description: Optional[str]
    absent_triplets: Optional[List[AbsentTriplet]]


def match_triplets_dg(G: BaseGraph, dialogues: list[Dialogue]) -> DGTripletsMatchResult:
    """
    Check if all graph triplets match triplets in set of dialogues.

    Args:
        G: BaseGraph object containing the dialogue graph
        dialogues: List of Dialogue objects to check against

    """

    dialogue_set = _get_dialogue_triplets(dialogues)
    graph_set = _get_graph_triplets(G)
    graph_absent = dialogue_set - graph_set
    dialogue_absent = graph_set - dialogue_set
    if dialogue_set.issubset(graph_set):
        print("Graph has all the dialogues")
    if not len(graph_absent) and not len(dialogue_absent):
        return {"value": True}
    if len(dialogue_absent):
        return {
            "value": False,
            "description": "Triplets missing in dialogues",
            "absent_triplets": [AbsentTriplet.from_tuple(triplet) for triplet in dialogue_absent],
        }
    else:
        return {
            "value": False,
            "description": "Triplets missing in graph",
            "absent_triplets": [AbsentTriplet.from_tuple(triplet) for triplet in graph_absent],
        }


def _match_ua(G: BaseGraph, user: str, assistant: str) -> bool:
    """
    Check if the graph G has a connection from user message to assistant message.

    Args:
        G: BaseGraph object containing the dialogue graph
        user, assistant: pair of neighboring utterances in a dialogue

    Returns:
        True if there is connection, False otherwise
    """

    nodes = G.nodes_by_utterance(assistant)

    for node in nodes:
        edges = G.edges_by_utterance(user)
        for edge in edges:
            if edge["target"] == node["id"]:
                return True
    return False


def _match_au(G: BaseGraph, assistant: str, user: str) -> bool:
    """
    Check if the graph G has a connection from assistant message to user message.

    Args:
        G: BaseGraph object containing the dialogue graph
        assistant, user: pair of neighboring utterances in a dialogue

    Returns:
        True if there is connection, False otherwise
    """

    nodes = G.nodes_by_utterance(assistant)

    for node in nodes:
        edges = G.edges_by_utterance(user)
        for edge in edges:
            if edge["source"] == node["id"]:
                return True
    return False


def _match_pair(G: BaseGraph, msg1: dict, msg2: dict) -> bool:
    """
    Check if the graph G has a connection from msg1 to msg2.

    Args:
        G: BaseGraph object containing the dialogue graph
        msg1, msg2: pair of neighboring utterances in a dialogue

    Returns:
        True if there is connection, False otherwise
    """
    if msg1.participant == "assistant" and msg2.participant == "user":
        return _match_au(G, msg1.text, msg2.text)
    if msg1.participant == "user" and msg2.participant == "assistant":
        return _match_ua(G, msg1.text, msg2.text)
    return False


class InvalidDialogueTransition(TypedDict):
    """To return invalid dialogue transition in DialogueValidationResult"""

    from_message: str
    to_message: str
    dialogue_id: str


class DialogueValidationResult(TypedDict):
    """To return result of dialogues_are_valid_paths"""

    value: bool
    invalid_transitions: Optional[List[InvalidDialogueTransition]]


def are_paths_valid(G: BaseGraph, dialogues: list[Dialogue]) -> DialogueValidationResult:
    """
    Check if all dialogues are valid paths in the graph.

    Args:
        G: BaseGraph object containing the dialogue graph
        dialogues: List of Dialogue objects to check against

    Returns:
        list: for every dialogue {"value": bool, "description": "description with dialogue_id and list of pairs when there is no connection from one message to another"}
    """

    invalid_transitions = []
    for dialogue in dialogues:
        for idx in range(len(dialogue.messages) - 1):
            if not _match_pair(G, dialogue.messages[idx], dialogue.messages[idx + 1]):
                invalid_transitions.append(
                    {"from_message": dialogue.messages[idx].text, "to_message": dialogue.messages[idx + 1].text, "dialogue_id": dialogue.id}
                )
    if invalid_transitions:
        return {"value": False, "invalid_transitions": invalid_transitions}
    return {"value": True}


def match_roles(D1: Dialogue, D2: Dialogue) -> bool:
    for phrase_1, phrase_2 in zip(D1.messages, D2.messages):
        if phrase_1.participant != phrase_2.participant:
            return False
    return True


def is_correct_length(D1: Dialogue, D2: Dialogue) -> bool:
    return len(D1.messages) == len(D2.messages)


def are_answers_similar(D1: Dialogue, D2: Dialogue, model, threshold: float) -> bool:
    raise NotImplementedError

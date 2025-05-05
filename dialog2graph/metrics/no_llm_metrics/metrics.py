"""
Automatic Metrics.
------------------

The module contains functions that automatically (without using LLMs) check Graphs and Dialogs
for various metrics.
"""

import logging
from typing import List, TypedDict, Optional
import numpy as np
import networkx as nx

from dialog2graph.pipelines.core.graph import BaseGraph
from dialog2graph.pipelines.core.dialog import Dialog
from dialog2graph.utils.logger import Logger

logger = Logger(__name__)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _collapse_multiedges(edges):
    """
    Collapse multiedges with the same (u -> v) into a single entry,
    accumulating their utterances into a single list.
    """
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
    """
    Calculate Jaccard similarity between edges of the original graph and the generated graph.

    Parameters:
        true_graph_edges (Graph.edges): Edges of the original graph (e.g. G1.edges(data=True)).
        generated_graph_edges (nx.Graph.edges): Edges of the generated graph. Format of edges: (node1, node2, {"utterances": ...})
        verbose (bool): Whether to print debug information.
        return_matrix (bool): If True, returns the entire Jaccard matrix.

    Returns:
        If return_matrix is False: returns (max_jaccard_values, max_jaccard_indices)
        If return_matrix is True: returns (max_jaccard_values, max_jaccard_indices, full_jaccard_matrix)
    """
    true_graph_edges = _collapse_multiedges(list(true_graph_edges))
    generated_graph_edges = _collapse_multiedges(list(generated_graph_edges))

    jaccard_values = np.zeros((len(true_graph_edges), len(generated_graph_edges)))
    print(jaccard_values.shape)
    for idx1, (k1, v1) in enumerate(true_graph_edges.items()):
        for idx2, (k2, v2) in enumerate(generated_graph_edges.items()):
            intersect_set = set(v1).intersection(set(v2))
            union_set = set(v1).union(set(v2))
            jaccard_values[idx1][idx2] = (
                len(intersect_set) / len(union_set) if len(union_set) > 0 else 0.0
            )

            if verbose:
                print(k1, v1)
                print(k2, v2)
                print(intersect_set, union_set)
                print("___")
    if verbose:
        print(jaccard_values)
    max_jaccard_values = np.max(jaccard_values, axis=1)
    max_jaccard_indices = np.argmax(jaccard_values, axis=1)
    if return_matrix:
        return max_jaccard_values, max_jaccard_indices, jaccard_values
    return list(max_jaccard_values), list(max_jaccard_indices)


def _get_list_of_node_utterances(node_utterances):
    """
    Ensure that node_utterances is always returned as a list of strings.
    """
    if isinstance(node_utterances, str):
        return [node_utterances]
    return node_utterances


def _collapse_multinodes(nodes):
    """
    Collapse node utterances if needed, returning a dict
    where key=node_id, value=list_of_utterances.
    """
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
    """
    Calculate Jaccard similarity between nodes of the original graph and the generated graph.

    Parameters:
        true_graph_nodes (Graph.nodes): Nodes of the original graph (e.g. G1.nodes(data=True)).
        generated_graph_nodes (nx.Graph.nodes): Nodes of the generated graph. Node format: (node_id, {"utterances": ...})
        verbose (bool): Whether to print debug information.
        return_matrix (bool): If True, returns the entire Jaccard matrix.

    Returns:
        If return_matrix is False: returns (max_jaccard_values, max_jaccard_indices)
        If return_matrix is True: returns (max_jaccard_values, max_jaccard_indices, full_jaccard_matrix)
    """
    true_graph_nodes = _collapse_multinodes(list(true_graph_nodes))
    generated_graph_nodes = _collapse_multinodes(list(generated_graph_nodes))

    jaccard_values = np.zeros(
        (len(true_graph_nodes) + 1, len(generated_graph_nodes) + 1)
    )
    print(true_graph_nodes)
    for node1_id, node1_utterances in true_graph_nodes.items():
        for node2_id, node2_utterances in generated_graph_nodes.items():
            node1_utterances = set(_get_list_of_node_utterances(node1_utterances))
            node2_utterances = set(_get_list_of_node_utterances(node2_utterances))

            jaccard_nominator = node1_utterances.intersection(node2_utterances)
            jaccard_denominator = node1_utterances.union(node2_utterances)

            jaccard_values[node1_id][node2_id] = (
                len(jaccard_nominator) / len(jaccard_denominator)
                if len(jaccard_denominator) > 0
                else 0.0
            )

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
    """
    Match edges for MultiDiGraph, checking if there is any intersection
    in the 'utterances' sets.
    """
    if isinstance(x, dict) and isinstance(y, dict):
        set1 = set([elem["utterances"] for elem in list(x.values())])
        set2 = set([elem["utterances"] for elem in list(y.values())])
    else:
        set1 = set(x)
        set2 = set(y)
    return set1.intersection(set2) is not None


def _parse_edge(edge):
    """
    Parse an edge string formatted as 'src->trg' into (src_index, trg_index),
    adjusting them to be zero-based.
    """
    src, trg = map(int, edge.split("->"))
    return src - 1, trg - 1


def match_graph_triplets(G1: BaseGraph, G2: BaseGraph, change_to_original_ids=False):
    """
    Match two graphs (G1 and G2) by:
      1) Checking isomorphism using NetworkX matchers (depending on whether it's DiGraph or MultiDiGraph).
      2) Building Jaccard similarity matrices for nodes and edges to refine the mapping.
      3) Potentially reverting mapped IDs to G1's original numbering if change_to_original_ids=True.

    Returns:
        node_mapping, edge_mapping (dict, dict):
            - node_mapping: which node in G1 corresponds to which node in G2
            - edge_mapping: which edge in G1 corresponds to which edge in G2
    """
    g1 = G1.graph
    g2 = G2.graph
    node_mapping = {node: None for node in g1.nodes}
    node_mapping.update({node: None for node in g2.nodes})
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
        print("Graphs are isomorphic")
        node_mapping = nx.vf2pp_isomorphism(g1, g2, node_label=None)

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
                    node1_src_nx = g1.nodes[node1_src + 1]
                    node2_src_nx = g2.nodes[node2_src + 1]
                    if node1_src_nx == node2_src_nx:
                        node_mapping[node1_src + 1] = node2_src + 1

                    node1_trg_nx = g1.nodes[node1_trg + 1]
                    node2_trg_nx = g2.nodes[node2_trg + 1]
                    if node1_trg_nx == node2_trg_nx:
                        node_mapping[node1_trg + 1] = node2_trg + 1
                    print(
                        f"""The nodes of edges {edges1[i]} and {
                            edges2[j]
                        } has something in common, but not complete match: Sources: {
                            node1_src_nx["utterances"]
                        }, {node2_src_nx["utterances"]}"""
                    )
                    print(
                        f"""The nodes of edges {edges1[i]} and {
                            edges2[j]
                        } has something in common, but not complete match: Targets: {
                            node1_trg_nx["utterances"]
                        }, {node2_trg_nx["utterances"]}"""
                    )

    if G1.node_mapping != {} and change_to_original_ids:
        new_node_mapping = {}
        new_edge_mapping = {}
        inverse_mapping = {v: k for k, v in G1.node_mapping.items()}

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
                f"{inverse_mapping[int(src1)]}->{inverse_mapping[int(trg1)]}"
            ] = edge2
        return new_node_mapping, new_edge_mapping

    return node_mapping, edge_mapping


def is_same_structure(G1: BaseGraph, G2: BaseGraph, sim_model=None) -> bool:
    """
    Check if graphs are isomorphic.

    Args:
        G1: BaseGraph object containing the dialog graph
        G2: BaseGraph object containing the dialog graph
    """
    g1 = G1.graph
    g2 = G2.graph
    return nx.is_isomorphic(g1, g2)


def _get_dialog_triplets(seq: list[Dialog]) -> set[tuple[str]]:
    """Find all dialog triplets with (source, edge, target) utterances"""
    result = []
    for dialog in seq:
        assist_texts = [
            d.text.lower() for d in dialog.messages if d.participant == "assistant"
        ]
        user_texts = [
            d.text.lower() for d in dialog.messages if d.participant == "user"
        ]
        result.extend(
            [
                (a1, u, a2)
                for a1, u, a2 in zip(
                    assist_texts[:-1],
                    user_texts[: len(assist_texts) - 1],
                    assist_texts[1:],
                )
            ]
        )
    return set(result)


def match_dialog_triplets(s1: list[Dialog], s2: list[Dialog]):
    """Match triplets of two dialog sequences"""
    return {"value": _get_dialog_triplets(s1) == _get_dialog_triplets(s2)}


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
                    for utt2 in [n for n in nodes if n["id"] == edge["target"]][0][
                        "utterances"
                    ]:
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
    """To return result of matching triplets between graph and dialogs"""

    value: bool
    description: Optional[str]
    absent_triplets: Optional[List[AbsentTriplet]]


def match_dg_triplets(G: BaseGraph, dialogs: list[Dialog]) -> DGTripletsMatchResult:
    """
    Check if all graph triplets match triplets in set of dialogs.

    Args:
        G: BaseGraph object containing the dialog graph
        dialogs: List of Dialog objects to check against
    """
    dialog_set = _get_dialog_triplets(dialogs)
    graph_set = _get_graph_triplets(G)
    graph_absent = dialog_set - graph_set
    dialog_absent = graph_set - dialog_set

    if dialog_set.issubset(graph_set):
        logger.info("Graph has all the dialogs")
    if not len(graph_absent) and not len(dialog_absent):
        return {"value": True}
    if len(dialog_absent):
        return {
            "value": False,
            "description": "Triplets missing in dialogs",
            "absent_triplets": [
                AbsentTriplet.from_tuple(triplet) for triplet in dialog_absent
            ],
        }
    else:
        return {
            "value": False,
            "description": "Triplets missing in graph",
            "absent_triplets": [
                AbsentTriplet.from_tuple(triplet) for triplet in graph_absent
            ],
        }


def _match_ua(G: BaseGraph, user: str, assistant: str) -> bool:
    """
    Check if the graph G has a connection from user message to assistant message.

    Args:
        G: BaseGraph object containing the dialog graph
        user, assistant: pair of neighboring utterances in a dialog

    Returns:
        True if there is connection, False otherwise
    """
    nodes = G.find_nodes_by_utterance(assistant)
    for node in nodes:
        edges = G.find_edges_by_utterance(user)
        for edge in edges:
            if edge["target"] == node["id"]:
                return True
    return False


def _match_au(G: BaseGraph, assistant: str, user: str) -> bool:
    """
    Check if the graph G has a connection from assistant message to user message.

    Args:
        G: BaseGraph object containing the dialog graph
        assistant, user: pair of neighboring utterances in a dialog

    Returns:
        True if there is connection, False otherwise
    """
    nodes = G.find_nodes_by_utterance(assistant)
    for node in nodes:
        edges = G.find_edges_by_utterance(user)
        for edge in edges:
            if edge["source"] == node["id"]:
                return True
    return False


def _match_pair(G: BaseGraph, msg1: dict, msg2: dict) -> bool:
    """
    Check if the graph G has a connection from msg1 to msg2.

    Args:
        G: BaseGraph object containing the dialog graph
        msg1, msg2: pair of neighboring utterances in a dialog

    Returns:
        True if there is connection, False otherwise
    """
    if msg1.participant == "assistant" and msg2.participant == "user":
        return _match_au(G, msg1.text, msg2.text)
    if msg1.participant == "user" and msg2.participant == "assistant":
        return _match_ua(G, msg1.text, msg2.text)
    return False


class InvalidDialogTransition(TypedDict):
    """To return invalid dialog transition in DialogValidationResult"""

    from_message: str
    to_message: str
    dialog_id: str


class DialogValidationResult(TypedDict):
    """To return result of dialogs_are_valid_paths"""

    value: bool
    invalid_transitions: Optional[List[InvalidDialogTransition]]


def are_paths_valid(G: BaseGraph, dialogs: list[Dialog]) -> DialogValidationResult:
    """
    Check if all dialogs are valid paths in the graph.

    Args:
        G: BaseGraph object containing the dialog graph
        dialogs: List of Dialog objects to check against

    Returns:
        list: for every dialog {"value": bool, "description": "description with dialog_id and list of pairs when there is no connection from one message to another"}
    """
    invalid_transitions = []
    for dialog in dialogs:
        for idx in range(len(dialog.messages) - 1):
            if not _match_pair(G, dialog.messages[idx], dialog.messages[idx + 1]):
                invalid_transitions.append(
                    {
                        "from_message": dialog.messages[idx].text,
                        "to_message": dialog.messages[idx + 1].text,
                        "dialog_id": dialog.id,
                    }
                )
    if invalid_transitions:
        return {"value": False, "invalid_transitions": invalid_transitions}
    return {"value": True}


def match_roles(D1: Dialog, D2: Dialog) -> bool:
    """
    Check if two dialogs have identical participant roles in each turn.

    Returns:
        True if they match in every turn, otherwise False.
    """
    for phrase_1, phrase_2 in zip(D1.messages, D2.messages):
        if phrase_1.participant != phrase_2.participant:
            return False
    return True


def is_correct_length(D1: Dialog, D2: Dialog) -> bool:
    """
    Check if two dialogs have the same number of messages.

    Returns:
        True if lengths are equal, False otherwise.
    """
    return len(D1.messages) == len(D2.messages)


def are_answers_similar(D1: Dialog, D2: Dialog, model, threshold: float) -> bool:
    """
    Placeholder for any advanced similarity check between the dialogs' answers.
    Not implemented.
    """
    raise NotImplementedError


def all_utterances_present(G: BaseGraph, dialogs: List[Dialog]):
    """
    Check whether every utterance in the graph (both from nodes and edges)
    appears at least once in the provided dialogs.

    Returns:
        True if all graph utterances are found within the dialogs, otherwise returns the set of missing utterances.
    """
    graph_utterances = set()

    # Collect node utterances
    for _, node_data in G.graph.nodes(data=True):
        if isinstance(node_data["utterances"], list):
            graph_utterances.update(node_data["utterances"])
        else:
            graph_utterances.add(node_data["utterances"])

    # Collect edge utterances
    for _, _, edge_data in G.graph.edges(data=True):
        if isinstance(edge_data["utterances"], list):
            graph_utterances.update(edge_data["utterances"])
        else:
            graph_utterances.add(edge_data["utterances"])

    # Collect utterances from dialogs
    dialog_utterances = set()
    for dialog in dialogs:
        dialog_utterances.update(utt.text for utt in dialog.messages)

    if graph_utterances.issubset(dialog_utterances):
        return True
    else:
        return graph_utterances.difference(dialog_utterances)


def triplet_match_accuracy(
    G1: BaseGraph, G2: BaseGraph, change_to_original_ids: bool = False
) -> dict:
    """
    Calculate a simple accuracy metric for node and edge matching based on 'match_graph_triplets'.

    Returns:
        {
            "node_accuracy": fraction_of_matched_nodes_in_G1,
            "edge_accuracy": fraction_of_matched_edges_in_G1
        }
    """
    node_mapping, edge_mapping = match_graph_triplets(
        G1, G2, change_to_original_ids=change_to_original_ids
    )

    # Count matching nodes
    g1_nodes = set(G1.graph.nodes())
    matched_nodes = sum(1 for n in g1_nodes if node_mapping.get(n) is not None)
    total_nodes = len(g1_nodes)
    node_accuracy = matched_nodes / total_nodes if total_nodes > 0 else 0.0

    # Count matching edges
    g1_edges = list(G1.graph.edges())
    matched_edges = sum(
        1 for edge_str in edge_mapping if edge_mapping[edge_str] is not None
    )
    total_edges = len(g1_edges)
    edge_accuracy = matched_edges / total_edges if total_edges > 0 else 0.0

    return {"node_accuracy": node_accuracy, "edge_accuracy": edge_accuracy}


def compute_graph_metrics(graph_list: List[BaseGraph]) -> dict:
    """
    Compute various statistics across a list of Graph objects,
    where each Graph has a 'graph_dict' containing 'edges' and 'nodes'.

    Expects each element in 'graph_list' to be something like:
        Graph(
            graph_dict={
                "edges": [...],
                "nodes": [...]
            },
            graph=<networkx graph object>,
            node_mapping={}
        )

    Returns:
        dict: a dictionary with the following keys:
        - "with_cycles" (int): How many of the graphs contain at least one cycle.
        - "percentage_with_cycles" (float): Percentage of graphs that have a cycle, out of all.
        - "average_edges_amount" (float): Average number of edges per graph.
        - "average_nodes_amount" (float): Average number of nodes per graph.
        - "total_graphs" (int): Total number of graphs processed.
        - "total_edges" (int): Sum of edges across all graphs.
        - "total_nodes" (int): Sum of nodes across all graphs.
    """
    total_graphs = len(graph_list)
    with_cycles = 0
    total_edges = 0
    total_nodes = 0

    for graph_object in graph_list:
        edges = graph_object.graph_dict["edges"]
        nodes = graph_object.graph_dict["nodes"]

        g = nx.DiGraph()
        for node_info in nodes:
            g.add_node(node_info["id"])
        for edge_info in edges:
            g.add_edge(edge_info["source"], edge_info["target"])

        edges_count = g.number_of_edges()
        nodes_count = g.number_of_nodes()
        total_edges += edges_count
        total_nodes += nodes_count

        if not nx.is_directed_acyclic_graph(g):
            with_cycles += 1

    average_edges_amount = total_edges / total_graphs if total_graphs > 0 else 0.0
    average_nodes_amount = total_nodes / total_graphs if total_graphs > 0 else 0.0
    percentage_with_cycles = (
        (with_cycles / total_graphs * 100) if total_graphs > 0 else 0.0
    )

    return {
        "with_cycles": with_cycles,
        "percentage_with_cycles": percentage_with_cycles,
        "average_edges_amount": average_edges_amount,
        "average_nodes_amount": average_nodes_amount,
        "total_graphs": total_graphs,
        "total_edges": total_edges,
        "total_nodes": total_nodes,
    }

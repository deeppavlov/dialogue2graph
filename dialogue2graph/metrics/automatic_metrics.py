"""
Automatic Metrics.
------------------

This module contains functions that automatically (without using LLMs) checks Graphs and Dialogues
for various metrics.
"""

import networkx as nx
import numpy as np

from dialogue2graph.pipelines.core.graph import BaseGraph
from dialogue2graph.pipelines.core.dialogue import Dialogue


def collapse_multiedges(edges):
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


def jaccard_edges(true_graph_edges, generated_graph_edges, verbose=False, return_matrix=False):
    """
    true_graph_edges: Graph.edges - ребра истинного графа
    generated_graph_edges: nx.Graph.edges - ребра сгенерированного графа
    формат ребер:
    (1, 2, {"utterances": ...})
    verbose: bool - печать отладочной информации
    """
    true_graph_edges = collapse_multiedges(list(true_graph_edges))
    generated_graph_edges = collapse_multiedges(list(generated_graph_edges))

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


def get_list_of_node_utterances(node1_utterances):
    if isinstance(node1_utterances, str):
        return [node1_utterances]
    return node1_utterances


def collapse_multinodes(nodes):
    collapsed_nodes = {}
    for key, data in nodes:
        if key not in collapsed_nodes:
            collapsed_nodes[key] = []
        if isinstance(data["utterances"], str):
            collapsed_nodes[key].append(data["utterances"])
        elif isinstance(data["utterances"], list):
            collapsed_nodes[key].extend(data["utterances"])
    return collapsed_nodes


def jaccard_nodes(true_graph_nodes, generated_graph_nodes, verbose=False, return_matrix=False):
    """
    true_graph_nodes: Graph.nodes - вершины истинного графа
    generated_graph_nodes: nx.Graph.nodes - вершины сгенерированного графа
    формат вершин:
    (1, {"utterances": ...})
    verbose: bool - печать отладочной информации
    """
    true_graph_nodes = collapse_multinodes(list(true_graph_nodes))
    generated_graph_nodes = collapse_multinodes(list(generated_graph_nodes))

    jaccard_values = np.zeros((len(true_graph_nodes) + 1, len(generated_graph_nodes) + 1))
    print(true_graph_nodes)
    for node1_id, node1_utterances in true_graph_nodes.items():
        for node2_id, node2_utterances in generated_graph_nodes.items():
            node1_utterances = set(get_list_of_node_utterances(node1_utterances))
            node2_utterances = set(get_list_of_node_utterances(node2_utterances))

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


def edge_match_for_multigraph(x, y):
    if isinstance(x, dict) and isinstance(y, dict):
        set1 = {elem["utterances"] for elem in x.values()}
        set2 = {elem["utterances"] for elem in y.values()}
    else:
        set1 = set(x)
        set2 = set(y)
    return set1.intersection(set2) is not None


def parse_edge(edge):
    src, trg = map(int, edge.split("->"))
    return src - 1, trg - 1


def triplet_match(G1, G2, change_to_original_ids=False):
    """
    Attempts to match two graphs (G1 and G2) by:
      1) Checking isomorphism using NetworkX matchers (depending on graph type).
      2) Building Jaccard similarity matrices for nodes and edges to refine the mapping.
      3) Potentially reverting mapped IDs to G1's original numbering if change_to_original_ids=True.

    Returns:
        node_mapping, edge_mapping (dict, dict):
            - node_mapping: which node in G1 corresponds to which node in G2
            - edge_mapping: which edge in G1 corresponds to which edge in G2
    """
    g1 = G1.graph
    g2 = G2.graph

    # Инициализируем mapping для всех узлов
    node_mapping = {node: None for node in g1.nodes}
    node_mapping.update({node: None for node in g2.nodes})

    # --- ИСПРАВЛЕННАЯ ПРОВЕРКА ТИПА ГРАФА ---
    if isinstance(g1, nx.DiGraph) and isinstance(g2, nx.DiGraph):
        GM = nx.isomorphism.DiGraphMatcher(
            g1,
            g2,
            edge_match=lambda x, y: (
                set(x["utterances"]).intersection(set(y["utterances"])) is not None
            )
        )
        are_isomorphic = GM.is_isomorphic()
    elif isinstance(g1, nx.MultiDiGraph) and isinstance(g2, nx.MultiDiGraph):
        GM = nx.isomorphism.MultiDiGraphMatcher(
            g1,
            g2,
            edge_match=edge_match_for_multigraph
        )
        are_isomorphic = GM.is_isomorphic()
    else:
        # Если один граф MultiDiGraph, а другой DiGraph,
        # или вообще неизвестный тип, можно либо бросать ошибку, либо делать fallback:
        GM = nx.isomorphism.DiGraphMatcher(
            g1,
            g2,
            edge_match=lambda x, y: (
                set(x["utterances"]).intersection(set(y["utterances"])) is not None
            )
        )
        are_isomorphic = GM.is_isomorphic()

    if are_isomorphic:
        print("Graphs are isomorphic")
        # получение точного mapping через метод nx.vf2pp_isomorphism, если это нужно:
        node_mapping = nx.vf2pp_isomorphism(g1, g2, node_label=None)

    edge_mapping = {}
    mapping_jaccard_values = {}

    # Получаем списки "u->v" для рёбер
    edges1 = list(collapse_multiedges(g1.edges(data=True)).keys())
    edges2 = list(collapse_multiedges(g2.edges(data=True)).keys())

    # Полные Jaccard-матрицы
    _, _, matrix_edges = jaccard_edges(
        g1.edges(data=True), g2.edges(data=True),
        verbose=False, return_matrix=True
    )
    _, _, matrix_nodes = jaccard_nodes(
        g1.nodes(data=True), g2.nodes(data=True),
        verbose=False, return_matrix=True
    )

    # Основной цикл по ребрам
    for i, edge1 in enumerate(edges1):
        edge_mapping[edge1] = None
        mapping_jaccard_values[edge1] = 0
        for j, edge2 in enumerate(edges2):
            if matrix_edges[i][j] > 0:
                node1_src, node1_trg = parse_edge(edge1)
                node2_src, node2_trg = parse_edge(edge2)

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

                        # +1, т.к. parse_edge возвращает (node_id - 1)
                        node_mapping[node1_src + 1] = node2_src + 1
                        node_mapping[node1_trg + 1] = node2_trg + 1
                else:
                    # Частичное совпадение узлов
                    node1_src_nx = g1.nodes[node1_src + 1]
                    node2_src_nx = g2.nodes[node2_src + 1]
                    if node1_src_nx == node2_src_nx:
                        node_mapping[node1_src + 1] = node2_src + 1

                    node1_trg_nx = g1.nodes[node1_trg + 1]
                    node2_trg_nx = g2.nodes[node2_trg + 1]
                    if node1_trg_nx == node2_trg_nx:
                        node_mapping[node1_trg + 1] = node2_trg + 1

                    print(
                        f"""The nodes of edges {edges1[i]} and {edges2[j]} have something in common,
                        but not a complete match:
                        Sources: {node1_src_nx["utterances"]}, {node2_src_nx["utterances"]}"""
                    )
                    print(
                        f"""The nodes of edges {edges1[i]} and {edges2[j]} have something in common,
                        but not a complete match:
                        Targets: {node1_trg_nx["utterances"]}, {node2_trg_nx["utterances"]}"""
                    )

    # Если у G1 есть свой собственный node_mapping и нужно восстановить ID
    if G1.node_mapping != {} and change_to_original_ids:
        new_node_mapping = {}
        new_edge_mapping = {}

        # Переворачиваем mapping из G1, чтобы найти «оригинальные» ID
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


def is_same_structure(G1: BaseGraph, G2: BaseGraph) -> bool:
    """
    Checks if two graphs have the same structure (i.e., are isomorphic).

    Parameters:
        G1 (BaseGraph): The first graph object (contains a NetworkX graph).
        G2 (BaseGraph): The second graph object (contains a NetworkX graph).

    Returns:
        bool: True if the two underlying NetworkX graphs are isomorphic, otherwise False.

    Calculation:
        - Uses networkx.is_isomorphic to verify structural equivalence.
        - Graphs are considered isomorphic if there's a one-to-one mapping between their nodes/edges.
    """
    g1 = G1.graph
    g2 = G2.graph
    return nx.is_isomorphic(g1, g2)


def all_utterances_present(G: BaseGraph, dialogues: list[Dialogue]) -> bool:
    """
    Checks whether every utterance in the graph (both from nodes and edges) appears
    at least once in the given dialogues.

    Parameters:
        G (BaseGraph): A graph containing node and edge data (utterances).
        dialogues (list of Dialogue): A list of Dialogue objects in which to look for those utterances.

    Returns:
        bool: True if all graph utterances are found within the dialogues, otherwise returns the missing utterances.

    Calculation:
        1. Collect all node utterances and edge utterances from the graph into a set.
        2. Collect all utterances from every message in all provided dialogues into another set.
        3. If the graph's utterances are a subset of the dialogues' utterances, return True.
           Otherwise, return the difference as an indicator of missing items.
    """
    graph_utterances = set()

    # Add node utterances
    for _, node_data in G.graph.nodes(data=True):
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

    if graph_utterances.issubset(dialogue_utterances):
        return True
    else:
        return graph_utterances.difference(dialogue_utterances)


def all_roles_correct(D1: Dialogue, D2: Dialogue) -> bool:
    """
    Checks if two dialogues have identical participant roles in each corresponding turn.

    Parameters:
        D1 (Dialogue): The first dialogue object.
        D2 (Dialogue): The second dialogue object.

    Returns:
        bool: True if both dialogues have the same participant in each turn, False otherwise.

    Calculation:
        - Iterate simultaneously over messages in D1 and D2.
        - Compare the 'participant' attribute for each pair of messages.
        - Return False if any mismatch is found; True if they all match.
    """
    for phrase_1, phrase_2 in zip(D1.messages, D2.messages):
        if phrase_1.participant != phrase_2.participant:
            return False
    return True


def is_correct_lenght(D1: Dialogue, D2: Dialogue) -> bool:
    """
    Checks if two dialogues have the same number of messages.

    Parameters:
        D1 (Dialogue): The first dialogue object.
        D2 (Dialogue): The second dialogue object.

    Returns:
        bool: True if the length of both dialogues (number of messages) is equal, False otherwise.

    Calculation:
        - Compare len(D1.messages) with len(D2.messages).
        - Return True if they are equal, False if not.
    """
    return len(D1.messages) == len(D2.messages)


def are_answers_similar(D1: Dialogue, D2: Dialogue, model, threshold: float) -> bool:
    raise NotImplementedError


def triplet_match_accuracy(G1, G2, change_to_original_ids=False):
    """
    Calculates a simple accuracy metric for node and edge matching based on 'triplet_match'.

    Args:
        G1 (BaseGraph): The first graph object containing a NetworkX graph (G1.graph).
        G2 (BaseGraph): The second graph object containing a NetworkX graph (G2.graph).
        change_to_original_ids (bool): If True, the node/edge mappings returned
                                       from 'triplet_match' are converted back
                                       to the original IDs of G1 (when applicable).

    Returns:
        dict:
            A dictionary containing:
                - "node_accuracy": float, fraction of G1's nodes that found a match in G2
                - "edge_accuracy": float, fraction of G1's edges that found a match in G2

    How it calculates:
        1. Calls 'triplet_match(G1, G2, change_to_original_ids)' to retrieve:
            - node_mapping (dict): which node in G1 corresponds to which node in G2
            - edge_mapping (dict): which edge in G1 corresponds to which edge in G2
        2. For nodes:
            - Counts how many nodes in G1 have a non-None partner in G2
            - Divides by the total number of nodes in G1
        3. For edges:
            - Counts how many edges in G1 have a non-None mapping in G2
            - Divides by the total number of edges in G1
        4. Returns both metrics as a dictionary.

    Note:
        - This metric does not check whether the mapped node or edge also belongs
          to an identical label or utterances. It solely checks if 'triplet_match' deemed
          it to have a partner (non-None). The exact matching logic is governed by
          'triplet_match'.
        - If G1 has 0 nodes or edges, the respective accuracy is set to 0 by default.
    """
    node_mapping, edge_mapping = triplet_match(G1, G2, change_to_original_ids=change_to_original_ids)

    g1_nodes = set(G1.graph.nodes())
    matched_nodes = sum(1 for n in g1_nodes if node_mapping.get(n) is not None)
    total_nodes = len(g1_nodes)
    node_accuracy = matched_nodes / total_nodes if total_nodes > 0 else 0.0

    g1_edges = list(G1.graph.edges())
    matched_edges = sum(1 for edge_str in edge_mapping if edge_mapping[edge_str] is not None)
    total_edges = len(g1_edges)
    edge_accuracy = matched_edges / total_edges if total_edges > 0 else 0.0

    return {
        "node_accuracy": node_accuracy,
        "edge_accuracy": edge_accuracy
    }


def compute_graph_metrics(graph_list):
    """
    Computes various statistics across a list of Graph objects,
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

    Returns a dictionary with the following keys:
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
    percentage_with_cycles = (with_cycles / total_graphs * 100) if total_graphs > 0 else 0.0

    return {
        "with_cycles": with_cycles,
        "percentage_with_cycles": percentage_with_cycles,
        "average_edges_amount": average_edges_amount,
        "average_nodes_amount": average_nodes_amount,
        "total_graphs": total_graphs,
        "total_edges": total_edges,
        "total_nodes": total_nodes,
    }


def all_paths_sampled(G, dialogue):
    """
    Checks if all possible paths from start nodes to end nodes in the given graph
    appear (as a subsequence of utterances) in the provided dialogue.

    Args:
        G (BaseGraph): A graph structure containing a NetworkX DiGraph (G.graph).
                       Each edge has a "source", "target", and an "utterances" list.
                       Each node can have "is_start" to indicate a start node.
        dialogue (Dialogue): A single dialogue that contains a list of DialogueMessage objects.
                             Each DialogueMessage has a "text" and "participant".

    Returns:
        bool: True if every path in G can be found as a subsequence of utterances in 'dialogue';
              otherwise False.

    How it works:
        1) Identify all start nodes in the graph (those with is_start=True),
           or if none are explicitly marked, treat all nodes with no incoming edges as start nodes.
        2) Identify end nodes in the graph as those with no outgoing edges.
        3) Generate all paths from each start node to any end node. For each path,
           we get a sequence of edges in order.
        4) For each path, check if its edge utterances appear as a subsequence in the dialogue:
            - We iterate over the dialogue messages in order.
            - For each edge in the path, we see if there's a message in the dialogue containing
              at least one of the utterances from that edge.
            - We must find them in the same order as the path. If we manage to match all edges,
              we say that path is "sampled" by the dialogue.
        5) If all such paths are found in the dialogue, return True; otherwise, False.
    """

    def get_start_nodes(graph):
        start_nodes = [n for n, data in graph.nodes(data=True) if data.get("is_start", False)]
        if not start_nodes:
            # If no node is explicitly marked as start,
            # consider nodes that have no incoming edges.
            start_nodes = [n for n in graph.nodes if graph.in_degree(n) == 0]
        return start_nodes

    def get_end_nodes(graph):
        return [n for n in graph.nodes if graph.out_degree(n) == 0]

    def get_all_paths(graph, start_nodes, end_nodes):
        """
        Returns a list of paths, where each path is a list of node IDs
        from one start node to one end node.
        """
        all_paths = []
        for s in start_nodes:
            for e in end_nodes:
                paths = nx.all_simple_paths(graph, source=s, target=e)
                all_paths.extend(list(paths))
        return all_paths

    def is_path_sampled_in_dialogue(path_nodes, graph, messages):
        """
        Returns True if the edges along 'path_nodes' appear as a subsequence
        in the messages of the dialogue. Otherwise False.

        'messages' is a list of DialogueMessage objects with .text and .participant
        """
        edge_sequence = []
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i + 1]
            if graph.has_edge(u, v):
                data_for_edges = graph.get_edge_data(u, v)
                all_utterances = set()

                if isinstance(data_for_edges, dict):
                    # If it's a multi-edge or single-edge, unify utterances
                    for key, edge_info in data_for_edges.items():
                        if isinstance(edge_info, dict) and "utterances" in edge_info:
                            if isinstance(edge_info["utterances"], list):
                                all_utterances.update(edge_info["utterances"])
                            else:
                                all_utterances.add(edge_info["utterances"])

                    # Fallback single-edge
                    if "utterances" in data_for_edges:
                        ut = data_for_edges["utterances"]
                        if isinstance(ut, list):
                            all_utterances.update(ut)
                        else:
                            all_utterances.add(ut)
                        # break (можно убрать или оставить, зависит от структуры)

                edge_sequence.append(all_utterances)
            else:
                return False

        msg_index = 0
        for edge_utterances in edge_sequence:
            found = False
            while msg_index < len(messages):
                text_in_msg = messages[msg_index].text
                if text_in_msg in edge_utterances:
                    found = True
                    msg_index += 1
                    break
                msg_index += 1

            if not found:
                return False
        return True

    graph_nx = G.graph
    messages = dialogue.messages

    start_nodes = get_start_nodes(graph_nx)
    end_nodes = get_end_nodes(graph_nx)
    all_paths = get_all_paths(graph_nx, start_nodes, end_nodes)

    if not all_paths:
        return True

    for path in all_paths:
        if not is_path_sampled_in_dialogue(path, graph_nx, messages):
            return False

    return True


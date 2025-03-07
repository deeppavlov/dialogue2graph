"""
Automatic Metrics.
------------------

This module contains functions that automatically (without using LLMs) checks Graphs and Dialogues
for various metrics.
"""
import numpy as np
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
import networkx as nx
from dialogue2graph.pipelines.core.graph import BaseGraph
from dialogue2graph.pipelines.core.dialogue import Dialogue
from dialogue2graph.metrics.embedder import get_embedding

import numpy as np


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
    true_graph_edges:Graph.edges - ребра истинного графа
    generated_graph_edges: nx.Graph.edges - ребра сгенерированного графу
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
    if type(node1_utterances) is str:
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
    generated_graph_nodes: nx.Graph.nodes - вершины сгенерированного графу
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

from chatsky_llm_autoconfig.schemas import CompareResponse
from chatsky_llm_autoconfig.utils import call_llm_api, graph2list, nodes2list
from chatsky_llm_autoconfig.settings import EnvSettings
from chatsky_llm_autoconfig.compare_prompt import (
    compare_graphs_prompt, graph_example_1, result_form
)

from langchain.chat_models import ChatOpenAI
env_settings = EnvSettings()

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

def edges_match_nodes(graph: dict) -> bool:

    node_ids = set([n['id'] for n in graph['nodes']])
    edge_ids = set([e['target'] for e in graph['edges']])
    return node_ids == edge_ids

def dialogue_edges(seq: list[Dialogue]) -> set[tuple[str]]:
    """ Find all dialogue edges in a form of triplets with (source, edge, target) utterances"""
    res = []
    for dialogue in seq:
         assist_texts = [d.text.lower() for d in dialogue.messages if d.participant=='assistant']
         user_texts = [d.text.lower() for d in dialogue.messages if d.participant=='user'] 
         res.extend([(a1,u,a2) for a1,u,a2 in zip(assist_texts[:-1],user_texts[:len(assist_texts)-1],assist_texts[1:])])
    # print("DIA: ", set(res))
    return set(res)


def graph_edges(G: BaseGraph):
    """ Find all graph edges in a form of triplets with (source, edge, target) utterances"""
    graph = G.graph_dict
    edges = graph['edges']
    nodes = graph['nodes']
    res = []
    for node in nodes:
        for edge in [e for e in edges if e['source'] == node['id']]:
            for utt in edge['utterances']:
                for utt1 in node['utterances']:
                    for utt2 in [n for n in nodes if n['id']==edge['target']][0]['utterances']:
                        res.append((utt1.lower(),utt.lower(),utt2.lower()))
    # print("GRAPH: ", set(res))
    return set(res)

def all_utterances_present(G: BaseGraph, dialogues: list[Dialogue]) -> bool:
    """
    Check if all graph utterances match with utterances in set of dialogues.

    Args:
        G: BaseGraph object containing the dialogue graph
        dialogues: List of Dialogue objects to check against

    """
    # Get all unique utterances from nodes and edges in the graph
    graph_utterances = set()

    # Add node utterances
    for node_id, node_data in G.graph.nodes(data=True):
        graph_utterances.update([u.lower() for u in node_data["utterances"]])

    # Add edge utterances
    for _, _, edge_data in G.graph.edges(data=True):
        if isinstance(edge_data["utterances"], list):
            graph_utterances.update([u.lower() for u in edge_data["utterances"]])
        else:
            graph_utterances.add(edge_data["utterances"].lower())

    # Collect all utterances from dialogues
    dialogue_utterances = set()
    for dialogue in dialogues:
        dialogue_utterances.update(utt.text.lower() for utt in dialogue.messages)

    # Check if graph utterances match the dialogues
    if graph_utterances.issubset(dialogue_utterances):
        set1 = dialogue_edges(dialogues)
        set2 = graph_edges(G)
        if len(set1-set2) <= 0:
            print("Graph has all the dialogues")
            for eq in set2-set1:
                print("absent in dialogues: ",eq)
        else:
            for eq in set1-set2:
                print("absent in graph: ",eq)
        if set1 == set2:
            return True
    else:
        print("absent in graph: ", dialogue_utterances - graph_utterances)
        print("absent in dialogues: ", graph_utterances - dialogue_utterances)
        if dialogue_utterances.issubset(graph_utterances):
            print("Graph has all the dialogues")
    return False
    # graph_utterances.difference(dialogue_utterances)


def all_roles_correct(D1: Dialogue, D2: Dialogue) -> bool:
    for phrase_1, phrase_2 in zip(D1.messages, D2.messages):
        if phrase_1.participant != phrase_2.participant:
            return False
    return True


def is_correct_lenght(D1: Dialogue, D2: Dialogue) -> bool:
    return len(D1.messages) == len(D2.messages)


def are_answers_similar(D1: Dialogue, D2: Dialogue, model, threshold: float) -> bool:
    raise NotImplementedError

def compare_edge_lens(G1: BaseGraph, G2: BaseGraph, max: list):
    
    nodes_map = {}
    graph1 = G1.graph_dict
    graph2 = G2.graph_dict
    nodes1 = [n['id'] for n in graph1['nodes']]
    nodes2 = [n['id'] for n in graph2['nodes']]
    for idx,n in enumerate(nodes1):
        nodes_map[n] = nodes2[max[idx]]

    for node1, node2 in zip(nodes1,[nodes_map[n] for n in nodes1]):
        edges1 = G1.edge_by_source(node1)
        edges2 = G2.edge_by_source(node2)
        if len(edges1) != len(edges2):
            return False
        for edge1 in edges1:
            for edge2 in edges2:
                if nodes_map[edge1['target']] == edge2['target'] and len(edge1['utterances']) != len(edge2['utterances']):
                    # print(edge1, edge2)
                    return False
    return True



def compare_graphs(G1: BaseGraph, G2: BaseGraph) -> bool:
    g1 = G1.graph_dict
    g2 = G2.graph_dict

    # print("ORIG: ", g1)
    # g1_order = graph_order(g1)
    # g2_order = graph_order(g2)
    # print("ORDER: ", g1_order, "\n")
    # print("2LIST: ", graph2list(g1_order), "\n")
    #matrix = get_embedding(graph2list(g1_order), graph2list(g2_order), env_settings.EMBEDDER_MODEL, env_settings.EMBEDDER_DEVICE)

    nodes1_list = nodes2list(g1)
    nodes2_list = nodes2list(g2)
    if len(nodes1_list) != len(nodes2_list):
        print("FIRST: ", len(nodes1_list), len(nodes2_list))
        return False

    g1_list, n1, len1 = graph2list(g1)
    g2_list, n2, len2 = graph2list(g2)
    print("LEN1: ", len1, "LEN2: ", len2)
    # for idx,g in enumerate(zip(g1_list,g2_list)):
    #     print(idx, ": ", g[0])
    #     print("G2: ", g[1], "\n")

    nodes_matrix = get_embedding(nodes1_list, nodes2_list, env_settings.EMBEDDER_MODEL, env_settings.EMBEDDER_DEVICE)
    matrix = get_embedding(g1_list, g2_list, env_settings.EMBEDDER_MODEL, env_settings.EMBEDDER_DEVICE)

    # nodes_matrix, matrix = get_2_rerankings(nodes1_list, nodes2_list, g1_list, g2_list)
    nodes_max = list(np.argmax(nodes_matrix, axis=1))
    max = list(np.argmax(matrix, axis=1))
    print("MAX: ", max)
    print("N_MAX: ", nodes_max)
    if len(set(nodes_max)) < len(nodes1_list):
        print("LLLLENS")
        return False



    # print("LENS: ", len1, len2)
    if n1 != n2:
        print("N!")
        return False
    

    # matrix = get_reranking(g1_list, g2_list)

    if len(set(max)) < len(g1_list) or nodes_max != max:
        print("MIX", len(set(max)), len(g1_list), nodes_max)
        return False


    if not compare_edge_lens(G1, G2, max):
        print("LENS")
        return False
    print("NODES: ", np.min(np.max(nodes_matrix, axis=1)))
    print("ALL: ", np.min(np.max(matrix, axis=1)))

    mmin = min(np.min(np.max(nodes_matrix, axis=1)),np.min(np.max(matrix, axis=1)))

    if mmin >= env_settings.SIM_THRESHOLD:
        return True
    # diags = get_diagonals(matrix)
    # # print("DIAGS: ", diags, "\n")
    # sums = np.sum(diags,axis=1)
    # max_index = np.argmax(sums)
    # g1_best = get_diagonal(g1,max_index)
    # min_value = np.min(diags[max_index])
    # print("MIN: ", min_value)
    # return True
    # print("\nG1: ", g1_best, "\n")
    # print("G2: ", g2_order, "\n")

    # if min_value >= env_settings.SIM_THRESHOLD:
    #     return True
    parser = PydanticOutputParser(pydantic_object=CompareResponse)
    format_model=ChatOpenAI(model=env_settings.FORMATTER_MODEL_NAME, api_key=env_settings.OPENAI_API_KEY, base_url=env_settings.OPENAI_BASE_URL)
    model=ChatOpenAI(model=env_settings.COMPARE_MODEL_NAME, api_key=env_settings.OPENAI_API_KEY, base_url=env_settings.OPENAI_BASE_URL)
    new_parser = OutputFixingParser.from_llm(parser=parser, llm=format_model)
    result = call_llm_api(compare_graphs_prompt.format(result_form=result_form,graph_example_1=graph_example_1, graph_1=g1, graph_2=g2), model|new_parser, temp=0).model_dump()
    # print("RES: ", result)
    return result['result']


def ua_match(G: BaseGraph, user: str, assistant: str) -> bool:
    """
    Check if there is a connection from user message to assistant message.

    Args:
        G: BaseGraph object containing the dialogue graph
        user, assistant: pair of neighboring utterances in a dialogue 

    Returns:
        list: True if there is connection, False otherwise 
    """

    nodes = G.nodes_by_utterance(assistant)

    for node in nodes:
        edges = G.edges_by_utterance(user)
        for edge in edges:           
            if edge['target'] == node['id']:
                return True
    return False

def au_match(G: BaseGraph, assistant: str, user: str) -> bool:
    """
    Check if there is a connection from assistant message to user message.

    Args:
        G: BaseGraph object containing the dialogue graph
        assistant, user: pair of neighboring utterances in a dialogue 

    Returns:
        list: True if there is connection, False otherwise 
    """

    nodes = G.nodes_by_utterance(assistant)

    for node in nodes:
        edges = G.edges_by_utterance(user)
        for edge in edges:           
            if edge['source'] == node['id']:
                return True
    return False

def pair_match(G: BaseGraph, msg1: dict, msg2: dict) -> bool:
    """
    Check if there is a connection from msg1 to msg2.

    Args:
        G: BaseGraph object containing the dialogue graph
        msg1, msg2: pair of neighboring utterances in a dialogue 

    Returns:
        list: True if there is connection, False otherwise 
    """
    if msg1.participant == 'assistant' and msg2.participant == 'user':
        return au_match(G, msg1.text, msg2.text)
    if msg1.participant == 'user' and msg2.participant == 'assistant':
        return ua_match(G, msg1.text, msg2.text)
    return False


def dialogues_are_valid_paths(G: BaseGraph, dialogues: list[Dialogue]) -> list:
    """
    Check if all dialogues are valid paths in the graph.

    Args:
        G: BaseGraph object containing the dialogue graph
        dialogues: List of Dialogue objects to check against

    Returns:
        list: for every dialogue either [True] or [False, message1, message2], when there is no connection from message1 to message2
    """


    result = []

    for dialogue in dialogues:
        idx = 0
        for idx in range(len(dialogue.messages)-1):
            if not pair_match(G, dialogue.messages[idx], dialogue.messages[idx+1]):
                result.append([False, dialogue.messages[idx].text, dialogue.messages[idx+1].text])
        result.append([True])
    return result
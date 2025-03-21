import copy
import numpy as np
import networkx as nx
import random
import json
from chatsky_llm_autoconfig.graph import Graph
from chatsky_llm_autoconfig.dialogue import Dialogue
from vectors import DialogueStore, NodeStore
from settings import EnvSettings
from embedder import compare_strings

from langchain.schema import HumanMessage
from langchain_community.embeddings import HuggingFaceEmbeddings

# from langchain_community.cross_encoders.huggingface import HuggingFaceCrossEncoder

env_settings = EnvSettings()
# evaluator = HuggingFaceCrossEncoder(model_name=env_settings.RERANKER_MODEL, model_kwargs={"device": env_settings.EMBEDDER_DEVICE})

# all func are currently unused
def check_if_nodes_identical(graph_1: Graph, graph_2: Graph):
    # check if we have the same amount of nodes:
    if len(graph_1.nodes) != len(graph_2.nodes):
        return False
    # check if the nodes are in the same
    return set(graph_1.nodes) == set(graph_2.nodes)


def check_if_links_identical(graph_1: Graph, graph_2: Graph):
    unmatched_first = []
    unmatched_second = []
    node_cnt = len(graph_1.nodes)
    for i in range(node_cnt):
        for j in range(node_cnt):
            if graph_1.transitions[i][j] is not None and graph_2.transitions[i][j] is not None:
                if graph_1.transitions[i][j].requests == graph_2.transitions[i][j].requests:
                    continue
                else:
                    if set(graph_1.transitions[i][j].requests) == set(graph_2.transitions[i][j].requests):
                        continue
                    else:
                        # print(graph_1.transitions[i][j].requests)
                        # print(graph_2.transitions[i][j].requests)
                        # raise ValueError("The target and source are identical, but the responses aren't")
                        unmatched_first.append((i, j, graph_1.transitions[i][j].requests))
                        unmatched_second.append((i, j, graph_2.transitions[i][j].requests))
            elif graph_1.transitions[i][j] is not None:
                unmatched_first.append((i, j, graph_1.transitions[i][j].requests))
            elif graph_2.transitions[i][j] is not None:
                unmatched_second.append((i, j, graph_2.transitions[i][j].requests))
            else:
                continue
    return unmatched_first, unmatched_second


def check_graph_isomorphism(graph1, graph2):
    if not check_if_nodes_identical(graph1, graph2):
        return False

    unmatched_first, unmatched_second = check_if_links_identical(graph1, graph2)

    for edge in unmatched_first:
        print(edge)
    print("_______")
    for edge in unmatched_second:
        print(edge)
    print("_______")


def find_split_nodes(g1, g2):
    # Create dictionaries to map edges based on 'requests' attribute
    def map_edges_by_request(graph):
        requests_map = {}
        edges_map = {}
        for u, v, data in graph.edges(data=True):
            request = data.get("requests")
            if request not in requests_map:
                requests_map[request] = []
            requests_map[request].append((u, v))
            key = f"{u}->{v}"
            if key not in edges_map:
                edges_map[key] = []
            edges_map[key].append(*data.values())
        return requests_map, edges_map

    def find_splits(graph_edges, other_graph_edges, requests_map, graph_split, other_requests_map):
        for edge, data in graph_edges.items():
            if len(data) > 1 and len(other_graph_edges.get(edge, [])) < len(data):
                node = int(edge.split("->")[1])
                end_nodes = [other_requests_map[request][0][1] for request in data]
                graph_split[node] = end_nodes

    g1_requests, g1_edges = map_edges_by_request(g1)
    g2_requests, g2_edges = map_edges_by_request(g2)

    g1_split = {}
    g2_split = {}
    find_splits(g1_edges, g2_edges, g1_requests, g1_split, g2_requests)
    find_splits(g2_edges, g1_edges, g2_requests, g2_split, g1_requests)

    for node, split_nodes in g1_split.items():
        print(f"In g1, node {node} is split into {split_nodes} in g2")
    for node, split_nodes in g2_split.items():
        print(f"In g2, node {node} is split into {split_nodes} in g1")

    return g1_split, g2_split


def do_mapping(g1, g2):
    if isinstance(g1, nx.MultiDiGraph):
        GM = nx.isomorphism.DiGraphMatcher(g1, g2, edge_match=lambda x, y: set(x["requests"]).intersection(set(y["requests"])) is not None)
    else:
        GM = nx.isomorphism.MultiDiGraphMatcher(
            g1,
            g2,
            edge_match=lambda x, y: set([elem["requests"] for elem in list(x.values())]).intersection(
                set([elem["requests"] for elem in list(y.values())])
            )
            is not None,
        )

    if GM.is_isomorphic():
        print("Graphs are isomorphic and correct")
        mapping = nx.vf2pp_isomorphism(g1, g2, node_label=None)
        return mapping

    mapping = {i: i for i in range(1, len(g1.nodes))}
    g1_unmatched_nodes, g2_unmatched_nodes = find_split_nodes(g1, g2)
    print(g1_unmatched_nodes)
    print(g2_unmatched_nodes)
    for k, v in g1_unmatched_nodes.items():
        elem = random.choice(v)
        mapping[k] = elem
        for node in v:
            if node != elem:
                mapping[node] = None
    print(mapping)

def graph2comparable(graph_dict: dict) -> dict:
    if not graph_dict:
        return graph_dict
    new_dict = copy.deepcopy(graph_dict)
    new_edges = []
    for edge in new_dict["edges"]:
        # print("ITERATION: ", edge, [node for node in graph_dict["nodes"] if node["id"] == edge["source"]])
        edge["utterances"] = [next(node for node in graph_dict["nodes"] if node["id"] == edge["source"])["utterances"][0] + " " + edge["utterances"][0]]
        new_edges.append(edge)
    new_dict["edges"] = new_edges
    return new_dict


def call_llm_api(query: str, llm, client=None, temp: float = 0.05, langchain_model=True) -> str | None:
    tries = 0
    while tries < 3:
        try:
            if langchain_model:
                messages = [HumanMessage(content=query)]
                response = llm.invoke(messages)
                return response
            else:
                messages.append({"role": "user", "content": query})
                response_big = client.chat.completions.create(
                    model=llm,  # id модели из списка моделей - можно использовать OpenAI, Anthropic и пр. меняя только этот параметр
                    messages=messages,
                    temperature=0.7,
                    n=1,
                    max_tokens=3000,  # максимальное число ВЫХОДНЫХ токенов. Для большинства моделей не должно превышать 4096
                    extra_headers={"X-Title": "My App"},  # опционально - передача информация об источнике API-вызова
                )
                return response_big.choices[0].message.content

        except Exception as e:
            print(e)
            print("error, retrying...")
            tries += 1
    return None


def save_json(data: dict, filename: str) -> None:
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def read_json(path):
    with open(path, mode="r") as file:
        data = file.read()
    return json.loads(data)



def graph_order(graph: dict) -> dict:
    nodes = []
    edges = []
    node = [node for node in graph["nodes"] if node["is_start"]][0]
    for _ in range(len(graph["nodes"])):
        edge = [e for e in graph['edges'] if e['source']==node["id"]]
        nodes.append(node)
        edges.extend(edge)
        node = [node for node in graph["nodes"] if node["id"]==edge['target']][0]
    return {"edges": edges, "nodes": nodes}

def graph2list(graph: dict) -> tuple[list,int]:
    res = []
    n_edges = 0
    lens = []

    # node = [node for node in graph["nodes"] if node["is_start"]][0]
    for node in graph["nodes"]:
        edges = [e for e in graph['edges'] if e['source']==node["id"]]
        utt = ""
        for n_utt in node['utterances']:
            utt += n_utt + " "
        for edge in edges:
            lens.append(len(edge['utterances']))
            for e_utt in edge['utterances']:
                utt += e_utt + " "
                n_edges += 1
        res.append(utt)

        # node = [node for node in graph["nodes"] if node["id"]==edge['target']][0] 
    return res, n_edges, lens
    # return [n['utterances'][0]+" "+e['utterances'][0] for n,e in zip(graph['nodes'], graph['edges'])]

def nodes2list(graph: dict) -> list:
    res = []

    for node in graph["nodes"]:
        utt = ""
        for n_utt in node['utterances']:
            utt += n_utt + " "
        res.append(utt)

    return res

def get_diagonals(matrix):
    s = matrix.shape[0]
    diag = np.diag(matrix,0)
    for n in range(1,s):
        diag = np.vstack([diag,np.concatenate((np.diag(matrix,n),np.diag(matrix,n-s)))])
    return diag

def get_diagonal(graph, i):
    result = {}
    result['edges'] = graph['edges'][i:] + graph['edges'][:i]
    result['nodes'] = graph['nodes'][i:] + graph['nodes'][:i]
    return result

def nodes2graph(nodes: list, dialogues: list[Dialogue], embeddings: HuggingFaceEmbeddings):
    """  Connecting nodes with edges for searching dialogue utterances in list of nodes based on embedding similarity
    Input: nodes and list of dialogues
    """
    edges = []
    node_store = NodeStore(nodes, embeddings)
    for d in dialogues:
        texts = d.to_list()
        print("TEXTS: ", texts)
        store = DialogueStore(texts, embeddings)
        for n in nodes:
            print("NODE: ", n)
            for u in n['utterances']:
                print("UTT: ", u)
                ids = store.search_assistant(u)
                print("IDS: ", ids)
                if ids:
                    for id,s in zip(ids, store.get_user(ids=ids)):
                        print("USER: ", s)
                        if len(texts) > 2*(int(id)+1):
                            target = node_store.find_node(texts[2*(int(id)+1)]['text'])
                            print("find_node: ", "target: ", target, texts[2*(int(id)+1)]['text'], n['id'], id, s)
                            existing = [e for e in edges if e['source']==n['id'] and e['target']==target]
                            if existing:
                                if not any([compare_strings(e,s,embeddings) for e in existing[0]['utterances']]):
                                    edges = [e for e in edges if e['source']!=n['id'] or e['target']!=target]
                                    edges.append({'source': n['id'], 'target':target, 'utterances': existing[0]['utterances']+[s]})
                                    print("EXIST: ", {'source': n['id'], 'target':target, 'utterances': existing[0]['utterances']+[s]})
                                else:
                                    print("NOOO")
                            else:
                                edges.append({'source': n['id'], 'target':target, 'utterances': [s]})
                                print("ADDED: ", {'source': n['id'], 'target':target, 'utterances': [s]})
    return {"edges": edges, "nodes": nodes}


def dialogues2list(dialogues: list[Dialogue]):
    """ Helper pre-pocessing list of dialogues for grouping.
    Returns:
    nodes - list of assistant utterances
    nexts - list of following user's utterances
    starts - list of starting utterances
    neigbhours - dictionary of adjacent assistants utterances
    last_user - sign of that dialogue finishes with user's utterance
    """
    nodes = []
    nexts = []
    starts = []
    neighbours = {}
    last_user = False
    for d in dialogues:
        start = 1
        texts = d.to_list()
        for idx,t in enumerate(texts):
            cur = t['text']
            next = ''
            if t['participant'] == 'assistant':
                neigh_nodes = []
                if idx > 1:
                    neigh_nodes.append(texts[idx-2]['text'])
                if idx < len(texts) - 2:
                    neigh_nodes.append(texts[idx+2]['text'])
                if cur in neighbours:
                    neighbours[cur] = neighbours[cur].union(set(neigh_nodes))
                else:
                    neighbours[cur] = set(neigh_nodes)
                if idx < len(texts)-1:
                    next = texts[idx+1]['text']
                if cur in nodes:
                    if next and next not in nexts[nodes.index(cur)]:
                        nexts[nodes.index(cur)].append(next)
                else:
                    nexts.append([next])
                    nodes.append(t['text'])
                    if start:
                        starts.append(t['text'])
                        start = 0
        if t['participant'] == 'user':
            last_user = True
    return nexts, nodes, starts, neighbours, last_user



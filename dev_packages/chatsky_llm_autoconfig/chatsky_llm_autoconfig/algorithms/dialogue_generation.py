import random
import itertools
import networkx as nx
import chatsky_llm_autoconfig.graph as ch_graph
from chatsky_llm_autoconfig.graph import BaseGraph
from chatsky_llm_autoconfig.algorithms.base import DialogueGenerator
# from chatsky_llm_autoconfig.dialogue import Dialogue
from chatsky_llm_autoconfig.schemas import Dialogue
from chatsky_llm_autoconfig.autometrics.registry import AlgorithmRegistry
from chatsky_llm_autoconfig.metrics.automatic_metrics import all_utterances_present
from chatsky_llm_autoconfig.metrics.llm_metrics import find_graph_ends
from chatsky_llm_autoconfig.settings import EnvSettings

from langchain_openai  import ChatOpenAI

env_settings = EnvSettings()

def len_in(a,b):
    return sum([b[x:x + len(a)] == a for x in range(len(b) - len(a) + 1)])

def mix_ends(graph: BaseGraph, ends: list[int], cycles: list[int]):
    # global visited_list
    visited = []
    for c in cycles:
        for e in ends:
            ch_graph.visited_list = [[]]
            graph.find_path(c, e, [])
            if any([e in v for v in ch_graph.visited_list]):
                visited.append(c)
    return [e for e in cycles if e not in visited] + ends

# def all_paths(graph: BaseGraph, start: int, visited: list, repeats: int):
#     global visited_list

#     # if len(visited) < 1 or len_in([visited[-1],start],visited) < repeats:
#     if len(visited) < repeats or not list_in(visited[-repeats:]+[start],visited):
#         # print("LEN: ", len(visited))
#         visited.append(start)
#         for edge in graph.edge_by_source(start):
#             # print("TARGET: ", edge['target'])
#             all_paths(graph, edge['target'], visited.copy(), repeats)
#     visited_list.append(visited)

def all_combinations(path: list, start: dict, next: int, visited: list):
    global visited_list
    visited.append(start)
    # print("APPEND: ", start)
    if next < len(path):
        for utt in path[next]['text']:
            all_combinations(path, {"participant": path[next]['participant'], "text": utt}, next+1, visited.copy())
    visited_list.append(visited)

def get_edges(dialogues: list[list[int]]) -> set[tuple]:
    pairs = []
    for dialogue in dialogues:
        for idx,n in enumerate(dialogue[:-1]):
            pairs.append((n,dialogue[idx+1]))
    return set(pairs)

def edges_in(dialogue: list[int], dialogues: list[list[int]]) -> bool:
    return get_edges([dialogue]).issubset(get_edges(dialogues))

def remove_duplicates(dialogues: list[list[int]]) -> list[list[int]]:
    ds_copy = dialogues.copy()
    idx = 0
    for dialogue in dialogues:
        if edges_in(dialogue, ds_copy[:idx]+ds_copy[idx+1:]):
            ds_copy = ds_copy[:idx]+ds_copy[idx+1:]
        else:
            idx += 1
    return ds_copy

# def get_utts(seq: list[list[dict]]) -> set[tuple[str]]:
#     res = []
#     for dialogue in seq:
#          texts = [d['text'] for d in dialogue]
#          res.extend(texts)
#     return set(res)

def get_utts(seq: list[list[dict]]) -> set[tuple[str]]:
    res = []
    for dialogue in seq:
         user_texts = [d['text'] for d in dialogue if d['participant']=='user']
         assist_texts = [d['text'] for d in dialogue if d['participant']=='assistant']
         if len(assist_texts) > len(user_texts):
             user_texts += [""]
         res.extend([(a,u) for u,a in zip(user_texts,assist_texts)])
    return set(res)

def dialogue_edges(seq: list[list[dict]]) -> set[tuple[str]]:

    res = []
    for dialogue in seq:
         assist_texts = [d['text'] for d in dialogue if d['participant']=='assistant']
         user_texts = [d['text'] for d in dialogue if d['participant']=='user']         
         res.extend([(a1,u,a2) for a1,u,a2 in zip(assist_texts[:-1],user_texts[:len(assist_texts)-1],assist_texts[1:])])
    # print("DIA: ", set(res))
    return set(res)

def remove_duplicated_utts(seq: list[list[dict]]):
    single_seq = [seq[0]]
    for s in seq[1:]:
        if not get_utts([s]).issubset(get_utts(single_seq)) or not dialogue_edges([s]).issubset(dialogue_edges(single_seq)):
            single_seq.append(s)
    return single_seq

def get_dialogues(graph: BaseGraph, repeats: int, ends: list[int]) -> list[Dialogue]:
    global visited_list
    paths = []
    starts = [n for n in graph.graph_dict.get("nodes") if n["is_start"]]
    for s in starts:
        ch_graph.visited_list = [[]]
        graph.all_paths(s['id'], [], repeats)
        paths.extend(ch_graph.visited_list)
    paths.sort()
    final = list(k for k,_ in itertools.groupby(paths))[1:]
    final.sort(key=len,reverse=True)
    # cycles = list(nx.simple_cycles(graph.graph))
    # cycles = [x for xs in cycles for x in xs]
    # if all([f not in cycles for f in finishes]):
    #     finishes += ends
    # print("ENDS: ", ends)
    node_paths = [f for f in final if f[-1] in ends]
    # print("NODES: ", node_paths)
    node_paths = remove_duplicates(node_paths)
    # print("REM: ", node_paths)
    full_paths = []
    for p in node_paths:
        path = []
        for idx,s in enumerate(p[:-1]):
            path.append({"text": graph.node_by_id(s)['utterances'], "participant": "assistant"})
            sources = graph.edge_by_source(s)
            targets = graph.edge_by_target(p[idx+1])
            edge = [e for e in sources if e in targets][0]
            path.append(({"text": edge['utterances'], "participant": "user"}))
        path.append({"text": graph.node_by_id(p[-1])['utterances'], "participant": "assistant"})
        full_paths.append(path)
    dialogues = []
    for f in full_paths:
        visited_list = [[]]
        # print("BEFORE comb")
        all_combinations(f, {}, 0, [])
        # print("AFTER comb")
        # for v in visited_list:
        #     print("LIST: ", v)
        # print("\n")
        dialogue = [el[1:] for el in visited_list if len(el)==len(f)+1]
        dialogues.extend(dialogue)

    # for d in dialogues:
    #    print("DGS: ", d)
    # print("\n")
    final = list(k for k,_ in itertools.groupby(dialogues))
    # print("BEFORE: ", len(final))
    final = remove_duplicated_utts(final)
    # print("AFTER: ", len(final))
    # for f in final:
    #     print("FINAL: ", f)
    # print("\n")
    result = [Dialogue().from_list(seq) for seq in final]
    return result


# @AlgorithmRegistry.register(input_type=BaseGraph, output_type=Dialogue)
class DialogueSampler(DialogueGenerator):

    def invoke(self, graph: BaseGraph, start_node: int = 1, end_node: int = -1, topic="") -> list[Dialogue]:
        nx_graph = graph.graph
        if end_node == -1:
            end_node = list(nx_graph.nodes)[-1]

        all_dialogues = []
        start_nodes = [n for n, attr in nx_graph.nodes(data=True) if attr.get("is_start", n == start_node)]

        for start in start_nodes:
            # Stack contains: (current_node, path, visited_edges)
            stack = [(start, [], set())]

            while stack:
                current_node, path, visited_edges = stack.pop()

                # Add assistant utterance
                current_utterance = random.choice(nx_graph.nodes[current_node]["utterances"])
                path.append({"text": current_utterance, "participant": "assistant"})

                if current_node == end_node:
                    # Check if the last node has edges and add the last edge utterances
                    edges = list(nx_graph.edges(current_node, data=True))
                    if edges:
                        # Get the last edge's data
                        last_edge_data = edges[-1][2]
                        last_edge_utterance = (
                            random.choice(last_edge_data["utterances"])
                            if isinstance(last_edge_data["utterances"], list)
                            else last_edge_data["utterances"]
                        )
                        path.append({"text": last_edge_utterance, "participant": "user"})

                    all_dialogues.append(Dialogue().from_list(path))
                    path.pop()
                    continue

                # Get all outgoing edges
                edges = list(nx_graph.edges(current_node, data=True))

                # Process each edge
                for source, target, edge_data in edges:
                    edge_key = (source, target)
                    if edge_key in visited_edges:
                        continue

                    # if topic and edge_data.get("theme") != topic:
                    #     continue

                    edge_utterance = random.choice(edge_data["utterances"]) if isinstance(edge_data["utterances"], list) else edge_data["utterances"]

                    # Create new path and visited_edges for this branch
                    new_path = path.copy()
                    new_path.append({"text": edge_utterance, "participant": "user"})

                    new_visited = visited_edges | {edge_key}
                    stack.append((target, new_path, new_visited))

                path.pop()

        return all_dialogues

    async def ainvoke(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)


# @AlgorithmRegistry.register(input_type=BaseGraph, output_type=Dialogue)
class DialoguePathSampler(DialogueGenerator):
    def invoke(self, graph: BaseGraph, start_node: int = 1, end_node: int = -1, topic="") -> list[Dialogue]:
        nx_graph = graph.graph

        # Find all nodes with no outgoing edges (end nodes)
        end_nodes = [node for node in nx_graph.nodes() if nx_graph.out_degree(node) == 0]
        dialogues = []
        # If no end nodes found, return empty list
        if not end_nodes:
            return []

        all_paths = []
        # Get paths from start node to each end node
        for end in end_nodes:
            paths = list(nx.all_simple_paths(nx_graph, source=start_node, target=end))
            all_paths.extend(paths)

        for path in all_paths:
            dialogue_turns = []
            # Process each node and edge in the path
            for i in range(len(path)):
                # Add assistant utterance from current node
                current_node = path[i]
                assistant_utterance = random.choice(nx_graph.nodes[current_node]["utterances"])
                dialogue_turns.append({"text": assistant_utterance, "participant": "assistant"})

                # Add user utterance from edge (if not at last node)
                if i < len(path) - 1:
                    next_node = path[i + 1]
                    edge_data = nx_graph.edges[current_node, next_node]
                    user_utterance = random.choice(edge_data["utterances"]) if isinstance(edge_data["utterances"], list) else edge_data["utterances"]
                    dialogue_turns.append({"text": user_utterance, "participant": "user"})

            dialogues.append(Dialogue().from_list(dialogue_turns))

        return dialogues

    async def ainvoke(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)


# @AlgorithmRegistry.register(input_type=BaseGraph, output_type=Dialogue)
class RecursiveDialogueSampler(DialogueGenerator):

    def invoke(self, graph: BaseGraph, upper_limit: int) -> list[Dialogue]:
        # global visited_list
        repeats = 1

        finishes = graph.get_ends()
        if not finishes:
            cycles = find_graph_ends(graph, model=ChatOpenAI(model=env_settings.GENERATION_MODEL_NAME, api_key=env_settings.OPENAI_API_KEY, base_url=env_settings.OPENAI_BASE_URL, temperature=1))['value']
            finishes = mix_ends(graph, finishes, cycles)
            print("ENDDS: ", finishes)
        while repeats <= upper_limit:
            dialogues = get_dialogues(graph,repeats,finishes)
            # dialogues = get_dialogues(graph,repeats,[4,5,6,7])
            pres = all_utterances_present(graph, dialogues)
            if pres == True:
                # print(f"{repeats} repeats works!")
                break
            # else:
            #     print("DIF: ", pres)
            else:
                repeats += 1
                # print("REPEATS: ", repeats)
        if repeats > upper_limit:
            print("Not all utterances present")
            return []
        return dialogues

    async def ainvoke(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)

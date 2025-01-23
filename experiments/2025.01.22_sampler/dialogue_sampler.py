import itertools
from chatsky_llm_autoconfig.graph import Graph
from chatsky_llm_autoconfig.dialogue import Dialogue

def list_in(a, b):
    return any(map(lambda x: b[x:x + len(a)] == a, range(len(b) - len(a) + 1)))

def all_paths(graph: Graph, start: int, visited: list, repeats: int):
    global visited_list   
    # print("start: ", start, len(visited_list))
    if len(visited) < repeats or not list_in(visited[-repeats:]+[start],visited):
        visited.append(start)
        # print("visited:", visited)
        for edge in graph.edge_by_source(start):
            all_paths(graph, edge['target'], visited.copy(), repeats)
    visited_list.append(visited)

def all_combinations(path: list, start: dict, next: int, visited: list):
    global visited_list
    # print("start: ", start, next)
    visited.append(start)
    if next < len(path):
        for utt in path[next]['text']:
            all_combinations(path, {"participant": path[next]['participant'], "text": utt}, next+1, visited.copy())
    visited_list.append(visited)

# def all_combinations(path: list, start: int, utt: str, visited: list):
#     global visited_list
#     visited.append({"participant": path[start]['participant'], "text": utt})
#     # print("start: ", start, next)
#     for utt in path[start]['text']:
#         print("ADDED: ", path[start]['participant'], utt)
#         if start < len(path)-1:
#             all_combinations(path, start+1, utt, visited.copy())
#     print("FINISHED")
#     visited_list.append(visited)

def get_dialogues(graph: Graph, repeats: int) -> list[Dialogue]:
    global visited_list
    paths = []
    starts = [n for n in graph.graph_dict.get("nodes") if n["is_start"]]
    for s in starts:
        visited_list = [[]]
        all_paths(graph, s['id'], [], repeats)
        paths.extend(visited_list)




    paths.sort()
    final = list(k for k,_ in itertools.groupby(paths))[1:]
    # print("FINAL:", final)
    sources = list(set([g['source'] for g in graph.graph_dict['edges']]))
    ends = [g['id'] for g in graph.graph_dict['nodes'] if g['id'] not in sources]
    # print("ENDS: ", ends)
    node_paths = [f for f in final if f[-1] in ends]
    # print("NODES: ", node_paths)
    full_paths = []
    for p in node_paths:
        path = []
        for idx,s in enumerate(p[:-1]):
            path.append({"participant": "assistant", "text": graph.node_by_id(s)['utterances']})
            # path.append({"user": list(set(gr.edge_by_source(s)) & set(gr.edge_by_target(p[idx+1])))[0]['utterances']})
            sources = graph.edge_by_source(s)
            targets = graph.edge_by_target(p[idx+1])
            # targets = set([(e['source'],e['target']) for e in gr.edge_by_target(p[idx+1])])
            edge = [e for e in sources if e in targets][0]
            path.append(({"participant": "user", "text": edge['utterances']}))
        path.append({"participant": "assistant", "text": graph.node_by_id(p[-1])['utterances']})
        full_paths.append(path)
    dialogues = []
    for f in full_paths:
        visited_list = [[]]
        all_combinations(f, {}, 0, [])
        dialogue = [el[1:] for el in visited_list if len(el)==len(f)+1]
        dialogues.extend(dialogue)
    # for d in dialogues:
    #     print("DIAL: ", d)
    final = list(k for k,_ in itertools.groupby(dialogues))
    result = [Dialogue.from_list(seq) for seq in final]
    return result

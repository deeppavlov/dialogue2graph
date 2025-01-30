import itertools
from chatsky_llm_autoconfig.graph import Graph
from chatsky_llm_autoconfig.schemas import Dialogue
from chatsky_llm_autoconfig.metrics.automatic_metrics import all_utterances_present

def list_in(a, b):
    return any(map(lambda x: b[x:x + len(a)] == a, range(len(b) - len(a) + 1)))

# def all_paths(graph: Graph, start: int, visited: list, repeats: int):
#     global visited_list   
#     # print("start: ", start, len(visited_list))
#     if len(visited) < repeats or not list_in(visited[-repeats:]+[start],visited):
#         visited.append(start)
#         # print("visited:", visited)
#         for edge in graph.edge_by_source(start):
#             all_paths(graph, edge['target'], visited.copy(), repeats)
#     visited_list.append(visited)

def count_sublists(lst, sublist):
  if len(lst) < len(sublist):
    return 0
  if lst[:len(sublist)] == sublist:
      return 1 + count_sublists(lst[len(sublist):], sublist)
  else:
     return count_sublists(lst[1:], sublist)

def all_paths(graph: Graph, start: int, visited: list[int], repeats: int, addition: list[int]):
    global visited_list   
    # print("start: ", start, len(visited_list))
    if visited:
        before = count_sublists(visited,[visited[-1],start])
        p_len = graph.pair_number(visited[-1],start)
        print("PAIR: ", before, start, visited, addition)
        if before >= 0:
            if addition[0] <= 0:
                addition[0] += 1
        if p_len > 1:
            comp_add = 0
        else:
            comp_add = addition[0]


        if before == 0:
            if addition[0] == -1 and not list_in(visited[-repeats:]+[start],visited):
                visited.append(start)
                print("NOT")
                for edge in graph.edge_by_source(start):
                    all_paths(graph, edge['target'], visited.copy(), repeats, addition.copy())
            elif before < p_len + comp_add and not list_in(visited[-repeats:]+[start],visited):
                print("SMALLER")
                visited.append(start)
                for edge in graph.edge_by_source(start):
                    all_paths(graph, edge['target'], visited.copy(), repeats, addition.copy())
        else:
            print("+++")
            if before < p_len + comp_add and not list_in(visited[-repeats:]+[start],visited):
                print("SMALLER")
                visited.append(start)
                for edge in graph.edge_by_source(start):
                    all_paths(graph, edge['target'], visited.copy(), repeats, addition.copy())
    elif not list_in(visited[-repeats:]+[start],visited):
        print("NOT")
        visited.append(start)
        for edge in graph.edge_by_source(start):
            all_paths(graph, edge['target'], visited.copy(), repeats, addition.copy())
    # if not visited or len([e for e in visited if e==start]) < graph.pair_number(visited[-1],start):
    # if start not in visited:
    # if not visited or len([e for e in visited if e==start]) < 2:
            # visited.append(start)
        # print("visited:", visited)
            # for edge in graph.edge_by_source(start):
            #     all_paths(graph, edge['target'], visited.copy(), repeats, addition.copy())
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
        all_paths(graph, s['id'], [], repeats, [-1])
        paths.extend(visited_list)

    paths.sort()
    final = list(k for k,_ in itertools.groupby(paths))[1:]
    # print("FINAL:", final)
    sources = list(set([g['source'] for g in graph.graph_dict['edges']]))
    ends = [g['id'] for g in graph.graph_dict['nodes'] if g['id'] not in sources]
    # print("ENDS: ", ends)
    node_paths = [f for f in final if f[-1] in ends]
    for n in node_paths:
        print("NODES: ", n)
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

def get_full_dialogues(graph: Graph, upper_limit: int):
    repeats = 1
    while repeats <= upper_limit:
        print("REPEAT: ", repeats)
        dialogues = get_dialogues(graph,repeats)
        if all_utterances_present(graph, dialogues):
            print(f"{repeats} repeats used")            
            break
        repeats += 1
    if repeats >= upper_limit:
        print("Not all utterances present")
    return dialogues
import itertools
from chatsky_llm_autoconfig.graph import Graph
from chatsky_llm_autoconfig.dialog import Dialog
from chatsky_llm_autoconfig.metrics.automatic_metrics import all_utterances_present
from chatsky_llm_autoconfig.metrics.llm_metrics import find_graph_ends
from chatsky_llm_autoconfig.settings import EnvSettings
from langchain_community.chat_models import ChatOpenAI

env_settings = EnvSettings()

model = ChatOpenAI(
    model=env_settings.GENERATION_MODEL_NAME,
    api_key=env_settings.OPENAI_API_KEY,
    base_url=env_settings.OPENAI_BASE_URL,
)


def list_in(a, b):
    return any(map(lambda x: b[x : x + len(a)] == a, range(len(b) - len(a) + 1)))


def all_paths(graph: Graph, start: int, visited: list, repeats: int):
    global visited_list
    if len(visited) < repeats or not list_in(visited[-repeats:] + [start], visited):
        visited.append(start)
        for edge in graph.edge_by_source(start):
            all_paths(graph, edge["target"], visited.copy(), repeats)
    visited_list.append(visited)


def all_combinations(path: list, start: dict, next: int, visited: list):
    global visited_list
    visited.append(start)
    if next < len(path):
        for utt in path[next]["text"]:
            all_combinations(
                path,
                {"participant": path[next]["participant"], "text": utt},
                next + 1,
                visited.copy(),
            )
    visited_list.append(visited)


def get_edges(dialogs: list[list[int]]) -> set[tuple]:
    pairs = []
    for dialog in dialogs:
        for idx, n in enumerate(dialog[:-1]):
            pairs.append((n, dialog[idx + 1]))
    return set(pairs)


def edges_in(dialog: list[int], dialogs: list[list[int]]) -> bool:
    return get_edges([dialog]).issubset(get_edges(dialogs))


def remove_duplicates(dialogs: list[list[int]]) -> list[list[int]]:
    ds_copy = dialogs.copy()
    idx = 0
    for dialog in dialogs:
        if edges_in(dialog, ds_copy[:idx] + ds_copy[idx + 1 :]):
            ds_copy = ds_copy[:idx] + ds_copy[idx + 1 :]
        else:
            idx += 1
    return ds_copy


def get_utts(seq: list[list[dict]]) -> set[tuple[str]]:
    res = []
    for dialog in seq:
        user_texts = [d["text"] for d in dialog if d["participant"] == "user"]
        assist_texts = [d["text"] for d in dialog if d["participant"] == "assistant"]
        res.extend([(a, u) for u, a in zip(user_texts, assist_texts)])
    return set(res)


def remove_duplicated_utts(seq: list[list[dict]]):
    seq_copy = seq.copy()
    idx = 0
    for s in seq:
        if get_utts([s]).issubset(get_utts(seq_copy[:idx] + seq_copy[idx + 1 :])):
            seq_copy = seq_copy[:idx] + seq_copy[idx + 1 :]
        else:
            idx += 1
    return seq_copy


def get_dialogs(graph: Graph, repeats: int, ends: list[int]) -> list[Dialog]:
    global visited_list
    paths = []
    starts = [n for n in graph.graph_dict.get("nodes") if n["is_start"]]
    for s in starts:
        visited_list = [[]]
        all_paths(graph, s["id"], [], repeats)
        paths.extend(visited_list)

    paths.sort()
    final = list(k for k, _ in itertools.groupby(paths))[1:]
    final.sort(key=len, reverse=True)
    # for f in final:
    #     print("FINAL: ", f)
    # sources = list(set([g['source'] for g in graph.graph_dict['edges']]))
    # ends = [g['id'] for g in graph.graph_dict['nodes'] if g['id'] not in sources]

    node_paths = [f for f in final if f[-1] in ends]
    # for n in node_paths:
    #     print("NODE_PATH: ", n)
    # node_paths = remove_duplicates(final)
    node_paths = remove_duplicates(node_paths)
    # print("REMOVED: ", node_paths)
    # print("\n")
    full_paths = []
    for p in node_paths:
        path = []
        for idx, s in enumerate(p[:-1]):
            path.append(
                {"participant": "assistant", "text": graph.node_by_id(s)["utterances"]}
            )
            sources = graph.edge_by_source(s)
            targets = graph.edge_by_target(p[idx + 1])
            edge = [e for e in sources if e in targets][0]
            path.append(({"participant": "user", "text": edge["utterances"]}))
        path.append(
            {"participant": "assistant", "text": graph.node_by_id(p[-1])["utterances"]}
        )
        full_paths.append(path)
    dialogs = []
    for f in full_paths:
        visited_list = [[]]
        all_combinations(f, {}, 0, [])
        dialog = [el[1:] for el in visited_list if len(el) == len(f) + 1]
        dialogs.extend(dialog)
    final = list(k for k, _ in itertools.groupby(dialogs))
    final = remove_duplicated_utts(final)
    # for f in final:
    #     print("FINAL: ", [t['text'] for t in f])
    # print("\n")
    result = [Dialog.from_list(seq) for seq in final]
    return result


def get_full_dialogs(graph: Graph, upper_limit: int):
    repeats = 1
    ends = find_graph_ends(graph, model=model)
    print("ENDS: ", ends["value"], ends["description"])
    while repeats <= upper_limit:
        dialogs = get_dialogs(graph, repeats, ends["value"])
        if all_utterances_present(graph, dialogs):
            print(f"{repeats} repeats works!")
            break
        repeats += 1
    if repeats > upper_limit:
        print("Not all utterances present")
    return dialogs

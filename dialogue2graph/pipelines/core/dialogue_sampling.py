import itertools
from typing import Literal
import pandas as pd
from typing import Optional
import dialogue2graph.pipelines.core.graph as ch_graph
from dialogue2graph.pipelines.core.graph import BaseGraph
from dialogue2graph.pipelines.core.dialogue import Dialogue
from dialogue2graph.pipelines.core.algorithms import DialogueGenerator
from dialogue2graph.metrics.no_llm_metrics import all_utterances_present
from dialogue2graph.datasets.complex_dialogues.find_graph_ends import find_graph_ends
from langchain_openai  import ChatOpenAI
from pydantic_settings import BaseSettings, SettingsConfigDict

class EnvSettings(BaseSettings, case_sensitive=True):

    model_config = SettingsConfigDict(env_file="./.env", env_file_encoding="utf-8")
    OPENAI_API_KEY: Optional[str]
    OPENAI_BASE_URL: Optional[str]
    HUGGINGFACE_TOKEN: Optional[str]
    EMBEDDER_DEVICE: Optional[str]

env_settings = EnvSettings()

# @AlgorithmRegistry.register(input_type=BaseGraph, output_type=Dialogue)
class RecursiveDialogueSampler(DialogueGenerator):

    def invoke(self, graph: BaseGraph, upper_limit: int, model_name: str="o1-mini") -> list[Dialogue]:
        # TODO: how to add caching?
        repeats = 1
        finishes = graph.get_ends()
        if not finishes:
            cycles = find_graph_ends(graph, model=ChatOpenAI(model=model_name, api_key=env_settings.OPENAI_API_KEY, base_url=env_settings.OPENAI_BASE_URL, temperature=1))['value']
            finishes = mix_ends(graph, finishes, cycles)
            print("ENDDS: ", finishes)
        while repeats <= upper_limit:
            dialogues = get_dialogues(graph,repeats,finishes)
            if dialogues:
                if all_utterances_present(graph, dialogues):
                    # print(f"{repeats} repeats works!")
                    break
            repeats += 1
            # print("REPEATS: ", repeats)
        if repeats > upper_limit:
            print("Not all utterances present")
            # return []
        return dialogues



    async def ainvoke(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)

    async def evaluate(self, graph, upper_limit, target_dialogues, report_type=Literal["dict", "dataframe"]):
        dialogues = self.invoke(graph, upper_limit)
        report = {
            "all_utterances_present": [all_utterances_present(graph, dialogues)],
            # "all_roles_correct": all(all_roles_correct(dialogues, target_dialogues)),
        }
        if report_type == "dataframe":
            report = pd.DataFrame.from_dict(report)
            return report
        elif report_type == "dict":
            return report
        else:
            raise ValueError(f"Invalid report_type: {report_type}")

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


def all_combinations(path: list, start: dict, next: int, visited: list):
    global visited_list, counter
    # print("APPEND: ", next, start, len(path))
    visited.append(start)

    if next < len(path):
        # print("MAX: ", max_v)
            for utt in path[next]['text']:
                all_combinations(path, {"participant": path[next]['participant'], "text": utt}, next+1, visited.copy())
    else:
        counter += 1
        if counter%10000000 == 0:
            print("CUR: ", counter)
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
    global visited_list, counter
    paths = []
    starts = [n for n in graph.graph_dict.get("nodes") if n["is_start"]]
    for s in starts:
        ch_graph.visited_list = [[]]
        graph.all_paths(s['id'], [], repeats)
        paths.extend(ch_graph.visited_list)
    paths.sort()
    final = list(k for k,_ in itertools.groupby(paths))[1:]
    final.sort(key=len,reverse=True)
    print("ENDS: ", ends)
    node_paths = [f for f in final if f[-1] in ends]

    if not graph.check_edges(node_paths):
        return False
    print("NODES: ", node_paths)
    node_paths = remove_duplicates(node_paths)
    print("REM: ", node_paths)
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
        counter = 0
        all_combinations(f, {}, 0, [])
        dialogue = [el[1:] for el in visited_list if len(el)==len(f)+1]
        dialogues.extend(dialogue)
    final = list(k for k,_ in itertools.groupby(dialogues))
    final = remove_duplicated_utts(final)

    result = [Dialogue().from_list(seq) for seq in final]
    return result

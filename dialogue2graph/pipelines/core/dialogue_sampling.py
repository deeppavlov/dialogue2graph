"""
Dialogue Sampling
-----------------
This module contains class for sampling dialogs from a graph.
"""
import os
import itertools
from typing import Literal
import pandas as pd
from typing import Optional
import dialogue2graph.pipelines.core.graph as ch_graph
from dialogue2graph.pipelines.core.graph import BaseGraph
from dialogue2graph.pipelines.core.dialogue import Dialogue
from dialogue2graph.pipelines.core.algorithms import DialogueGenerator
from dialogue2graph.metrics.no_llm_metrics import match_triplets_dg
from dialogue2graph.datasets.complex_dialogues.find_graph_ends import find_graph_ends
from langchain_openai import ChatOpenAI
from pydantic_settings import BaseSettings, SettingsConfigDict


class EnvSettings(BaseSettings, case_sensitive=True):
    """Pydantic settings to get env variables"""

    model_config = SettingsConfigDict(
        env_file=os.environ.get("PATH_TO_ENV", ".env"),
        env_file_encoding="utf-8",
        env_file_exists_ok=False,  # Makes .env file optional
    )
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_BASE_URL: Optional[str] = None
    HUGGINGFACE_TOKEN: Optional[str] = None
    SAMPLING_MAX: Optional[int] = 1000000  # Default value
    DEVICE: Optional[str] = "cpu"  # Default value


# Try to load settings, fall back to defaults if fails
try:
    env_settings = EnvSettings()
except Exception:
    env_settings = EnvSettings(_env_file=None)  # Initialize without env file


class RecursiveDialogueSampler(DialogueGenerator):
    """Recursive dialogue sampler for the graph"""

    def invoke(
        self, graph: BaseGraph, upper_limit: int, model_name: str = "o1-mini", temp=1
    ) -> list[Dialogue]:
        """Finds all the dialogues in the graph
        upper_limit is used to limit repeats used in graph.all_paths method
        model_name is LLM to find cycling nodes when list of finishing nodes is empty
        Returns list fo dialogues"""

        # TODO: how to add caching?
        repeats = 1
        cycle_nodes = []
        finished_nodes = graph.get_ends()
        if not finished_nodes:
            cycle_nodes = find_graph_ends(
                graph,
                model=ChatOpenAI(
                    model=model_name,
                    api_key=env_settings.OPENAI_API_KEY,
                    base_url=env_settings.OPENAI_BASE_URL,
                    temperature=temp,
                ),
            )["value"]
        finished_nodes = mix_ends(graph, finished_nodes, cycle_nodes)
        while repeats <= upper_limit:
            dialogues = get_dialogues(graph, repeats, finished_nodes)
            if dialogues:
                if match_triplets_dg(graph, dialogues)["value"]:
                    break
            repeats += 1
        if repeats > upper_limit:
            raise ValueError("Not all utterances present")
        return dialogues

    async def ainvoke(self, *args, **kwargs):
        # TODO: add docs
        return self.invoke(*args, **kwargs)

    async def evaluate(
        self,
        graph,
        upper_limit,
        target_dialogues,
        report_type=Literal["dict", "dataframe"],
    ):
        # TODO: add docs
        dialogues = self.invoke(graph, upper_limit)
        report = {
            "all_utterances_present": [match_triplets_dg(graph, dialogues)].value,
            # "all_roles_correct": all(all_roles_correct(dialogues, target_dialogues)),
        }
        if report_type == "dataframe":
            report = pd.DataFrame.from_dict(report)
            return report
        elif report_type == "dict":
            return report
        else:
            raise ValueError(f"Invalid report_type: {report_type}")


def mix_ends(graph: BaseGraph, ends: list[int], cycles: list[int]) -> list[int]:
    """Find ids of all graph nodes which do not have paths in the graph to any node id in ends.
    So adds more ends to ends when necessary and returns altogether"""
    visited = []
    for c in cycles:
        for e in ends:
            ch_graph.visited_list = [[]]
            graph.find_path(c, e, [])
            if any([e in v for v in ch_graph.visited_list]):
                visited.append(c)
    return [e for e in cycles if e not in visited] + ends


def all_combinations(path: list, start: dict, next: int, visited: list):
    """Recursion to find all utterances combinations in the path of nodes and edges with miltiple utterances
    which start from next element
    visited is path traveled
    visited_list is global variable to store the result
    counter counts number of combinations
    If counter is too big, an error raised"""

    global visited_list, counter
    visited.append(start)

    if next < len(path):
        # print("MAX: ", max_v)
        for utt in path[next]["text"]:
            all_combinations(
                path,
                {"participant": path[next]["participant"], "text": utt},
                next + 1,
                visited.copy(),
            )
    else:
        counter += 1
        if counter == env_settings.SAMPLING_MAX:
            raise ValueError("Too many combinations in the graph")
        if counter % 10000000 == 0:
            print("Number of found combinations: ", counter)
    visited_list.append(visited)


def get_edges(dialogues: list[list[int]]) -> set[tuple[int]]:
    """Find all pairs of adjacent nodes in dialogues"""
    pairs = []
    for dialogue in dialogues:
        for idx, n in enumerate(dialogue[:-1]):
            pairs.append((n, dialogue[idx + 1]))
    return set(pairs)


def edges_in(dialogue: list[int], dialogues: list[list[int]]) -> bool:
    """Checks whether all the pairs of nodes from dialogue are included in dialogues"""
    return get_edges([dialogue]).issubset(get_edges(dialogues))


def remove_duplicates(dialogues: list[list[int]]) -> list[list[int]]:
    """Remove dialogues with duplicated paths from dialogues"""
    ds_copy = dialogues.copy()
    idx = 0
    for dialogue in dialogues:
        if edges_in(dialogue, ds_copy[:idx] + ds_copy[idx + 1 :]):
            ds_copy = ds_copy[:idx] + ds_copy[idx + 1 :]
        else:
            idx += 1
    return ds_copy


def dialogue_doublets(seq: list[list[dict]]) -> set[tuple[str]]:
    """Find all dialogue doublets with (edge, target) utterances"""
    res = []
    for dialogue in seq:
        user_texts = [d["text"] for d in dialogue if d["participant"] == "user"]
        assist_texts = [d["text"] for d in dialogue if d["participant"] == "assistant"]
        if len(assist_texts) > len(user_texts):
            user_texts += [""]
        res.extend([(a, u) for u, a in zip(user_texts, assist_texts)])
    return set(res)


def dialogue_triplets(seq: list[list[dict]]) -> set[tuple[str]]:
    """Find all dialogue triplets with (source, edge, target) utterances"""

    res = []
    for dialogue in seq:
        assist_texts = [d["text"] for d in dialogue if d["participant"] == "assistant"]
        user_texts = [d["text"] for d in dialogue if d["participant"] == "user"]
        res.extend(
            [
                (a1, u, a2)
                for a1, u, a2 in zip(
                    assist_texts[:-1],
                    user_texts[: len(assist_texts) - 1],
                    assist_texts[1:],
                )
            ]
        )
    return set(res)


def remove_duplicated_dialogues(seq: list[list[dict]]) -> list[list[dict]]:
    """Removes duplicated dialogues from list of dialogues seq"""
    single_seq = [seq[0]]
    for s in seq[1:]:
        if not dialogue_doublets([s]).issubset(
            dialogue_doublets(single_seq)
        ) or not dialogue_triplets([s]).issubset(dialogue_triplets(single_seq)):
            single_seq.append(s)
    return single_seq


def get_dialogues(graph: BaseGraph, repeats: int, ends: list[int]) -> list[Dialogue]:
    """Find all the dialogues in the graph finishing with nodes ids from ends
    repeats is used for graph.all_paths method to limit set of sampled dialogues
    visited_list is global variable to store the result of all_combinations method
    counter counts number of combinations there"""

    global visited_list, counter
    paths = []
    starts = [n for n in graph.graph_dict.get("nodes") if n["is_start"]]
    for s in starts:
        ch_graph.visited_list = [[]]
        graph.all_paths(s["id"], [], repeats)
        paths.extend(ch_graph.visited_list)
    paths.sort()
    final_paths = list(k for k, _ in itertools.groupby(paths))[1:]
    final_paths.sort(key=len, reverse=True)
    node_paths = [f for f in final_paths if f[-1] in ends]
    if not graph.check_edges(node_paths):
        return False
    node_paths = remove_duplicates(node_paths)
    full_paths = []
    for p in node_paths:
        path = []
        for idx, s in enumerate(p[:-1]):
            path.append(
                {"text": graph.node_by_id(s)["utterances"], "participant": "assistant"}
            )
            sources = graph.edge_by_source(s)
            targets = graph.edge_by_target(p[idx + 1])
            edge = [e for e in sources if e in targets][0]
            path.append(({"text": edge["utterances"], "participant": "user"}))
        path.append(
            {"text": graph.node_by_id(p[-1])["utterances"], "participant": "assistant"}
        )
        full_paths.append(path)
    dialogues = []
    for f in full_paths:
        visited_list = [[]]
        counter = 0
        all_combinations(f, {}, 0, [])
        dialogue = [el[1:] for el in visited_list if len(el) == len(f) + 1]
        dialogues.extend(dialogue)
    final_dialogues = list(k for k, _ in itertools.groupby(dialogues))
    final_dialogues = remove_duplicated_dialogues(final_dialogues)
    result = [Dialogue().from_list(seq) for seq in final_dialogues]
    return result

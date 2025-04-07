import os
import itertools
import logging
from typing import Literal
import pandas as pd
from typing import Optional
from dialogue2graph.pipelines.core.graph import BaseGraph
from dialogue2graph.pipelines.core.dialogue import Dialogue
from dialogue2graph.pipelines.core.algorithms import DialogueGenerator
from dialogue2graph.metrics.no_llm_metrics import match_dg_triplets, match_dialogue_triplets
from dialogue2graph.datasets.complex_dialogues.find_cycle_ends import find_cycle_ends
from langchain_openai import ChatOpenAI
from pydantic_settings import BaseSettings, SettingsConfigDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class _DialoguePathsCounter:
    counter: int = 0


path_counter = _DialoguePathsCounter()


class EnvSettings(BaseSettings, case_sensitive=True):
    """Pydantic settings to get env variables"""

    model_config = SettingsConfigDict(
        env_file=os.environ.get("PATH_TO_ENV", ".env"), env_file_encoding="utf-8", env_file_exists_ok=False  # Makes .env file optional
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

    def invoke(self, graph: BaseGraph, upper_limit: int, model_name: str = "o1-mini", temp=1) -> list[Dialogue]:
        """Finds all the dialogues in the graph
        upper_limit is used to limit repeats used in graph.get_all_paths method
        model_name is LLM to find cycling nodes when list of finishing nodes is empty
        Returns list fo dialogues"""

        # TODO: how to add caching?
        repeats = 1
        cycle_ends = []
        finished_nodes = graph.get_ends()
        if not finished_nodes:
            cycle_ends = find_cycle_ends(
                graph,
                model=ChatOpenAI(model=model_name, api_key=env_settings.OPENAI_API_KEY, base_url=env_settings.OPENAI_BASE_URL, temperature=temp),
            )["value"]
        finished_nodes = mix_ends(graph, finished_nodes, cycle_ends)
        while repeats <= upper_limit:
            dialogues = get_dialogues(graph, repeats, finished_nodes)
            if dialogues:
                if match_dg_triplets(graph, dialogues)["value"]:
                    break
            repeats += 1
        if repeats > upper_limit:
            raise ValueError("Not all utterances present")
        return dialogues

    async def ainvoke(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)

    async def evaluate(self, graph, upper_limit, target_dialogues, report_type=Literal["dict", "dataframe"]):
        dialogues = self.invoke(graph, upper_limit)
        report = {
            "dialogues_match": [match_dialogue_triplets(dialogues, target_dialogues)["value"]],
            # "all_roles_correct": all(all_roles_correct(dialogues, target_dialogues)),
        }
        if report_type == "dataframe":
            report = pd.DataFrame.from_dict(report)
            return report
        elif report_type == "dict":
            return report
        else:
            raise ValueError(f"Invalid report_type: {report_type}")


def mix_ends(graph: BaseGraph, end_ids: list[int], cycle_ends_ids: list[int]) -> list[int]:
    """Find ids from cycle_ends_ids from where there are no paths to any node id in end_ids.
    So adds more ends to ends when necessary and returns altogether"""
    end_paths = []
    for c in cycle_ends_ids:
        for e in end_ids:
            found_paths = graph.find_paths(c, e, [])
            if any([e in v for v in found_paths]):
                end_paths.append(c)
    return [e for e in cycle_ends_ids if e not in end_paths] + end_ids


def get_all_sequences(full_path: list, last_message: dict, start_idx: int, visited_messages: list):
    """Recursion to find all dialogue suquences in the full_path of nodes and edges with miltiple utterances
    which start from element of full_path with start_idx index number
    visited_messages is path traveled so far
    last_message is last visited message so far
    path_counter counts number of sequences
    If counter is too big, an error raised"""

    visited_messages.append(last_message)
    visited_paths = [[]]

    if start_idx < len(full_path):
        for utt in full_path[start_idx]["text"]:
            visited_paths += get_all_sequences(
                full_path, {"participant": full_path[start_idx]["participant"], "text": utt}, start_idx + 1, visited_messages.copy()
            )
    else:
        path_counter.counter += 1
        if path_counter.counter == env_settings.SAMPLING_MAX:
            raise ValueError("Too many combinations in the graph")
        if path_counter.counter % 10000000 == 0:
            logger.warning("Number of found combinations: ", path_counter.counter)
    visited_paths.append(visited_messages)
    return visited_paths


def get_edges(dialogues: list[list[int]]) -> set[tuple[int]]:
    """Find all pairs of adjacent nodes in dialogues"""
    pairs = []
    for dialogue in dialogues:
        for idx, n in enumerate(dialogue[:-1]):
            pairs.append((n, dialogue[idx + 1]))
    return set(pairs)


def are_edges_in(dialogue: list[int], dialogues: list[list[int]]) -> bool:
    """Checks whether all the pairs of nodes from dialogue are included in dialogues"""
    return get_edges([dialogue]).issubset(get_edges(dialogues))


def remove_duplicates(dialogues: list[list[int]]) -> list[list[int]]:
    """Remove dialogues with duplicated paths from dialogues"""
    ds_copy = dialogues.copy()
    idx = 0
    for dialogue in dialogues:
        if are_edges_in(dialogue, ds_copy[:idx] + ds_copy[idx + 1 :]):
            ds_copy = ds_copy[:idx] + ds_copy[idx + 1 :]
        else:
            idx += 1
    return ds_copy


def get_dialogue_doublets(seq: list[list[dict]]) -> set[tuple[str]]:
    """Find all dialogue doublets with (edge, target) utterances"""
    res = []
    for dialogue in seq:
        user_texts = [d["text"] for d in dialogue if d["participant"] == "user"]
        assist_texts = [d["text"] for d in dialogue if d["participant"] == "assistant"]
        if len(assist_texts) > len(user_texts):
            user_texts += [""]
        res.extend([(a, u) for u, a in zip(user_texts, assist_texts)])
    return set(res)


def get_dialogue_triplets(seq: list[list[dict]]) -> set[tuple[str]]:
    """Find all dialogue triplets with (source, edge, target) utterances"""

    res = []
    for dialogue in seq:
        assist_texts = [d["text"] for d in dialogue if d["participant"] == "assistant"]
        user_texts = [d["text"] for d in dialogue if d["participant"] == "user"]
        res.extend([(a1, u, a2) for a1, u, a2 in zip(assist_texts[:-1], user_texts[: len(assist_texts) - 1], assist_texts[1:])])
    return set(res)


def remove_duplicated_dialogues(seq: list[list[dict]]) -> list[list[dict]]:
    """Removes duplicated dialogues from list of dialogues seq"""
    non_empty_seq = [s for s in seq if s]
    if not non_empty_seq:
        return []
    single_seq = [non_empty_seq[0]]
    for s in non_empty_seq[1:]:
        if not get_dialogue_doublets([s]).issubset(get_dialogue_doublets(single_seq)) or not get_dialogue_triplets([s]).issubset(
            get_dialogue_triplets(single_seq)
        ):
            single_seq.append(s)
    return single_seq


def get_dialogues(graph: BaseGraph, repeats_limit: int, end_nodes_ids: list[int]) -> list[Dialogue]:
    """Find all the dialogues in the graph finishing with end_nodes_ids
    repeats_limit is used for graph.all_paths method to limit set of sampled dialogues"""

    paths = []
    starts = [n for n in graph.graph_dict.get("nodes") if n["is_start"]]
    for s in starts:
        visited_paths = graph.get_all_paths(s["id"], [], repeats_limit)
        paths.extend(visited_paths)
    paths.sort()
    final_paths = list(k for k, _ in itertools.groupby(paths))[1:]
    final_paths.sort(key=len, reverse=True)
    node_paths = [f for f in final_paths if f[-1] in end_nodes_ids]
    if not graph.check_edges(node_paths):
        return False
    node_paths = remove_duplicates(node_paths)
    full_paths = []
    for p in node_paths:
        path = []
        for idx, s in enumerate(p[:-1]):
            path.append({"text": graph.node_by_id(s)["utterances"], "participant": "assistant"})
            sources = graph.edge_by_source(s)
            targets = graph.edge_by_target(p[idx + 1])
            edge = [e for e in sources if e in targets][0]
            path.append(({"text": edge["utterances"], "participant": "user"}))
        path.append({"text": graph.node_by_id(p[-1])["utterances"], "participant": "assistant"})
        full_paths.append(path)
    dialogues = []
    for f in full_paths:
        path_counter.counter = 0
        paths_combs = get_all_sequences(f, {}, 0, [])
        dialogue = [el[1:] for el in paths_combs if len(el) == len(f) + 1]
        dialogues.extend(dialogue)
    final_dialogues = list(k for k, _ in itertools.groupby(dialogues))
    final_dialogues = remove_duplicated_dialogues(final_dialogues)
    result = [Dialogue().from_list(seq) for seq in final_dialogues]
    return result

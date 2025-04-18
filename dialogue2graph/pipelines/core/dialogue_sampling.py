"""
Dialogue Sampling
-----------------

The module contains class for sampling dialogs from a graph.
"""

import itertools
import logging
from typing import Literal
import pandas as pd
from dialogue2graph.pipelines.core.graph import BaseGraph
from dialogue2graph.pipelines.core.dialogue import Dialogue
from dialogue2graph.pipelines.core.algorithms import DialogueGenerator
from dialogue2graph.metrics.no_llm_metrics import (
    match_dg_triplets,
    match_dialogue_triplets,
)
from dialogue2graph.pipelines.helpers.find_cycle_ends import find_cycle_ends
from langchain_core.language_models.chat_models import BaseChatModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class _DialogPathsCounter:
    counter: int = 0


class RecursiveDialogueSampler(DialogueGenerator):
    """Recursive dialog sampler for the graph"""

    def invoke(
        self,
        graph: BaseGraph,
        cycle_ends_model: BaseChatModel,
        upper_limit: int,
        sampling_max: int = 1000000,
    ) -> list[Dialogue]:
        """Extract all the dialogues from the graph

        Args:
            graph: used to extract dialogues from it
            cycle_ends_model: LLM(BaseChatModel) to find cycling ends of the graph
            upper_limit: limits from above repeats_limit used in recursive get_dialogues method
            sampling_max: maximum number of found dialogues

        Returns:
            list of dialogues

        Raises:
            ValueError: "Not all utterances present" if match_dg_triplets returns False
        """

        # TODO: how to add caching?
        repeats = 1
        cycle_ends = []
        finish_nodes = graph.get_ends()
        if not finish_nodes:
            cycle_ends = find_cycle_ends(graph, cycle_ends_model)["value"]
        finish_nodes = mix_ends(graph, finish_nodes, cycle_ends)
        while repeats <= upper_limit:
            try:
                dialogues = get_dialogues(graph, repeats, finish_nodes, sampling_max)
            except ValueError:
                dialogues = []
            if dialogues:
                if match_dg_triplets(graph, dialogues)["value"]:
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
            "dialogues_match": [
                match_dialogue_triplets(dialogues, target_dialogues)["value"]
            ],
            # "all_roles_correct": all(all_roles_correct(dialogues, target_dialogues)),
        }
        if report_type == "dataframe":
            report = pd.DataFrame.from_dict(report)
            return report
        elif report_type == "dict":
            return report
        else:
            raise ValueError(f"Invalid report_type: {report_type}")


def mix_ends(
    graph: BaseGraph, end_ids: list[int], cycle_ends_ids: list[int]
) -> list[int]:
    """Find ids from cycle_ends_ids which do not have paths to any node id from end_ids.

    Args:
        graph: graph to work with
        end_ids: finishing graph node ids
        cycle_ends_ids: ids of graph nodes looping cycles in the graph

    Returns:
        Adds found ids to end_ids and returns as a result
    """
    end_paths = []
    for c in cycle_ends_ids:
        for e in end_ids:
            found_paths = graph.find_paths(c, e, [])
            if any([e in v for v in found_paths]):
                end_paths.append(c)
    return [e for e in cycle_ends_ids if e not in end_paths] + end_ids


def get_all_sequences(
    path: list[dict],
    last_message: dict,
    start_idx: int,
    visited_messages: list,
    sampling_max: int,
    path_counter: _DialogPathsCounter,
) -> list[list[dict]]:
    """Find all dialogue sequences recursively in the path of nodes and edges with miltiple utterances

    Args:
        path: dialogue path with multiple utterances
        start_idx: index in path to start from
        visited_messages: path traveled so far
        last_message: last visited message so far
        sampling_max: maximum number of found sequences

    Returns:
        All found sequences

    Raises:
        path_counter counts number of sequences
        If counter exceeds sampling_max, ValueError raised
    """

    visited_messages.append(last_message)
    dialogues = [[]]

    if start_idx < len(path):
        for utt in path[start_idx]["text"]:
            dialogues += get_all_sequences(
                path,
                {"participant": path[start_idx]["participant"], "text": utt},
                start_idx + 1,
                visited_messages.copy(),
                sampling_max,
                path_counter,
            )
    else:
        path_counter.counter += 1
        if path_counter.counter == sampling_max:
            raise ValueError("Sampling Max exceeded")
        if path_counter.counter % 10000000 == 0:
            logger.warning("Number of found combinations: ", path_counter.counter)
    dialogues.append(visited_messages)
    return dialogues


def remove_duplicated_paths(node_paths: list[list[int]]) -> list[list[int]]:
    """Remove duplicating paths from node_paths

    Args:
        node_paths: list of dialog graph paths in a form of node ids

    Returns:
        List of node paths without duplications
    """
    edges = set()
    res = []
    for path in node_paths:
        path_edges = set((path[i], path[i + 1]) for i in range(len(path) - 1))
        if not path_edges.issubset(edges):
            edges.update(path_edges)
            res.append(path)
    return res


def get_dialogue_doublets(seq: list[list[dict]]) -> set[tuple[str]]:
    """Find all dialogue doublets with (edge, target) utterances

    Args:
        seq: sequence of dialogs

    Returns:
        Set of (user_utterance, assistant_utterance)
    """
    doublets = set()
    for dialogue in seq:
        user_texts = [d["text"] for d in dialogue if d["participant"] == "user"]
        assist_texts = [d["text"] for d in dialogue if d["participant"] == "assistant"]
        if len(assist_texts) > len(user_texts):
            user_texts += [""]
        doublets.update(zip(user_texts, assist_texts))
    return doublets


def get_dialogue_triplets(seq: list[list[dict]]) -> set[tuple[str]]:
    """Find all dialogue triplets with (source, edge, target) utterances

    Args:
        seq: sequence of dialogs
    Returns:
        Set of (assistant_utterance, user_utterance, assistant_utterance)
    """
    triplets = set()
    for dialogue in seq:
        assist_texts = [d["text"] for d in dialogue if d["participant"] == "assistant"]
        user_texts = [d["text"] for d in dialogue if d["participant"] == "user"]
        for i in range(len(assist_texts) - 1):
            triplets.add((assist_texts[i], user_texts[i], assist_texts[i + 1]))
    return triplets


def remove_duplicated_dialogues(seq: list[list[dict]]) -> list[list[dict]]:
    """Remove duplicated dialogues from list of dialogs seq

    Args:
        seq: sequence of dialogs

    Returns:
        List of dialogs without duplications
    """
    non_empty_seq = [s for s in seq if s]
    if not non_empty_seq:
        return []
    uniq_seq = [non_empty_seq[0]]
    for s in non_empty_seq[1:]:
        if not get_dialogue_doublets([s]).issubset(
            get_dialogue_doublets(uniq_seq)
        ) or not get_dialogue_triplets([s]).issubset(get_dialogue_triplets(uniq_seq)):
            uniq_seq.append(s)
    return uniq_seq


def get_dialogues(
    graph: BaseGraph,
    repeats_limit: int,
    end_nodes_ids: list[int],
    sampling_max: int,
) -> list[Dialogue]:
    """Find all the dialogues in the graph finishing with end_nodes_ids

    Args:
        graph: graph to work with
        repeats_limit: used for graph.all_paths method to limit set of sampled dialogues
        end_nodes_ids: ids of nodes finishing dialogs to find
        sampling_max: maximum number of found sequences

    Returns:
        list of dialogs
    """

    node_paths = []
    start_nodes = {n["id"] for n in graph.graph_dict.get("nodes") if n["is_start"]}
    for s in start_nodes:
        node_paths.extend(graph.get_all_paths(s, [], repeats_limit))
    node_paths.sort()
    node_paths = list(k for k, _ in itertools.groupby(node_paths))[1:]
    node_paths.sort(key=len, reverse=True)

    node_paths = [f for f in node_paths if f[-1] in end_nodes_ids]
    if not graph.check_edges(node_paths):
        return []
    node_paths = remove_duplicated_paths(node_paths)
    dialogue_paths = []
    for path in node_paths:
        dialogue = []
        for idx, s in enumerate(path[:-1]):
            dialogue.append(
                {
                    "text": graph.get_nodes_by_id(s)["utterances"],
                    "participant": "assistant",
                }
            )
            sources = graph.get_edges_by_source(s)
            targets = graph.get_edges_by_target(path[idx + 1])
            edge = next((e for e in sources if e in targets), None)
            dialogue.append({"text": edge["utterances"], "participant": "user"})
        dialogue.append(
            {
                "text": graph.get_nodes_by_id(path[-1])["utterances"],
                "participant": "assistant",
            }
        )
        dialogue_paths.append(dialogue)
    dialogues = []
    for path in dialogue_paths:
        single_path = get_all_sequences(
            path, {}, 0, [], sampling_max, _DialogPathsCounter()
        )
        dialogue = [el[1:] for el in single_path if len(el) == len(path) + 1]
        dialogues.extend(dialogue)
    dialogues = remove_duplicated_dialogues(dialogues)
    return [Dialogue().from_list(seq) for seq in dialogues]

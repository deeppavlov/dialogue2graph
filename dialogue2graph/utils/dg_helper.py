from dialogue2graph.pipelines.core.dialogue import Dialogue
from dialogue2graph.utils.vector_stores import DialogueStore, NodeStore
from dialogue2graph.metrics.similarity import compare_strings

from langchain_community.embeddings import HuggingFaceEmbeddings


def connect_nodes(
    nodes: list[dict], dialogues: list[Dialogue], utt_sim: HuggingFaceEmbeddings
) -> dict[str,list[dict]]:
    """Connecting dialog graph nodes with edges via searching dialogue utterances
    in list of nodes based on utt_sim similarity model
    Args:
      nodes: list of dictionaries in a form {"id": id, "label": label, "is_start": bool, "utterances": list}
      dialogues: list of dialogues to build a graph from
      utt_sim: similarity model used to build vector stores
    Returns:
    graph dict in a form {"edges": [edges], "nodes": [nodes]}
    """
    edges = []
    node_store = NodeStore(nodes, utt_sim)
    for dialogue in dialogues:
        turns = dialogue.to_list()
        dialog_store = DialogueStore(turns, utt_sim)
        for node in nodes:
            for utt in node["utterances"]:
                ids = dialog_store.search_assistant(utt)
                if ids:
                    for id, user_utt in zip(ids, dialog_store.get_user_by_id(ids=ids)):
                        if len(turns) > 2 * (int(id) + 1):
                            target = node_store.find_node(
                                turns[2 * (int(id) + 1)]["text"]
                            )
                            existing_edges = [
                                e
                                for e in edges
                                if e["source"] == node["id"] and e["target"] == target
                            ]
                            if existing_edges:
                                if not any(
                                    [
                                        compare_strings(e, user_utt, utt_sim)
                                        for e in existing_edges[0]["utterances"]
                                    ]
                                ):
                                    edges = [
                                        e
                                        for e in edges
                                        if e["source"] != node["id"]
                                        or e["target"] != target
                                    ]
                                    edges.append(
                                        {
                                            "source": node["id"],
                                            "target": target,
                                            "utterances": existing_edges[0]["utterances"]
                                            + [user_utt],
                                        }
                                    )
                            else:
                                edges.append(
                                    {
                                        "source": node["id"],
                                        "target": target,
                                        "utterances": [user_utt],
                                    }
                                )
    return {"edges": edges, "nodes": nodes}


def get_helpers(dialogues: list[Dialogue]) -> tuple[list[str], list[set[str]]]:
    """Helper pre-pocessing list of dialogues for grouping.
    Args:
      dialogues: list of dialogues
    Returns:
      node_utts - list of assistant utterances
      start_utts - list of starting utterances
      user_end - sign of that dialogue finishes with user's utterance
    """

    node_utts = []
    start_utts = []
    user_end = False
    for dialogue in dialogues:
        start_flag = 1
        turns = dialogue.to_list()
        for turn in turns:
            cur_utt = turn["text"]
            if turn["participant"] == "assistant":
                if start_flag:
                    start_utts.append(turn["text"])
                    start_flag = 0
                if cur_utt not in node_utts:
                    node_utts.append(turn["text"])
        if turn["participant"] == "user":
            user_end = True
    return node_utts, list(set(start_utts)), user_end

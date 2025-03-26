from dialogue2graph.pipelines.core.dialogue import Dialogue

from .vector_stores import DialogueStore, NodeStore
from dialogue2graph.metrics.similarity import compare_strings

from langchain_community.embeddings import HuggingFaceEmbeddings


def connect_nodes(nodes: list, dialogues: list[Dialogue], utt_sim: HuggingFaceEmbeddings) -> dict:
    """Connecting dialogue graph nodes with edges via searching dialogue utterances in list of nodes based on utt_sim similarity model
    Input: nodes and list of dialogues
    """
    edges = []
    node_store = NodeStore(nodes, utt_sim)
    for d in dialogues:
        texts = d.to_list()
        store = DialogueStore(texts, utt_sim)
        for n in nodes:
            for u in n["utterances"]:
                ids = store.search_assistant(u)
                if ids:
                    for id, s in zip(ids, store.get_user(ids=ids)):
                        if len(texts) > 2 * (int(id) + 1):
                            target = node_store.find_node(texts[2 * (int(id) + 1)]["text"])
                            existing = [e for e in edges if e["source"] == n["id"] and e["target"] == target]
                            if existing:
                                if not any([compare_strings(e, s, utt_sim) for e in existing[0]["utterances"]]):
                                    edges = [e for e in edges if e["source"] != n["id"] or e["target"] != target]
                                    edges.append({"source": n["id"], "target": target, "utterances": existing[0]["utterances"] + [s]})
                            else:
                                edges.append({"source": n["id"], "target": target, "utterances": [s]})
    return {"edges": edges, "nodes": nodes}


def get_helpers(dialogues: list[Dialogue]) -> tuple[list[str], list[set[str]]]:
    """Helper pre-pocessing list of dialogues for grouping.
    Returns:
    nodes - list of assistant utterances
    starts - list of starting utterances
    last_user - sign of that dialogue finishes with user's utterance
    """

    nodes = []
    starts = []
    last_user = False
    for d in dialogues:
        start = 1
        texts = d.to_list()
        for idx, t in enumerate(texts):
            cur = t["text"]
            if t["participant"] == "assistant":
                if start:
                    starts.append(t["text"])
                    start = 0
                neigh_nodes = []
                if idx > 1:
                    neigh_nodes.append(texts[idx - 2]["text"])
                if idx < len(texts) - 2:
                    neigh_nodes.append(texts[idx + 2]["text"])
                if cur not in nodes:
                    nodes.append(t["text"])
        if t["participant"] == "user":
            last_user = True
    return nodes, list(set(starts)), last_user

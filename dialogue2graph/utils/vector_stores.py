import uuid

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


class DialogueStore:
    """Vector store for dialogues to conduct searching
    User and assistant utterances vectorized separately"""

    assistant_store: Chroma
    user_store: Chroma
    assistant_size: int
    user_size: int
    score_threshold: int

    def _load_dialogue(self, dialogue: list, embedder: HuggingFaceEmbeddings):

        self.assistant_store = Chroma(collection_name=str(uuid.uuid4()), embedding_function=embedder)
        self.user_store = Chroma(collection_name=str(uuid.uuid4()), embedding_function=embedder)
        assistant = [
            Document(page_content=d["text"].lower(), id=id, metadata={"id": id})
            for id, d in enumerate([d for d in dialogue if d["participant"] == "assistant"])
        ]
        user = [
            Document(page_content=d["text"].lower(), id=id, metadata={"id": id})
            for id, d in enumerate(d for d in dialogue if d["participant"] == "user")
        ]
        self.assistant_size = len(assistant)
        self.user_size = len(user)
        self.assistant_store.add_documents(documents=assistant)
        self.user_store.add_documents(documents=user)

    def __init__(self, dialogue: list, embedder: HuggingFaceEmbeddings, score_threshold=0.995):
        self.score_threshold = score_threshold
        self._load_dialogue(dialogue, embedder)

    def search_assistant(self, utterance) -> list[str]:
        """search for utterance over assistant store
        return found documents ids"""
        docs = self.assistant_store.similarity_search_with_relevance_scores(
            utterance.lower(), k=self.assistant_size, score_threshold=self.score_threshold
        )
        res = [d[0].metadata["id"] for d in docs]
        res.sort()
        res = [str(r) for r in res]

        return res

    def get_user(self, ids: list[str]):
        """get utterances of user with ids"""
        res = self.user_store.get(ids=ids)["documents"]
        return res


class NodeStore:
    """Vector store for graph nodes"""

    nodes_store: Chroma
    utterances: list[tuple[str, int]] = []

    def _load_nodes(self, nodes: list, utt_sim: HuggingFaceEmbeddings):

        self.nodes_store = Chroma(collection_name=str(uuid.uuid4()), embedding_function=utt_sim)
        self.utterances = [(u.lower(), n["id"]) for n in nodes for u in n["utterances"]]
        docs = [Document(page_content=u[0], id=id, metadata={"id": id}) for id, u in enumerate(self.utterances)]

        self.nodes_store.add_documents(documents=docs)

    def __init__(self, nodes: list, utt_sim: HuggingFaceEmbeddings):
        self._load_nodes(nodes, utt_sim)

    def find_node(self, utterance: str):
        """Search for node with utterance, return node id or None if not found"""
        docs = self.nodes_store.similarity_search_with_relevance_scores(utterance.lower(), k=1, score_threshold=0.9)
        if docs:
            return self.utterances[docs[0][0].metadata["id"]][1]
        return None

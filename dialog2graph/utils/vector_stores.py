"""
Vector Stores
-------------

The module contains storage of text vectors for dialog turns.
"""

import uuid

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


class DialogStore:
    """Vector store for dialogs to conduct searching
    User and assistant utterances vectorized separately

    Attributes:
      assistant_store: store for assistant utterances
      user_store: store for user utterances
      assistant_size: number of assistant utterances
      user_size: number of user utterances
      _score_threshold: simlarity threshold
    """

    assistant_store: Chroma
    user_store: Chroma
    assistant_size: int
    user_size: int
    _score_threshold: int

    def _load_dialog(
        self,
        dialog: list,
        embedder: HuggingFaceEmbeddings,
    ):
        """Auxiliary method to initialize instance

        Args:
          dialog: list of dicts in a form {"participant": "user" or "assistant", "text": text}
          embedder: embedding function for vector store
        """
        self.assistant_store = Chroma(
            collection_name=str(uuid.uuid4()), embedding_function=embedder
        )
        self.user_store = Chroma(
            collection_name=str(uuid.uuid4()), embedding_function=embedder
        )
        assistant_docs = [
            Document(page_content=turn["text"].lower(), id=id, metadata={"id": id})
            for id, turn in enumerate(
                [d for d in dialog if d["participant"] == "assistant"]
            )
        ]
        user_docs = [
            Document(page_content=turn["text"].lower(), id=id, metadata={"id": id})
            for id, turn in enumerate([d for d in dialog if d["participant"] == "user"])
        ]
        self.assistant_size = len(assistant_docs)
        self.user_size = len(user_docs)
        self.assistant_store.add_documents(documents=assistant_docs)
        self.user_store.add_documents(documents=user_docs)

    def __init__(
        self,
        dialog: list[dict[str, str]],
        embedder: HuggingFaceEmbeddings,
        score_threshold=0.995,
    ):
        """Initialize instance for dialog based on embedder

        Args:
          dialog: list of dicts in a form {"participant": "user" or "assistant", "text": text}
          embedder: embedding function for vector sore
          score_threshold: simlarity threshold
        """
        self._score_threshold = score_threshold
        self._load_dialog(dialog, embedder)

    def search_store(self, store: Chroma, size: int, utterance: str) -> list[str]:
        """Search for utterance over store

        Args:
          store: Chroma store
          size: size of the store
          utterance: utterance to search for
        Returns:
          list of found documents ids
        """

        docs = store.similarity_search_with_relevance_scores(
            utterance.lower(),
            k=size,
            score_threshold=self._score_threshold,
        )
        res = [d[0].metadata["id"] for d in docs]
        res.sort()
        res = [str(r) for r in res]

        return res


    def get_user_by_id(self, ids: list[str]) -> list[str]:
        """Get utterances of user with ids

        Args:
          ids: ids of user documents to get
        Returns:
          list of utterances
        """
        res = self.user_store.get(ids=ids)["documents"]
        return res


class NodeStore:
    """Vector store for graph nodes

    Attributes:
      _nodes_store: store for assistant utterances
      _utterances: list of (node_utterance, node_id)
    """

    _nodes_store: Chroma
    _utterances: list[tuple[str, int]] = []

    def _load_nodes(
        self,
        nodes: list,
        embedder: HuggingFaceEmbeddings,
    ):
        """Auxiliary method to initialize instance

        Args:
          nodes: list of dicts in a form {"id": id, "label": label, "is_start": bool, "utterances": list}
          embedder: embedding function for vector sore
        """
        self._nodes_store = Chroma(
            collection_name=str(uuid.uuid4()), embedding_function=embedder
        )
        self._utterances = [
            (utt.lower(), n["id"]) for n in nodes for utt in n["utterances"]
        ]
        docs = [
            Document(page_content=utt[0], id=id, metadata={"id": id})
            for id, utt in enumerate(self._utterances)
        ]
        self._nodes_store.add_documents(documents=docs)

    def __init__(
        self,
        nodes: list[dict],
        embedder: HuggingFaceEmbeddings,
    ):
        """Initialize instance for nodes based on embedder

        Args:
          nodes: list of dicts in a form {"id": id, "label": label, "is_start": bool, "utterances": list}
          embedder: embedding function for vector sore
        """
        self._load_nodes(nodes, embedder)

    def find_node(self, utterance: str) -> int:
        """Search for node by utterance

        Args:
          utterance: uttetance to search by
        Returns:
          Found node id or None if not found
        """
        docs = self._nodes_store.similarity_search_with_relevance_scores(
            utterance.lower(), k=1, score_threshold=0.9
        )
        if docs:
            return self._utterances[docs[0][0].metadata["id"]][1]
        return None

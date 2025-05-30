import uuid

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from settings import EnvSettings

env_settings = EnvSettings()


class DialogStore:
    assistant_store: Chroma
    user_store: Chroma
    assistant_size: int
    user_size: int

    # def _normalize(self):
    #     done = {}
    #     for d in self.assistant_store.get()['documents']:
    #         if not done[d.page_content]:
    #             docs = self.assistant_store.similarity_search_with_relevance_scores(d.page_content, score_threshold=env_settings.EMBEDDER_THRESHOLD)
    #             ids = [str(d[0].metadata['id']) for d in docs]

    def _load_dialog(self, dialog: list, embeddings: HuggingFaceEmbeddings):
        self.assistant_store = Chroma(
            collection_name=str(uuid.uuid4()), embedding_function=embeddings
        )
        self.user_store = Chroma(
            collection_name=str(uuid.uuid4()), embedding_function=embeddings
        )
        assistant = [
            Document(page_content=d["text"].lower(), id=id, metadata={"id": id})
            for id, d in enumerate(
                [d for d in dialog if d["participant"] == "assistant"]
            )
        ]
        user = [
            Document(page_content=d["text"].lower(), id=id, metadata={"id": id})
            for id, d in enumerate(d for d in dialog if d["participant"] == "user")
        ]
        self.assistant_size = len(assistant)
        self.user_size = len(user)
        self.assistant_store.add_documents(documents=assistant)
        # print("ASSISTANT_STORE: ", self.assistant_store.get())
        self.user_store.add_documents(documents=user)
        # print("USER_STORE: ", self.user_store.get())

    def __init__(self, dialog: list, embeddings: HuggingFaceEmbeddings):
        self._load_dialog(dialog, embeddings)

    def search_assistant(self, utterance):
        docs = self.assistant_store.similarity_search_with_relevance_scores(
            utterance.lower(),
            k=self.assistant_size,
            score_threshold=env_settings.EMBEDDER_TYPO,
        )
        # print("DOCS: ", docs)
        # print("UTT: ", utterance)
        res = [d[0].metadata["id"] for d in docs]
        res.sort()
        res = [str(r) for r in res]

        return res

    # def search_user(self, utterance):
    #     docs = self.user_store.similarity_search_with_relevance_scores(utterance, k=env_settings.DIALOG_MAX, score_threshold=env_settings.EMBEDDER_TYPO)
    #     res = [d[0].metadata['id'] for d in docs]
    #     res.sort()
    #     res = [str(r) for r in res]
    #     return res

    def get_user(self, ids: list[str]):
        res = self.user_store.get(ids=ids)["documents"]
        # print(res)
        return res


class NodeStore:
    nodes: list
    nodes_store: Chroma
    # nodes_dict: dict = {}
    utterances = []

    def _load_nodes(self, nodes: list, embeddings: HuggingFaceEmbeddings):
        self.nodes_store = Chroma(
            collection_name=str(uuid.uuid4()), embedding_function=embeddings
        )

        self.utterances = [(u.lower(), n["id"]) for n in nodes for u in n["utterances"]]
        # self.nodes_dict = {u[0]:u[1] for u in utterances}
        # docs = [Document(page_content=u[0], id=u[1], metadata={"id":u[1]}) for u in utterances]
        docs = [
            Document(page_content=u[0], id=id, metadata={"id": id})
            for id, u in enumerate(self.utterances)
        ]

        self.nodes_store.add_documents(documents=docs)
        # print("NODE_STORE: ", self.nodes_store.get())

    def __init__(self, nodes: list, embeddings: HuggingFaceEmbeddings):
        self._load_nodes(nodes, embeddings)

    def find_node(self, utterance: str):
        # print("LOOK_NODE: ", utterance)
        docs = self.nodes_store.similarity_search_with_relevance_scores(
            utterance.lower(), k=1, score_threshold=0.9
        )
        if docs:
            return self.utterances[docs[0][0].metadata["id"]][1]
            # return self.nodes_dict[docs[0][0].metadata['id']]
        return None

from dialog2graph.pipelines.core.dialog import Dialog

from vectors import DialogStore, NodeStore
from settings import EnvSettings

# from dialog2graph.metrics.embedder import compare_strings
from embedder import compare_strings

from langchain.schema import HumanMessage
from langchain_community.embeddings import HuggingFaceEmbeddings

env_settings = EnvSettings()


def call_llm_api(query: str, llm, client=None, langchain_model=True) -> str | None:
    tries = 0
    while tries < 3:
        try:
            if langchain_model:
                messages = [HumanMessage(content=query)]
                # response = llm.invoke(messages, temperature=temp)
                print("LLM")
                response = llm.invoke(messages)
                return response
            else:
                messages.append({"role": "user", "content": query})
                response_big = client.chat.completions.create(
                    model=llm,  # id модели из списка моделей - можно использовать OpenAI, Anthropic и пр. меняя только этот параметр
                    messages=messages,
                    # temperature=temp,
                    n=1,
                    max_tokens=3000,  # максимальное число ВЫХОДНЫХ токенов. Для большинства моделей не должно превышать 4096
                    extra_headers={
                        "X-Title": "My App"
                    },  # опционально - передача информация об источнике API-вызова
                )
                return response_big.choices[0].message.content

        except Exception as e:
            print(e)
            print("error, retrying...")
            tries += 1
    return None


def nodes2graph(nodes: list, dialogs: list[Dialog], embeddings: HuggingFaceEmbeddings):
    """Connecting nodes with edges for searching dialog utterances in list of nodes based on embedding similarity
    Input: nodes and list of dialogs
    """
    edges = []
    node_store = NodeStore(nodes, embeddings)
    for d in dialogs:
        texts = d.to_list()
        store = DialogStore(texts, embeddings)
        for n in nodes:
            for u in n["utterances"]:
                ids = store.search_assistant(u)
                if ids:
                    for id, s in zip(ids, store.get_user(ids=ids)):
                        if len(texts) > 2 * (int(id) + 1):
                            target = node_store.find_node(
                                texts[2 * (int(id) + 1)]["text"]
                            )
                            existing = [
                                e
                                for e in edges
                                if e["source"] == n["id"] and e["target"] == target
                            ]
                            if existing:
                                if not any(
                                    [
                                        compare_strings(e, s, embeddings)
                                        for e in existing[0]["utterances"]
                                    ]
                                ):
                                    edges = [
                                        e
                                        for e in edges
                                        if e["source"] != n["id"]
                                        or e["target"] != target
                                    ]
                                    edges.append(
                                        {
                                            "source": n["id"],
                                            "target": target,
                                            "utterances": existing[0]["utterances"]
                                            + [s],
                                        }
                                    )

                            else:
                                edges.append(
                                    {
                                        "source": n["id"],
                                        "target": target,
                                        "utterances": [s],
                                    }
                                )

    return {"edges": edges, "nodes": nodes}


def dialogs2list(dialogs: list[Dialog]):
    """Helper pre-pocessing list of dialogs for grouping.
    Returns:
    nodes - list of assistant utterances
    nexts - list of following user's utterances
    starts - list of starting utterances
    neigbhours - dictionary of adjacent assistants utterances
    last_user - sign of that dialog finishes with user's utterance
    """

    nodes = []
    nexts = []
    starts = []
    neighbours = {}
    last_user = False
    for d in dialogs:
        start = 1
        texts = d.to_list()
        for idx, t in enumerate(texts):
            cur = t["text"]
            next = ""
            if t["participant"] == "assistant":
                if start:
                    starts.append(t["text"])
                    start = 0
                neigh_nodes = []
                if idx > 1:
                    neigh_nodes.append(texts[idx - 2]["text"])
                if idx < len(texts) - 2:
                    neigh_nodes.append(texts[idx + 2]["text"])
                if cur in neighbours:
                    neighbours[cur] = neighbours[cur].union(set(neigh_nodes))
                else:
                    neighbours[cur] = set(neigh_nodes)
                if idx < len(texts) - 1:
                    next = texts[idx + 1]["text"]
                if cur in nodes:
                    if next and next not in nexts[nodes.index(cur)]:
                        nexts[nodes.index(cur)].append(next)
                else:
                    nexts.append([next])
                    nodes.append(t["text"])
        if t["participant"] == "user":
            last_user = True
    return nexts, nodes, list(set(starts)), neighbours, last_user

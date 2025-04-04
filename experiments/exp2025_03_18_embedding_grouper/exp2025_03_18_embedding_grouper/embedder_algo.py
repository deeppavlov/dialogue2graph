import copy
from sentence_transformers import SentenceTransformer

# from chatsky_llm_autoconfig.schemas import DialogueMessage
from dialogue2graph.pipelines.core.dialogue import Dialogue
from settings import EnvSettings


env_settings = EnvSettings()
embedding = {}


class EmbeddableString:
    def __init__(self, element: str):
        self.element = element

    def __eq__(self, other):
        return compare_strings(self.element, other.element)

    def __hash__(self):
        return hash("")

    def __str__(self):
        return self.element


def emb_list(x):
    # print("EMB_LIST: ", x)
    return [EmbeddableString(el) for el in x]


def get_embedding(generated: list[str], golden: list[str], emb_name: str, device: str):
    if emb_name not in embedding:
        embedding[emb_name] = SentenceTransformer(emb_name, device=device)

    golden_vectors = embedding[emb_name].encode(golden, normalize_embeddings=True)
    generated_vectors = embedding[emb_name].encode(generated, normalize_embeddings=True)
    similarities = generated_vectors @ golden_vectors.T
    return similarities


def get_cross(search, list2):
    sublist = [p for p in list2 if search in p]
    to_add = []
    for s in sublist:
        if s[0] != search:
            to_add.append(s[0])
        else:
            to_add.append(s[1])
    return len(to_add)


def unite_pairs(pairs: list[tuple[float, tuple[int, int]]]):
    """Clustering nodes based on similar pairs and their scores
    1. Sort pairs in decreasing order based on scores
    2. From the start of this list look for nodes paired with first pair,
    and add them to groups
    """
    pairs_in = copy.deepcopy(pairs)
    pairs_in.sort(reverse=True)
    pairs_in = [x[1] for x in pairs_in]
    groups = []
    while pairs_in:
        cur = [p for p in pairs_in if p[0] in p or p[1] in p]
        # print("CUR: ", cur)
        x = cur[0]
        list1 = [p for p in cur if x[0] in p and x != p]
        list2 = [p for p in cur if x[1] in p and x != p]
        # print("LIST1: ", list1)
        # print("LIST2: ", list2)
        to_add = []
        for y in list1:
            to_add = []
            for el in x:
                if el == y[0]:
                    search = y[1]
                else:
                    search = y[0]
                if get_cross(search, list2):
                    to_add += [search]
            # print("TOADD: ", to_add)

        # Дальше надо объединить их и удалить, потом удаление
        to_add = list(set(([x[0], x[1]] + to_add)))
        groups.append(to_add)
        pairs_in = [p for p in pairs_in if p[0] not in to_add and p[1] not in to_add]

        # print("TO_ADD: ", to_add)
        # print("LEFT: ", pairs_in)
    return groups


def indices(lst, element):
    result = []
    offset = -1
    while True:
        try:
            offset = lst.index(element, offset + 1)
        except ValueError:
            return result
        result.append(offset)


def get_tails(dialogue: Dialogue, assistant: list, node: str):
    # end = assistant.index(node) if node in assistant else None
    # start = assistant_back.index(node) if node in assistant else None
    ids = indices(assistant, node)
    tails = []
    for id in ids:
        tail = dialogue.messages[: id * 2][-4:]
        tail += dialogue.messages[id * 2 + 1 :][:4]
        tails.append(tail)
    return tails


def compare_tails(dialogues: list[Dialogue], node1: str, node2: str):
    tails1 = []
    tails2 = []
    for dialogue in dialogues:
        # user = [d for d in dialogue.messages if d.participant=='user']
        assistant = [d.text for d in dialogue.messages if d.participant == "assistant"]
        # assistant_back = list(reversed(assistant))
        tails1.extend(get_tails(dialogue, assistant, node1))
        tails2.extend(get_tails(dialogue, assistant, node2))

        # if tail1:
        #     tails1.append(tail1)
        # if tail2:
        #     tails2.append(tail2)

    for t1 in tails1:
        for t2 in tails2:
            if t1 == t2:
                # print("\n")
                # print("STARTS: ", node1, t1)
                # print("STARTS: ", node2, t2)

                return True
    return False


def nodes2groups(
    dialogues: list[Dialogue],
    nodes_list: list[str],
    next_list: list[str],
    mix_list: list[str],
    neigbhours: dict,
):
    """Rule based algorithm to group graph nodes
    nodes_list: list of assistant's utterances
    next_list: list of user's utterances
    mix_list: list of nodes and edges concatenation
    neighbours: dictionary of adjacent nodes
    Based on cross-encoder similarity and some more empirical rules
    """

    pairs = []

    for ind1, node1 in enumerate(nodes_list):
        cur_nodes_list = nodes_list[ind1 + 1 :]
        for ind2, node2 in zip(
            range(ind1 + 1, ind1 + 1 + len(cur_nodes_list)), cur_nodes_list
        ):
            tail_cond = compare_tails(dialogues, node1, node2)
            if tail_cond:
                pairs.append((1, (ind1, ind2)))

    groups = unite_pairs(pairs)
    grouped = []
    for el in groups:
        grouped += el
    singles = [[idx] for idx in range(len(nodes_list)) if idx not in grouped]
    groups += singles
    # print("INDEX: ", groups)
    groups = [[nodes_list[el] for el in g] for g in groups]

    return groups

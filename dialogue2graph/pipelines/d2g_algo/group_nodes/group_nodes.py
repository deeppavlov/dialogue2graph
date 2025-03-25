import copy
from dialogue2graph.pipelines.core.dialogue import Dialogue


def _get_cross(search, list2):

    sublist = [p for p in list2 if search in p]
    to_add = []
    for s in sublist:
        if s[0]!=search:
            to_add.append(s[0])
        else:
            to_add.append(s[1])
    return len(to_add)

def _unite_pairs(pairs: list[tuple[int,int]]):
    """Clustering nodes based on similar pairs
    From the start of this list look for nodes paired with first pair,
    and add them to groups
    """
    pairs_in = copy.deepcopy(pairs) 
    groups = []
    while pairs_in:
        cur = [p for p in pairs_in if p[0] in p or p[1] in p]
        x = cur[0]
        list1 = [p for p in cur if x[0] in p and x!=p]
        list2 = [p for p in cur if x[1] in p and x!=p]
        to_add = []
        for y in list1:
            to_add = []
            for el in x:
                if el == y[0]:
                    search = y[1]
                else:
                    search= y[0]
                if _get_cross(search,list2):
                    to_add += [search]

        # Next we unite them and remove, and remove again
        to_add = list(set(([x[0],x[1]]+to_add)))
        groups.append(to_add)
        pairs_in = [p for p in pairs_in if p[0] not in to_add and p[1] not in to_add]

    return groups

def _get_indices(lst, element):
    result = []
    offset = -1
    while True:
        try:
            offset = lst.index(element, offset+1)
        except ValueError:
            return result
        result.append(offset)

def _get_tails(dialogue: Dialogue, assistant: list, node: str):

        ids = _get_indices(assistant, node)
        tails = []
        for id in ids:
            tail = dialogue.messages[:id*2][-4:]
            tail += dialogue.messages[id*2+1:][:4]
            tails.append(tail)
        return tails

def _compare_tails(dialogues: list[Dialogue], node1: str, node2: str):
    tails1 = []
    tails2 = []
    for dialogue in dialogues:
        # user = [d for d in dialogue.messages if d.participant=='user']
        assistant = [d.text for d in dialogue.messages if d.participant=='assistant']
        # assistant_back = list(reversed(assistant))
        tails1.extend(_get_tails(dialogue, assistant, node1))
        tails2.extend(_get_tails(dialogue, assistant, node2))
    for t1 in tails1:
        for t2 in tails2:
            if t1 == t2:
                return True
    return False

def group_nodes(dialogues: list[Dialogue], nodes_list: list[str]) -> list[list[str]]:
    """ Rule based algorithm to group graph nodes based on context matching
    nodes_list: list of assistant's utterances
    """

    pairs = []

    for ind1, node1 in enumerate(nodes_list):
        cur_nodes_list = nodes_list[ind1+1:]
        for ind2, node2 in zip(range(ind1+1,ind1+1+len(cur_nodes_list)),cur_nodes_list):

            tail_cond = _compare_tails(dialogues, node1, node2)
            if tail_cond:
                pairs.append((ind1,ind2))

    groups = _unite_pairs(pairs)
    grouped = []
    for el in groups:
        grouped += el
    singles = [[idx] for idx in range(len(nodes_list)) if idx not in grouped]
    groups += singles
    groups = [[nodes_list[el] for el in g] for g in groups]

    return groups


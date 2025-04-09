import copy
from dialogue2graph.pipelines.core.dialogue import Dialogue


def _get_intersection(to_search: int, index_pairs: list[tuple[int, int]]) -> int:
    """Form list of indices from pairs of indices which have pair with to_search
    Args:
      to_search: index to search for
      index_pairs: list of pairs to search in
    Returns:
      Length of the list of indices from index_pairs which have pairs with to_search
    """
    found = [p for p in index_pairs if to_search in p]
    intersection = []
    for pair in found:
        if pair[0] != to_search:
            intersection.append(pair[0])
        else:
            intersection.append(pair[1])
    return len(intersection)


def _unite_pairs(index_pairs: list[tuple[int, int]]) -> list[tuple[int]]:
    """Clustering nodes based on similar pairs
    From the start of this list look for nodes paired with first pair,
    and add them to groups
    Args:
      index_pairs: list of indices paired based on similarity
    Returns:
      list of clusters where indices are united in groups based on index_pairs
    """
    pairs_left = copy.deepcopy(index_pairs)
    groups = []

    while pairs_left:
        first_pair = pairs_left[0]
        first_list = [p for p in pairs_left if first_pair[0] in p and first_pair != p]
        second_list = [p for p in pairs_left if first_pair[1] in p and first_pair != p]
        intersection = set()
        for pair in first_list:
            for el in first_pair:
                if el == pair[0]:
                    to_search = pair[1]
                else:
                    to_search = pair[0]
                if _get_intersection(to_search, second_list):
                    intersection.add(to_search)
        # Next we unite them and remove, and remove again
        intersection.update(first_pair)
        groups.append(list(intersection))
        pairs_left = [p for p in pairs_left if p[0] not in intersection and p[1] not in intersection]

    return groups


def _get_indices(role_list: list[str], utterance: str) -> list[int]:
    """Get list of dialog utterances of one role and returns indices
    of utterance in this list
    Args:
      role_list: list of utterances to search for utterance
      utterance: utterance to search for
    Returns: list of indices
    """
    result = []
    offset = -1
    while True:
        try:
            offset = role_list.index(utterance, offset + 1)
        except ValueError:
            return result
        result.append(offset)


def _get_tails(dialog: Dialogue, utt: str) -> list[list[dict[str,str]]]:
    """Get dialog messages 2 assistant utterances
    before and after utterance utt in dialog
    Args:
      dialog: dialog to look tails in
      utt: utterance to search tails by
    Returns:
      list of lists of dicts in a form {"participant": "user" or "assistant", "text": text}
    """
    assistant_list = [m.text for m in dialog.messages if m.participant == "assistant"]
    ids = _get_indices(assistant_list, utt)
    tails = []
    for id in ids:
        tail = dialog.messages[: id * 2][-4:]
        tail += dialog.messages[id * 2 + 1 :][:4]
        tails.append(tail)
    return tails


def _compare_tails(dialogs: list[Dialogue], utt1: str, utt2: str) -> bool:
    """ Compares tails of two utterances in a list of dialogues
    Args:
      dialogs: list of dialogs to get tails from
      utt1, utt2: utterances to get tails
    Returns:
      True when tails intersect, False otherwise
    """
    tails1 = []
    tails2 = []
    for dialog in dialogs:
        # user = [d for d in dialogue.messages if d.participant=='user']
        # assistant_back = list(reversed(assistant))
        tails1.extend(_get_tails(dialog, utt1))
        tails2.extend(_get_tails(dialog, utt2))
    for t1 in tails1:
        for t2 in tails2:
            if t1 == t2:
                return True
    return False


def group_nodes(dialogues: list[Dialogue], nodes_list: list[str]) -> list[list[str]]:
    """Rule based algorithm to group graph nodes based on context matching
    Args:
      dialogues: list of dialogues to form nodes from
      nodes_list: previously rendered list of assistant utterances
    Returns:
      list of assistant utterances by groups of nodes in a form
      [[group1_utterances], [[group2_utterances], ...]]
    """

    index_pairs = []
    for idx1, assist_utt1 in enumerate(nodes_list):
        after_assist1 = nodes_list[idx1 + 1 :]
        for idx2, assist_utt2 in zip(
            range(idx1 + 1, idx1 + 1 + len(after_assist1)), after_assist1
        ):
            if _compare_tails(dialogues, assist_utt1, assist_utt2):
                index_pairs.append((idx1, idx2))
    index_groups = _unite_pairs(index_pairs)
    grouped_indices = []
    for group in index_groups:
        grouped_indices += group
    singles = [[idx] for idx in range(len(nodes_list)) if idx not in grouped_indices]
    index_groups += singles
    groups = [[nodes_list[el] for el in group] for group in index_groups]

    return groups

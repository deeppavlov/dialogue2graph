from dialogue2graph.pipelines.core.dialogue import Dialogue


def _unite_pairs(index_pairs: list[tuple[int, int]]) -> list[tuple[int]]:
    """Clustering nodes based on similar pairs
    From the start of this list look for nodes paired with first pair,
    and add them to groups
    Args:
      index_pairs: list of indices paired based on similarity
    Returns:
      list of clusters where indices are united in groups based on index_pairs
    """
    groups = []
    while index_pairs:
        first_pair, *index_pairs = index_pairs
        group = {first_pair[0], first_pair[1]}
        for pair in index_pairs[:]:
            if any(el in group for el in pair):
                group.update(pair)
                index_pairs.remove(pair)
        groups.append(list(group))

    return groups


def _get_tails(dialog: Dialogue, utt: str) -> list[list[dict[str,str]]]:
    """Get dialog messages 2 assistant utterances
    before and after utterance utt in dialog
    Args:
      dialog: dialog to look tails in
      utt: utterance to search tails by
    Returns:
      list of lists of dicts in a form {"participant": "user" or "assistant", "text": text}
    """
    assistant_list = [(i, m.text) for i, m in enumerate(dialog.messages) if m.participant == "assistant"]
    ids = [i for i, text in assistant_list if text == utt]
    tails = []
    for id in ids:
        tail = dialog.messages[max(0, id-2):id]
        tail += dialog.messages[id+1:min(len(dialog.messages), id+3)]
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

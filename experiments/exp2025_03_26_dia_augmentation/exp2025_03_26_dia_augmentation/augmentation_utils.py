from sentence_transformers import SentenceTransformer
import numpy as np
from collections import Counter


def to_set(x):
    return set(x)


def check_no_duplicates_all_dialogues(dialogues):
    all_utterances = []
    for dia in dialogues:
        utterances = [uttr for turn in dia for uttr in turn["text"]]
        all_utterances.append(utterances)

    sets = map(to_set, all_utterances)

    all_common_elements = []
    for i, uttr_set_1 in enumerate(sets):
        for j, uttr_set_2 in enumerate(sets):
            if i != j:
                common_elements = uttr_set_1 & uttr_set_2
                if common_elements:
                    common_elements = [el for el in common_elements]
                    all_common_elements += common_elements

    if len(all_common_elements) == 0:
        return True
    else:
        print("common_elements:", set(all_common_elements))
        return False


def check_no_duplicates_one_dialogue(dialogue):
    utterances = [uttr for turn in dialogue for uttr in turn["text"]]

    if len(utterances) == len(set(utterances)):
        return True

    else:
        counter = Counter(utterances)
        common_elements = [k for k, v in counter.items() if v > 1]
        print("common_elements:", common_elements)
        return False


def check_no_duplicates_one_uttr_list(dialogue):
    utterances_lists = [turn["text"] for turn in dialogue]
    for utterances in utterances_lists:
        if len(utterances) == len(set(utterances)):
            return True

        else:
            counter = Counter(utterances)
            common_elements = [k for k, v in counter.items() if v > 1]
            print("common_elements:", common_elements.items())
            return False


def check_diagonal_similarity(dialogue_1, dialogue_2, embedder):
    utterances_1 = [uttr["text"] for uttr in dialogue_1]
    utterances_2 = [uttr["text"] for uttr in dialogue_2]

    model = SentenceTransformer(embedder)
    embeddings_1 = model.encode(utterances_1)
    embeddings_2 = model.encode(utterances_2)

    model.similarity_fn_name = "cosine"
    similarities = model.similarity(embeddings_1, embeddings_2)

    diagonal_elements = np.diag(similarities)
    non_diagonal_elements = similarities[
        np.where(~np.eye(len(similarities), dtype=bool))
    ]
    mean_diagonal_similarity = np.mean(diagonal_elements)

    for i in range(len(diagonal_elements)):
        if diagonal_elements[i] <= non_diagonal_elements.max():
            return (False, mean_diagonal_similarity)

    return (True, mean_diagonal_similarity)


def is_correct_length_modified(dialogue_1, dialogue_2):
    return len(dialogue_1) == len(dialogue_2)


def match_roles_modified(dialogue_1, dialogue_2):
    for phrase_1, phrase_2 in zip(dialogue_1, dialogue_2):
        if phrase_1["participant"] != phrase_2["participant"]:
            return False
    return True

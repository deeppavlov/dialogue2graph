grouping_prompt_1 = """Your input is a list of dialogues from customer chatbot system.
Your task is to genearate set of nodes for the dialogue graph corresponding to these dialogues.
Next is an example of the graph (set of rules) how chatbot system looks like - it is
a set of nodes with assistant's utterances and a set of edges that are
triggered by user requests: """

grouping_prompt_2 = """This is the end of the example.
**Rules:**
1) is_start field is mandatory for all the nodes.
2) is_start=True field in the node is an entry point to the whole graph.
3) Nodes must be assistant's utterances only.
4) All the nodes for the graph are created from the resulting groups in point 5) according to the rules 6,7,8,9
with exclusively assistant's utterances only in their original unmodified form.
5) The grouping process looks like follows:
a. Go over all the dialogues, take every assistant's utterance one by one.
b. Search for all assistant's utterances surrounded by utterances with same contextual meaning: one pair of assistant's and user's utterances previous to
current assistants' utterance and one pair of user's and assistant's utterances next to current assistants' utterance.
Search only for those assistant's utterances where both previous and next pairs are contextually similar to their counterparts.
c. Below is an example when two assistant's utterances have surrounding pairs with same contextual meaning:
'Please, enter the payment method you would like to use: cash or credit card.', and
'How would you prefer to pay?'
Surrounding pairs for both of them are similar so these two assistant's utterances have surrounding pairs with same contextual meaning.
d. Contexts with opposite intents go to different groups.
e. Assitant's utterances with different intents or distant meanings go to different groups.
f. When one assistant's utterance contains a reference to a previous context, which is not clear in the other assistant's utterance, those utterances go to different groups.
For example, one utterance refers to some objects called by such the words like there, that, etc, and the other utterance doesn't have such references.
g. Place utterances satisfying point 5b (taking into account exceptions in points 5d, 5e and 5f) into one group.
h. Go to next utterance in step 5a. Make sure you don't miss any utterance.
6) Don't use user's utterances for grouping process.
7) Every assistant's utterance not included in any group shall be present in its own group of single utterance.
8) You must doublecheck that all the assistant's utterances are present in resulting set of nodes, make sure not a single utterance is missed.
9) Always place adjacent assistant's utterances separated by only one user's utterance into different groups.
10) You must always return valid JSON fenced by a markdown code block. Do not return any additional text.
11) Add reason point to your answer with an explanation why some of the assistant's utterances were not included in the resulting groups.
I will give a list of dialogues, your task is to build a set of nodes for this list according to the rules and examples above.
List of dialogues: """

graph_example_1 = {
    "edges": [
        {"source": 1, "target": 2, "utterances": ["I need to make an order", "I want to order from you"]},
        {
            "source": 2,
            "target": 3,
            "utterances": [
                "I would like to purchase Pale Fire and Anna Karenina, please",
                "One War and Piece in hard cover and one Pride and Prejudice",
            ],
        },
        {"source": 3, "target": 4, "utterances": ["With credit card, please", "Cash"]},
        {"source": 4, "target": 2, "utterances": ["Start new order"]},
    ],
    "nodes": [
        {"id": 1, "label": "start", "is_start": True, "utterances": ["How can I help?", "Hello"]},
        {"id": 2, "label": "ask_books", "is_start": False, "utterances": ["What books do you like?"]},
        {
            "id": 3,
            "label": "ask_payment_method",
            "is_start": False,
            "utterances": ["Please, enter the payment method you would like to use: cash or credit card.", "How would you prefer to pay?"],
        },
        {
            "id": 4,
            "label": "ask_to_redo",
            "is_start": False,
            "utterances": ["Something is wrong, can you please use other payment method or start order again"],
        },
    ],
    "reason": "",
}

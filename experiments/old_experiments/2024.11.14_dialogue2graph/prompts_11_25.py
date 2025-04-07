from langchain.prompts import PromptTemplate

prompts = {}

create_graph_prompt = PromptTemplate.from_template(
    "You have an example of dialogue from customer chatbot system. You also have an "
    "example of set of rules how chatbot system works should be looking - it is "
    "a set of nodes when chatbot system respons and a set of transitions that are "
    "triggered by user requests. "
    "Here is the example of set of rules: "
    "'edges': [ [ 'source': 1, 'target': 2, 'utterances': 'I need to make an order' ], "
    "[ 'source': 1, 'target': 2, 'utterances': 'I want to order from you' ], "
    "[ 'source': 2, 'target': 3, 'utterances': 'I would like to purchase 'Pale Fire' and 'Anna Karenina', please' ], "
    "'nodes': [ [ 'id': 1, 'label': 'start', 'is_start': true, 'utterances': [ 'How can I help?', 'Hello' ], "
    "[ 'id': 2, 'label': 'ask_books', 'is_start': false, 'utterances': [ 'What books do you like?'] ] "
    "I will give a dialogue, your task is to build a graph for this dialogue in the format above. We allow several edges with equal "
    "source and target and also multiple responses on one node so try not to add new nodes if it is logical just to extend an "
    "exsiting one. utterances in one node or on multiedge should close between each other and correspond to different answers "
    "to one question or different ways to say something. For example, for question about preferences or a Yes/No question "
    "both answers can be fit in one multiedge, there’s no need to make a new node. If two nodes has the same responses they "
    "should be united in one node. Do not make up utterances that aren’t present in the dialogue. Please do not combine "
    "utterances for multiedges in one list, write them separately like in example above. Every utterance from the dialogue, "
    "whether it is from user or assistanst, should contain in one of the nodes. Edges must be utterances from the user. Do not forget ending nodes with goodbyes. "
    "Sometimes dialogue can correspond to several iterations of loop, for example: "
    "['text': 'Do you have apples?', 'participant': 'user'], "
    "['text': 'Yes, add it to your cart?', 'participant': 'assistant'], "
    "['text': 'No', 'participant': 'user'], "
    "['text': 'Okay. Anything else?', 'participant': 'assistant'], "
    "['text': 'I need a pack of chips', 'participant': 'user'], "
    "['text': 'Yes, add it to your cart?', 'participant': 'assistant'], "
    "['text': 'Yes', 'participant': 'user'], "
    "['text': 'Done. Anything else?', 'participant': 'assistant'], "
    "['text': 'No, that’s all', 'participant': 'user'], "
    "it corresponds to following graph: "
    "[ nodes: "
    "'id': 1, "
    "'label': 'confirm_availability_and_ask_to_add', "
    "'is_start': false, "
    "'utterances': 'Yes, add it to your cart?' "
    "], "
    "[ "
    "'id': 2, "
    "'label': 'reply_to_yes', "
    "'is_start': false, "
    "'utterances': ['Done. Anything else?', 'Okay. Anything else?'] "
    "], "
    "[ "
    "'id': 3, "
    "'label': 'finish_filling_cart', "
    "'is_start': false, "
    "'utterances': 'Okay, everything is done, you can go to cart and finish the order.' "
    "], "
    "edges: "
    "[ "
    "'source': 1, "
    "'target': 2, "
    "'utterances': 'Yes' "
    "], "
    "[ "
    "'source': 1, "
    "'target': 2, "
    "'utterances': 'No' "
    "], "
    "[ "
    "'source': 2, "
    "'target': 1, "
    "'utterances': 'I need a pack of chips' "
    "], "
    "[ "
    "'source': 2, "
    "'target': 3, "
    "'utterances': 'No, that’s all' "
    "]. "
    "We encourage you to use cycles and complex interwining structure of the graph with 'Yes'/'No' edges for the branching."
    "This is the end of the example. Brackets must be changed back into curly braces to create a valid JSON string. Return ONLY JSON string in plain text (no code blocks) without any additional commentaries."
    "Dialogue: {dialog}"
)

# prompts["general_graph_generation_prompt"] = PromptTemplate.from_template(
#     "You have an example of a dialogue from customer chatbot system. You also have an "
#     "example of the graph (set of rules) how chatbot system looks like) - it is "
#     "a set of nodes with chatbot system utterances and a set of transitions that are "
#     "triggered by user requests. "
#     "Here is the graph example: "
#     "'edges': [ [ 'source': 1, 'target': 2, 'utterances': ['I need to make an order'] ], "
#     "[ 'source': 1, 'target': 2, 'utterances': ['I want to order from you'] ], "
#     "[ 'source': 2, 'target': 3, 'utterances': ['I would like to purchase 'Pale Fire' and 'Anna Karenina', please'] ], "
#     "'nodes': [ [ 'id': 1, 'label': 'start', 'is_start': true, 'utterances': [ 'How can I help?', 'Hello' ] ], "
#     "[ 'id': 2, 'label': 'ask_books', 'is_start': false, 'utterances': [ 'What books do you like?'] ], "
#     "[ 'id': 3, 'label': 'ask_payment_method', 'is_start': false, 'utterances': [ 'Please, enter the payment method you would like to use: cash or credit card.'] ]"
#     "I will give a dialogue, your task is to build a graph for this dialogue in the format above. We allow several edges with same "
#     "source and target and also multiple responses in one node so try not to add new nodes if it is logical just to extend an "
#     "exsiting one. Utterances in one node or in multiedge should be close to each other and correspond to different answers "
#     "to one question or different ways to say something. For example, for question about preferences or a Yes/No question "
#     "both answers can go in one multiedge, there’s no need to make a new node. If two nodes have the same response they "
#     "should be united in one node. Do not make up utterances that aren’t present in the dialogue. Please do not combine "
#     "utterances from multiedges in one list, write them separately like in example above. Every utterance from the dialogue, "
#     "whether it is from user or assistanst, shall be present in the graph. Nodes must be assistant's utterances, edges must be utterances from the user. Do not forget ending nodes with goodbyes. "
#     "Sometimes dialogue can correspond to several iterations of loop, for example: "
#     """
#         [
#             [
#                 "text": "How can I help?",
#                 "participant": "assistant"
#             ],
#             [
#                 "text": "I need to make an order",
#                 "participant": "user"
#             ],
#             [
#                 "text": "Which books would you like to order?",
#                 "participant": "assistant"
#             ],
#             [
#                 "text": "One War and Piece in hard cover and one Pride and Prejudice",
#                 "participant": "user"
#             ],
#             [
#                 "text": "Please, enter the payment method you would like to use: cash or credit card.",
#                 "participant": "assistant"
#             ],
#             [
#                 "text": "With credit card, please",
#                 "participant": "user"
#             ],
#             [
#                 "text": "Something is wrong, can you please use other payment method or start order again",
#                 "participant": "assistant"
#             ],
#             [
#                 "text": "I will enter new payment method",
#                 "participant": "user"
#             ],
#             [
#                 "text": "Please, enter the payment method you would like to use: cash or credit card.",
#                 "participant": "assistant"
#             ],
#             [
#                 "text": "Start new order",
#                 "participant": "user"
#             ],
#             [
#                 "text": "Which books would you like to order?",
#                 "participant": "assistant"
#             ]
#         ]
#     """
#     "It shall result in the graph below:"
#     """
#     "edges": [
#                 [
#                     "utterances": [
#                         "I need to make an order",
#                         "I want to order from you"
#                     ],
#                     "source": 1,
#                     "target": 2
#                 ],
#                 [
#                     "utterances": [
#                         "I would like to purchase 'Pale Fire' and 'Anna Karenina', please",
#                         "One War and Piece in hard cover and one Pride and Prejudice"
#                     ],
#                     "source": 2,
#                     "target": 3
#                 ],
#                 [
#                     "utterances": [
#                         "Cash",
#                         "With credit card, please"
#                     ],
#                     "source": 3,
#                     "target": 4
#                 ],
#                 [
#                     "utterances": [
#                         "I will enter new payment method"
#                     ],
#                     "source": 4,
#                     "target": 3
#                 ],
#                 [
#                     "utterances": [
#                         "Start new order"
#                     ],
#                     "source": 4,
#                     "target": 2
#                 ]
#             ],
#             "nodes": [
#                 [
#                     "id": 1,
#                     "label": "start",
#                     "is_start": true,
#                     "utterances": [
#                         "How can I help?",
#                         "Hello"
#                     ]
#                 ],
#                 [
#                     "id": 2,
#                     "label": "ask_item",
#                     "is_start": false,
#                     "utterances": [
#                         "Which books would you like to order?"
#                     ]
#                 ],
#                 [
#                     "id": 3,
#                     "label": "ask_payment_method",
#                     "is_start": false,
#                     "utterances": [
#                         "Please, enter the payment method you would like to use: cash or credit card."
#                     ]
#                 ],
#                 [
#                     "id": 4,
#                     "label": "ask_to_redo",
#                     "is_start": false,
#                     "utterances": [
#                         "Something is wrong, can you please use other payment method or start order again"
#                     ]
#                 ]
#             ]
#     """
#     "This is the end of the example."
#     "Brackets must be changed back into curly braces to create a valid JSON string. Return ONLY JSON string in plain text (no code blocks) without any additional commentaries."
#     "Dialogue: {dialog}"
# )


graph_example_1 = {
    "edges": [
        #        {'source': 1, 'target': 2, 'utterances': ['I need to make an order']},
        {"source": 1, "target": 2, "utterances": ["I want to order from you"]},
        {
            "source": 2,
            "target": 3,
            "utterances": [
                "I would like to purchase Pale Fire and Anna Karenina, please"
            ],
        },
        {"source": 3, "target": 4, "utterances": ["With credit card, please"]},
        {"source": 4, "target": 2, "utterances": ["Start new order"]},
    ],
    "nodes": [
        {
            "id": 1,
            "label": "start",
            "is_start": True,
            "utterances": ["How can I help?", "Hello"],
        },
        {
            "id": 2,
            "label": "ask_books",
            "is_start": False,
            "utterances": ["What books do you like?"],
        },
        {
            "id": 3,
            "label": "ask_payment_method",
            "is_start": False,
            "utterances": [
                "Please, enter the payment method you would like to use: cash or credit card."
            ],
        },
        {
            "id": 4,
            "label": "ask_to_redo",
            "is_start": False,
            "utterances": [
                "Something is wrong, can you please use other payment method or start order again"
            ],
        },
    ],
    "reason": "",
}

dialogue_example_2 = [
    {"text": "How can I help?", "participant": "assistant"},
    {"text": "I need to make an order", "participant": "user"},
    {"text": "Which books would you like to order?", "participant": "assistant"},
    {
        "text": "One War and Piece in hard cover and one Pride and Prejudice",
        "participant": "user",
    },
    {
        "text": "Please, enter the payment method you would like to use: cash or credit card.",
        "participant": "assistant",
    },
    {"text": "With credit card, please", "participant": "user"},
    {
        "text": "Something is wrong, can you please use other payment method or start order again",
        "participant": "assistant",
    },
    {"text": "I will enter new payment method", "participant": "user"},
    # {
    #     "text": "Please, enter the payment method you would like to use: cash or credit card.",
    #     "participant": "assistant"
    # }
    # {
    #     "text": "Which books would you like to order?",
    #     "participant": "assistant"
    # }
]

graph_example_2 = {
    "edges": [
        {
            "source": 1,
            "target": 2,
            "utterances": [
                "I need to make an order"
                #                        "I want to order from you"
            ],
        },
        {
            "source": 2,
            "target": 3,
            "utterances": [
                #                        "I would like to purchase 'Pale Fire' and 'Anna Karenina', please",
                "One War and Piece in hard cover and one Pride and Prejudice"
            ],
        },
        {
            "source": 3,
            "target": 4,
            "utterances": [
                # "Cash",
                "With credit card, please"
            ],
        },
        {"source": 4, "target": 3, "utterances": ["I will enter new payment method"]},
    ],
    "nodes": [
        {
            "id": 1,
            "label": "start",
            "is_start": True,
            "utterances": ["How can I help?"],
        },
        {
            "id": 2,
            "label": "ask_item",
            "is_start": False,
            "utterances": ["Which books would you like to order?"],
        },
        {
            "id": 3,
            "label": "ask_payment_method",
            "is_start": False,
            "utterances": [
                "Please, enter the payment method you would like to use: cash or credit card."
            ],
        },
        {
            "id": 4,
            "label": "ask_to_redo",
            "is_start": False,
            "utterances": [
                "Something is wrong, can you please use other payment method or start order again"
            ],
        },
    ],
}


# dialogue_example_2 = [
#             {
#                 "text": "How can I help?",
#                 "participant": "assistant"
#             },
#             {
#                 "text": "I need to make an order",
#                 "participant": "user"
#             },
#             {
#                 "text": "Which books would you like to order?",
#                 "participant": "assistant"
#             },
#             {
#                 "text": "One War and Piece in hard cover and one Pride and Prejudice",
#                 "participant": "user"
#             },
#             {
#                 "text": "Please, enter the payment method you would like to use: cash or credit card.",
#                 "participant": "assistant"
#             },
#             {
#                 "text": "With credit card, please",
#                 "participant": "user"
#             },
#             {
#                 "text": "Something is wrong, can you please use other payment method or start order again",
#                 "participant": "assistant"
#             },
#             {
#                 "text": "I will enter new payment method",
#                 "participant": "user"
#             },
#             {
#                 "text": "Please, enter the payment method you would like to use: cash or credit card.",
#                 "participant": "assistant"
#             },
#             {
#                 "text": "Start new order",
#                 "participant": "user"
#             }
#             # {
#             #     "text": "Which books would you like to order?",
#             #     "participant": "assistant"
#             # }
#         ]

# graph_example_2 = {
#     "edges": [
#                 {
#                     "source": 1,
#                     "target": 2,
#                     "utterances": [
#                         "I need to make an order"
# #                        "I want to order from you"
#                     ]
#                 },
#                 {
#                     "source": 2,
#                     "target": 3,
#                     "utterances": [
# #                        "I would like to purchase 'Pale Fire' and 'Anna Karenina', please",
#                         "One War and Piece in hard cover and one Pride and Prejudice"
#                     ]
#                 },
#                 {
#                     "source": 3,
#                     "target": 4,
#                     "utterances": [
#                         # "Cash",
#                         "With credit card, please"
#                     ]
#                 },
#                 {
#                     "source": 4,
#                     "target": 3,
#                     "utterances": [
#                         "I will enter new payment method"
#                     ]
#                 },
#                 {
#                     "source": 4,
#                     "target": 2,
#                     "utterances": [
#                         "Start new order"
#                     ]
#                 }
#             ],
#             "nodes": [
#                 {
#                     "id": 1,
#                     "label": "start",
#                     "is_start": True,
#                     "utterances": [
#                         "How can I help?"
#                     ]
#                 },
#                 {
#                     "id": 2,
#                     "label": "ask_item",
#                     "is_start": False,
#                     "utterances": [
#                         "Which books would you like to order?"
#                     ]
#                 },
#                 {
#                     "id": 3,
#                     "label": "ask_payment_method",
#                     "is_start": False,
#                     "utterances": [
#                         "Please, enter the payment method you would like to use: cash or credit card."
#                     ]
#                 },
#                 {
#                     "id": 4,
#                     "label": "ask_to_redo",
#                     "is_start": False,
#                     "utterances": [
#                         "Something is wrong, can you please use other payment method or start order again"
#                     ]
#                 }
#             ]
# }

prompts["general_graph_generation_prompt"] = PromptTemplate.from_template(
    "You have an example of a dialogue from customer chatbot system. You also have an "
    "example of the graph (set of rules) how chatbot system looks like) - it is "
    "a set of nodes with chatbot system utterances and a set of transitions that are "
    "triggered by user requests. "
    "Here is the graph example: {graph_example_1}"
    "I will give a dialogue, your task is to build a cyclic graph for this dialogue in the format above. "
    # "We allow several edges with same "
    # "source and target and also multiple responses in one node so try not to add new nodes if it is logical just to extend an "
    # "exsiting one. Utterances in one node or in multiedge should be close to each other and correspond to different answers "
    # "to one question or different ways to say something. For example, for question about preferences or a Yes/No question "
    # "both answers can go in one multiedge, there’s no need to make a new node. "
    # "If two nodes have the same response they "
    # "should be united in one node. "
    "Do not make up utterances that aren’t present in the dialogue. "
    # "Please do not combine "
    # "utterances from multiedges in one list, write them separately like in example above. "
    "Every utterance from the dialogue, "
    "whether it is from user or assistanst, shall be present in the graph. Nodes must be assistant's utterances, edges must be utterances from the user. "
    # "It is impossible to have an edge connecting to non-existent node. "
    "Never create nodes with same utterance. "
    # "Do not forget ending nodes with goodbyes. "
    # "Every dialogue corresponds to a cycled graph, for example: {dialogue_example_2}. "
    # "It shall result in the graph below: {graph_example_2}. "
    # "This is the end of the example. "
    "Cyclic graph means you don't duplicate nodes, but connect new edge to one of previously created nodes instead. "
    "When you go to next user's utterance, first try to answer to that utterance with utterance from one of previously created nodes. "
    "If you see it is possible not to create new node with same or similar utterance, but instead create next edge connecting back to that node, then it is place for a cycle here. "
    # "Never create nodes with user's utterances. "
    # "Don't repeat assistance's utterance from one of existing nodes, just cycle to previously created node with that utterance. "
    "IMPORTANT: All assistant's utterances are nodes, but all user's utterances are edges. "
    "All the dialogues you've prompted are cyclic. "
    "Before answering you must check where the dialogue can cycle and make the first node of a cycle a target node for the last node of the cycle. "
    # "Return ONLY JSON string in plain text (no code blocks) without any additional commentaries. "
    "You must always return valid JSON fenced by a markdown code block. Do not return any additional text. "
    "Here goes the dialogue, build a cyclic graph according to the instructions above. "
    "Dialogue: {dialog}"
)

prompts["general_graph_generation_prompt_try_2"] = PromptTemplate.from_template(
    "You have an example of a dialogue from customer chatbot system. You also have an "
    "example of the graph (set of rules) how chatbot system looks like) - it is "
    "a set of nodes with chatbot system utterances and a set of transitions that are "
    "triggered by user requests. "
    "Here is the graph example: {graph_example_1}"
    "I will give a dialogue, your task is to build a graph for this dialogue in the format above. "
    # "We allow several edges with same "
    # "source and target and also multiple responses in one node so try not to add new nodes if it is logical just to extend an "
    # "exsiting one. Utterances in one node or in multiedge should be close to each other and correspond to different answers "
    # "to one question or different ways to say something. For example, for question about preferences or a Yes/No question "
    # "both answers can go in one multiedge, there’s no need to make a new node. "
    "If two nodes have the same response they "
    "should be united in one node."
    "Do not make up utterances that aren’t present in the dialogue. "
    # "Please do not combine "
    # "utterances from multiedges in one list, write them separately like in example above. "
    "Every utterance from the dialogue, "
    "whether it is from user or assistanst, shall be present in the graph. "
    # "Nodes must be assistant's utterances, edges must be utterances from the user. "
    # "Do not forget ending nodes with goodbyes. "
    "Dialogue can correspond to several iterations of loop, for example: {dialogue_example_2}. "
    "It shall result in the graph below: {graph_example_2}. "
    "This is the end of the example. "
    "Graph must be cyclic - no dead ends. "
    # "A cycle is a closed path in a graph, which means that it starts and ends at the same vertex and passes through a sequence of distinct vertices and edges."
    "When you go to next user's utterance, first try to answer to that utterance with utterance from one of previously created nodes. "
    "If you see it is possible not to create new node with same or similar utterance, but instead create next edge connecting back to that node, then it is place for a cycle here. "
    "When you create new edge connecting to new node, you must create this node. "
    # "Never create nodes with user's utterances. "
    # "Don't repeat assistance's utterance from one of existing nodes, just cycle to previously created node with that utterance. "
    "IMPORTANT: "
    "All assistant's utterances are nodes, but all user's utterances are edges. "
    "All the dialogues you've prompted are cyclic. "
    "Before answering you must check where the dialogue can loop or cycle and make the first node of a cycle a target node for the last node of the cycle. "
    "Return ONLY JSON string in plain text (no code blocks) without any additional commentaries. "
    "Dialogue: {dialog}"
)


# prompts["general_graph_generation_prompt"] = PromptTemplate.from_template(
#     "You have an example of a dialogue from customer chatbot system. You also have an "
#     "example of the graph (set of rules) how chatbot system looks like - it is "
#     "a set of nodes with chatbot system utterances and a set of transitions that are "
#     "triggered by user requests. "
#     # "Every transition in all edges shall start from and result in nodes which are present in set of nodes."
#     "Here is the graph example: {graph_example_1}"
#     "I will give a dialogue, your task is to build a graph for this dialogue in the format above. "
#     # "We allow several edges with same "
#     # "source and target and also multiple responses in one node so try not to add new nodes if it is logical just to extend an "
#     # "exsiting one. Utterances in one node or in multiedge should be close to each other and correspond to different answers "
#     # "to one question or different ways to say something. For example, for question about preferences or a Yes/No question "
#     # "both answers can go in one multiedge, there’s no need to make a new node. "
#     #"If two nodes have the same response they should be united in one node. "
#     "Do not make up utterances that aren’t present in the dialogue. "
#     # "Please do not combine "
#     # "utterances from multiedges in one list, write them separately like in example above. "
#     "Every utterance from the dialogue, "
#     "whether it is from user or assistanst, shall be present in the graph. "
#     "Nodes must be assistant's utterances, edges must be utterances from the user. "
#     # "Do not forget ending nodes with goodbyes. "
#     "Dialogue may to contain cycles, for example: {dialogue_example_2}"
#     "It shall result in the graph below: {graph_example_2}"
#     "This is the end of the example."
#     #"A cycle is a closed path in a graph, which means that it starts and ends at the same vertex and passes through a sequence of distinct vertices and edges."
#     "When you look at next user's phrase, first try to answer to that phrase with utterance from one of previously created nodes. "
#     "If you see it is possible not to create new node with same or similar utterance, but create next edge leading back to that node, then it is place for a cycle here. "
#     "Don't repeat previously assistance's utterances from one of previous nodes, just cycle to existing one with that utterance. "
#     # "Use cycle in a graph when you see that assistant's next answer logically is a phrase which already was used earlier in dialogue, and make this node the first node of this cycle. "
#     # "So you don't create new node but create edge leading to existing node - fisrt node of the cycle. "
#     #"This repeat means that you don't create new node, but use node you created before for this assistant's utterance. "
#     "This user's phrase shall generate that edge leading to the node, it will be the edge connecting cycle's last node to the cycle's start node. "
#     "And you see it in a dialogue: Next user's phrase is: Start new order, "
#     "and you see that logical answer to this is: Which books would you like to order? "
#     "And node with that utterance already exists, so don't create new node, just cycle next edge to that existing node. "
#     "IMPORTANT: All the dialogues you've prompted are cyclic. "
#     # "Before answering you must check where the dialogue can loop or cycle and make the first node of a cycle a target node for the last node of the cycle. "
#     # "All the dialogues start from assistant's utterance, so first node of any loop cannot be first node of the whole graph. "

# #    "Return ONLY JSON string in plain text (no code blocks) without any additional commentaries."
#     "You must always return valid JSON fenced by a markdown code block. Do not return any additional text."
#     "Dialogue: {dialog}"
# )


prompts["specific_graph_generation_prompt"] = PromptTemplate.from_template(
    "You have a dialogue from customer chatbot system as input. "
    "Your task is to create a cyclic dialogue graph corresponding to that dialogue."
    "Next is an example of the graph (set of rules) how chatbot system looks like - it is "
    "a set of nodes with chatbot system utterances and a set of transitions that are "
    "triggered by user requests: {graph_example_1} "
    # "Every transition in all edges shall start from and result in nodes which are present in set of nodes."
    "Another dialogue example: {dialogue_example_2}"
    "It shall result in the graph below: {graph_example_2}"
    "This is the end of the example."
    # "A cycle is a closed path in a graph, which means that it starts and ends at the same vertex and passes through a sequence of distinct vertices and edges."
    # "Use cycle in a graph when you see that assistant repeats phrase which already was used earlier in dialogue, and make this repeat the first node of this cycle. "
    # "This repeat means that you don't create new node, but use node you created before for this assistant's utterance. "
    # "User's utterance in a dialogue before this repeating utterance will be an edge leading from cycle's last node to the cycle's start node (the node with the repeating assistant's utterance). "
    "**Rules:**"
    "1) Nodes must be assistant's utterances, edges must be utterances from the user. "
    "2) Every utterance from the dialogue, "
    "whether it is from user or assistanst, shall be present in the graph. "
    "3) Do not make up utterances that aren’t present in the dialogue. "
    # "3) The final node MUST connect back to an existing node. "
    "4) Graph must be cyclic - no dead ends. "
    "5) When you go to next user's utterance, first try to find answer relevant to that utterance from one of previously created nodes. "
    "If it is found then create next edge connecting back to the node with the right answer. "
    # "The cycle shall be organized so that you don't duplicate either user's utterances or nodes with same utterances. "
    # # "4) Assistance's responses must acknowledge what the user has already specified. "
    "6) Exceeding the number of nodes over the number of assistant's utterances is prohibited. "
    "7) Exceeding the number of edges over the number of user's utterances is prohibited.  "
    # "8) It is prohibited to duplicate edges with same user's utterances. "
    # "9) It is prohibited to duplicate nodes with same assistant's utterances. "
    # "8) The nodes are duplicated if there are at least two nodes with same utterances. "
    "8) After the graph is created, it is necessary to check whether utterances in the nodes are duplicated. "
    "If they are, it is necessary to remove the node duplicating utterances from preceding ones and connect the edges "
    "that led to this deleted node with the original node. "
    "Also remove all the edges emanating from deleted nodes. "
    # "9) Next check if there are extra nodes (exceeding number of assistance's utterances), "
    # "then find duplicates and repeat procedure from step 8. "
    # "13) Next check if there are extra edges (exceeding number of user's utterances), "
    # "then find node duplicates and repeat procedure from step 11. "
    # "5) All edges must connect to existing nodes"
    "9) You must always return valid JSON fenced by a markdown code block. Do not return any additional text. "
    # "7) Responses must acknowledge what the user has already specified. "
    # "6) The cycle point should make logical sense. "
    "I will give a dialogue, your task is to build a graph for this dialogue according to the rules and examples above. "
    # "Do not make up utterances that aren’t present in the dialogue. "
    # "Please do not combine "
    # "utterances from multiedges in one list, write them separately like in example above. "
    # "Do not forget ending nodes with goodbyes. "
    # "IMPORTANT: All the dialogues you've prompted are cyclic so the conversation MUST return to an existing node"
    # "When you go to next user's utterance, first try to answer to that utterance with utterance from one of previously created nodes. "
    # "If you see it is possible not to create new node with same or similar utterance, but instead create next edge connecting back to that node, then it is place for a cycle here. "
    # "The cycle shall be organized so that you don't duplicate either user's utterances or nodes with same utterances. "
    # "Before answering you must check where the dialogue can loop or cycle and make the first node of a cycle a target node for the last node of the cycle. "
    # "All the dialogues start from assistant's utterance, so first node of any loop cannot be first node of the whole graph. "
    #    "Return ONLY JSON string in plain text (no code blocks) without any additional commentaries."
    "Dialogue: {dialog}"
)

prompts["second_graph_generation_prompt"] = PromptTemplate.from_template(
    "Your input is a dialogue from customer chatbot system. "
    "Your task is to create a cyclic dialogue graph corresponding to the dialogue. "
    "Next is an example of the graph (set of rules) how chatbot system looks like - it is "
    "a set of nodes with chatbot system utterances and a set of edges that are "
    "triggered by user requests: {graph_example_1} "
    "This is the end of the example."
    "**Rules:**"
    "1) Nodes must be assistant's utterances, edges must be utterances from the user. "
    "2) When you go to next user's utterance, first try to find answer relevant to that utterance from one of previously created nodes. "
    "If it is found then create next edge connecting back to the node with the right answer. Don't create more nodes after that step. "
    "3) Every assistance's utterance from the dialogue shall be present in one and only one node of a graph. "
    "4) Every user's utterance from the dialogue shall be present in one and only one edge of a graph. "
    "5) Use ony utterances from the dialogue. It is prohibited to create new utterances different from input ones. "
    "6) Never create nodes with user's utterances. "
    "7) Graph must be cyclic - no dead ends. "
    "8) Number of nodes must be equal to the number of assistant's utterances. "
    "9) Number of edges must be equal to the number of user's utterances. "
    "10) You must always return valid JSON fenced by a markdown code block. Do not return any additional text. "
    "I will give a dialogue, your task is to build a graph for this dialogue according to the rules and examples above. "
    "Dialogue: {dialog}"
)


prompts["third_graph_generation_prompt"] = PromptTemplate.from_template(
    "Your input is a dialogue from customer chatbot system. "
    "Your task is to create a cyclic dialogue graph corresponding to the dialogue. "
    "Next is an example of the graph (set of rules) how chatbot system looks like - it is "
    "a set of nodes with chatbot system utterances and a set of edges that are "
    "triggered by user requests: {graph_example_1} "
    "This is the end of the example."
    "**Rules:**"
    "1) Nodes must be assistant's utterances, edges must be utterances from the user. "
    "2) Every assistance's utterance from the dialogue shall be present in one and only one node of a graph. "
    "3) Every user's utterance from the dialogue shall be present in one and only one edge of a graph. "
    "4) Use ony utterances from the dialogue. It is prohibited to create new utterances different from input ones. "
    "6) Never create nodes with user's utterances. "
    "7) Graph must be cyclic - no dead ends. "
    "8) The number of nodes shall be equal to number of assistant's phrases. "
    "9) You must always return valid JSON fenced by a markdown code block. Do not return any additional text. "
    "I will give a dialogue, your task is to build a graph for this dialogue according to the rules and examples above. "
    "Dialogue: {dialog}"
)

# user's follow-up concern loops back to the problem elaboration stage, maintaining a logical and continuous support flo
# "Cycle starts at the 'elaborate_problem' node (id:2) because it acknowledges and addresses new user concerns when the user mentions another issue."
# "The cycle starts at the 'ask_membership_type' node as it represents the problem elaboration stage. This allows the user's intent to register another member to loop back to selecting a membership type, ensuring a continuous and logical registration flow."

prompts["fourth_graph_generation_prompt"] = PromptTemplate.from_template(
    "Your input is a dialogue from customer chatbot system. "
    "Your task is to create a cyclic dialogue graph corresponding to the dialogue. "
    "Next is an example of the graph (set of rules) how chatbot system looks like - it is "
    "a set of nodes with chatbot system utterances and a set of edges that are "
    "triggered by user requests: {graph_example_1} "
    "This is the end of the example."
    "Note that is_start field in the node is an entry point to the whole graph, not to the cycle. "
    # "Every transition in all edges shall start from and result in nodes which are present in set of nodes."
    # "Another dialogue example: {dialogue_example_2}"
    # "It shall result in the graph below: {graph_example_2}"
    # "This is the end of the example."
    # "A cycle is a closed path in a graph, which means that it starts and ends at the same vertex and passes through a sequence of distinct vertices and edges."
    # "Use cycle in a graph when you see that assistant repeats phrase which already was used earlier in dialogue, and make this repeat the first node of this cycle. "
    # "This repeat means that you don't create new node, but use node you created before for this assistant's utterance. "
    # "User's utterance in a dialogue before this repeating utterance will be an edge leading from cycle's last node to the cycle's start node (the node with the repeating assistant's utterance). "
    "**Rules:**"
    "1) Nodes must be assistant's utterances, edges must be utterances from the user. "
    "2) Every assistance's utterance from the dialogue shall be present in one and only one node of a graph. "
    "3) Every user's utterance from the dialogue shall be present in one and only one edge of a graph. "
    "4) Use ony utterances from the dialogue. It is prohibited to create new utterances different from input ones. "
    "6) Never create nodes with user's utterances. "
    # "3) The final node MUST connect back to an existing node. "
    "7) Graph must be cyclic - no dead ends. "
    "8) The cycle point should make logical sense. "
    "9) The starting node of the cycle cannot be the beginning of a conversation with the user. "
    "It must be a continuation of the user's previous phrase, kind of problem elaboration stage. "
    "Typically it is clarifying question to previous users' phrase for example. "
    "So cycle start cannot be greeting (first) node of the whole graph, it shall be another one node. "
    # "9) When you go to next user's utterance, first try to answer to that utterance with utterance from one of previously created nodes. "
    # "9) Choose the start of the cycle so that user's follow-up concern loops back to the problem elaboration stage, maintaining a logical and continuous support flow. "
    # "It shall not look like assistants"
    # "9) For the start of the cycle choose the node where assistant's answer will show user that information from their request (in looping back edge) is understood and taken into account. "
    # "If you see it is possible not to create new node with same or similar utterance, but instead create next edge connecting back to that node, then it is place for a cycle here. "
    # "For the start of the cycle choose such a node where the assistant's answer will be based on information from that edge. "
    # "10) Use one of assistant's utterances from the dialogue for the cycle point, don't add/create more nodes with same or simiar utterances. "
    # "Number of nodes is the number of unique node ID's. "
    # "Remember that the number of nodes cannot exceed the number of assistant's phrases. "
    # "When you add node with same utterances it duplicates nodes and increases number of nodes. So when this situation takes place, just combine such two duplicates into one node. "
    # "10) It is prohibited to duplicate nodes with same assistant's utterances. "
    # "11) Duplicated nodes are "
    "10) Number of nodes and edges cannot exceed number of utterances in a dialogue. "
    # "9) Categorical imperative: The number of nodes must be equal to number of assistant's phrases. "
    # "10) Categorical imperative: The number of edges must be equal to the number of user's utterances. "
    # "Don't create more nodes, use existing ones. "
    # "The cycle shall be organized so that you don't duplicate either user's utterances or nodes with same utterances. "
    # # "4) Assistance's responses must acknowledge what the user has already specified. "
    # "8) Number of nodes must be equal to the number of assistant's utterances. "
    # "9) Number of edges must be equal to the number of user's utterances. "
    # "7) Exceeding the number of edges over the number of user's utterances is prohibited.  "
    # "8) The nodes are duplicated if there are at least two nodes with same utterances. "
    # "6) After the graph is created, it is necessary to check whether utterances in the nodes are duplicated. "
    # "If they are, it is necessary to remove duplicating nodes and connect the edges "
    # "that led to the deleted nodes with the original ones. "
    # "Also remove all the edges emanating from deleted nodes. "
    # "8) It is prohibited to duplicate edges with same user's utterances. "
    # "10) After the graph is created, it is necessary to check if there are extra nodes (exceeding number of assistance's utterances), "
    # "then find duplicates and if they exist, it is necessary to remove duplicating nodes and connect the edges "
    # "that led to the deleted nodes with the original ones. "
    # "Also remove all the edges emanating from deleted nodes. "
    # "13) Next check if there are extra edges (exceeding number of user's utterances), "
    # "then find node duplicates and repeat procedure from step 11. "
    # "5) All edges must connect to existing nodes"
    "11) You must always return valid JSON fenced by a markdown code block. Do not return any additional text. "
    "12) Add reason point to the graph with explanation how cycle start point has been chosen. "
    # "12) Add reason point to the graph where put the result of 6+6. "
    # "7) Responses must acknowledge what the user has already specified. "
    # "6) The cycle point should make logical sense. "
    "I will give a dialogue, your task is to build a graph for this dialogue according to the rules and examples above. "
    # "Do not make up utterances that aren’t present in the dialogue. "
    # "Please do not combine "
    # "utterances from multiedges in one list, write them separately like in example above. "
    # "Do not forget ending nodes with goodbyes. "
    # "IMPORTANT: All the dialogues you've prompted are cyclic so the conversation MUST return to an existing node"
    # "When you go to next user's utterance, first try to answer to that utterance with utterance from one of previously created nodes. "
    # "If you see it is possible not to create new node with same or similar utterance, but instead create next edge connecting back to that node, then it is place for a cycle here. "
    # "The cycle shall be organized so that you don't duplicate either user's utterances or nodes with same utterances. "
    # "Before answering you must check where the dialogue can loop or cycle and make the first node of a cycle a target node for the last node of the cycle. "
    # "All the dialogues start from assistant's utterance, so first node of any loop cannot be first node of the whole graph. "
    #    "Return ONLY JSON string in plain text (no code blocks) without any additional commentaries."
    "Dialogue: {dialog}"
)

result_form = {"result": True, "reason": ""}

compare_graphs_prompt = PromptTemplate.from_template(
    "You will get two dialogue graphs in following format: {graph_example_1}. "
    "Graphs are equivalent when they have the same number of nodes connected in the same way, meaning there is one-to-one correspondence "
    "between their nodes which preserves adjacency. "
    "Equal nodes or edges may have different utterances when utterances have same intents, logics and similar meaning. "
    "Equivalent graphs are equal when corresponding nodes from both graphs are equal, "
    "and corresponding edges from both graphs are equal. Labels do not matter. "
    "In your answer return True if graphs are equal and False otherwise. "
    "In a field marked by reason explain your answer. "
    "Form of the answer is {result_form} ."
    # "Output your response in the demanded json format. "
    "You must always return only valid JSON fenced by a markdown code block. Do not return any additional text. "
    "Next are graph1: {graph_1} and graph2: {graph_2}"
)

check_graph_utterances_prompt = PromptTemplate.from_template(
    "You have a dialogue and a structure of graph built on this dialogue it is a "
    "set of nodes when chatbot system responses and a set of transitions that are triggered by user requests.\n"
    "Please say if for every utterance in the dialogue there exist either a utteranse in node or in some edge. "
    "be attentive to what nodes we actually have in the 'nodes' list, because some nodes from the list of edges maybe non existent:\n"
    "Graph: {graph}\n"
    "Dialogue: {dialog}\n"
    "just print the list of utteance and whether there exsit a valid edge of node contating it, if contains print the node or edge"
)

check_graph_validity_prompt = PromptTemplate.from_template(
    "1. You have an example of dialogue from customer chatbot system.\n"
    "2. You also have a set of rules how chatbot system works - a set of "
    "nodes when chatbot system respons and a set of transitions that are triggered by user requests.\n"
    "3. Chatbot system can move only along transitions listed in 2.  If a transition from node A to "
    "node B is not listed we cannot move along it.\n"
    "4. If a dialog doesn't contradcit with the rules listed in 2 print YES otherwise if such dialog "
    "could'nt happen because it contradicts the rules print NO.\nDialogue: {dialog}.\nSet of rules: {rules}"
)

cycle_graph_generation_prompt = PromptTemplate.from_template(
    "You have an example of dialogue from customer chatbot system. You also have an "
    "example of set of rules how chatbot system works should be looking - it is "
    "a set of nodes when chatbot system respons and a set of transitions that are "
    "triggered by user requests. "
    "Here is the example of set of rules: "
    "'edges': [ [ 'source': 1, 'target': 2, 'utterances': 'I need to make an order' ], "
    "[ 'source': 1, 'target': 2, 'utterances': 'I want to order from you' ], "
    "[ 'source': 2, 'target': 3, 'utterances': 'I would like to purchase 'Pale Fire' and 'Anna Karenina', please' ], "
    "'nodes': [ [ 'id': 1, 'label': 'start', 'is_start': true, 'utterances': [ 'How can I help?', 'Hello' ], "
    "[ 'id': 2, 'label': 'ask_books', 'is_start': false, 'utterances': [ 'What books do you like?'] ] "
    "I will give a dialogue, your task is to build a graph for this dialogue in the format above. We allow several edges with equal "
    "source and target and also multiple responses on one node so try not to add new nodes if it is logical just to extend an "
    "exsiting one. utterances in one node or on multiedge should close between each other and correspond to different answers "
    "to one question or different ways to say something. "
    "If two nodes has the same responses they "
    "should be united in one node. Do not make up utterances that aren’t present in the dialogue. Please do not combine "
    "utterances for multiedges in one list, write them separately like in example above. Every utterance from the dialogue, "
    "whether it is from user or assistanst, should contain in one of the nodes. Edges must be utterances from the user. Do not forget ending nodes with goodbyes. "
    "Sometimes dialogue can correspond to several iterations of loop, for example: "
    "['text': 'Do you have apples?', 'participant': 'user'], "
    "['text': 'Yes, add it to your cart?', 'participant': 'assistant'], "
    "['text': 'No', 'participant': 'user'], "
    "['text': 'Okay. Anything else?', 'participant': 'assistant'], "
    "['text': 'I need a pack of chips', 'participant': 'user'], "
    "['text': 'Yes, add it to your cart?', 'participant': 'assistant'], "
    "['text': 'Yes', 'participant': 'user'], "
    "['text': 'Done. Anything else?', 'participant': 'assistant'], "
    "['text': 'No, that’s all', 'participant': 'user'], "
    "it corresponds to following graph: "
    "[ nodes: "
    "'id': 1, "
    "'label': 'confirm_availability_and_ask_to_add', "
    "'is_start': false, "
    "'utterances': 'Yes, add it to your cart?' "
    "], "
    "[ "
    "'id': 2, "
    "'label': 'reply_to_yes', "
    "'is_start': false, "
    "'utterances': ['Done. Anything else?', 'Okay. Anything else?'] "
    "], "
    "[ "
    "'id': 3, "
    "'label': 'finish_filling_cart', "
    "'is_start': false, "
    "'utterances': 'Okay, everything is done, you can go to cart and finish the order.' "
    "], "
    "edges: "
    "[ "
    "'source': 1, "
    "'target': 2, "
    "'utterances': 'Yes' "
    "], "
    "[ "
    "'source': 1, "
    "'target': 2, "
    "'utterances': 'No' "
    "], "
    "[ "
    "'source': 2, "
    "'target': 1, "
    "'utterances': 'I need a pack of chips' "
    "], "
    "[ "
    "'source': 2, "
    "'target': 2, "
    "'utterances': 'No, that’s all' "
    "]. "
    "Another example:"
    """
        [
            [
                "text": "How can I help?",
                "participant": "assistant"
            ],
            [
                "text": "I need to make an order",
                "participant": "user"
            ],
            [
                "text": "Which books would you like to order?",
                "participant": "assistant"
            ],
            [
                "text": "One War and Piece in hard cover and one Pride and Prejudice",
                "participant": "user"
            ],
            [
                "text": "Please, enter the payment method you would like to use: cash or credit card.",
                "participant": "assistant"
            ],
            [
                "text": "With credit card, please",
                "participant": "user"
            ],
            [
                "text": "Something is wrong, can you please use other payment method or start order again",
                "participant": "assistant"
            ],
            [
                "text": "I will enter new payment method",
                "participant": "user"
            ]
        ]
    """
    "Should result in graph like this (note that even in the case of negative result 'something is wrong' it must be cycled):"
    """
    "edges": [
                [
                    "utterances": [
                        "I need to make an order",
                        "I want to order from you"
                    ],
                    "source": 1,
                    "target": 2
                ],
                [
                    "utterances": [
                        "I would like to purchase 'Pale Fire' and 'Anna Karenina', please",
                        "One War and Piece in hard cover and one Pride and Prejudice"
                    ],
                    "source": 2,
                    "target": 3
                ],
                [
                    "utterances": [
                        "Cash",
                        "With credit card, please"
                    ],
                    "source": 3,
                    "target": 4
                ],
                [
                    "utterances": [
                        "I will enter new payment method"
                    ],
                    "source": 4,
                    "target": 3
                ],
                [
                    "utterances": [
                        "Start new order"
                    ],
                    "source": 4,
                    "target": 1
                ]
            ],
            "nodes": [
                [
                    "id": 1,
                    "label": "start",
                    "is_start": true,
                    "utterances": [
                        "How can I help?",
                        "Hello"
                    ]
                ],
                [
                    "id": 2,
                    "label": "ask_item",
                    "is_start": false,
                    "utterances": [
                        "Which books would you like to order?"
                    ]
                ],
                [
                    "id": 3,
                    "label": "ask_payment_method",
                    "is_start": false,
                    "utterances": [
                        "Please, enter the payment method you would like to use: cash or credit card."
                    ]
                ],
                [
                    "id": 4,
                    "label": "ask_to_redo",
                    "is_start": false,
                    "utterances": [
                        "Something is wrong, can you please use other payment method or start order again"
                    ]
                ]
            ]
    """
    "This is the end of the example."
    "IMPORTANT: all the dialogues you've prompted are cyclic. Before answering you must check where the dialog can loop or cycle and make the first node of a cycle a target node for the last node of the cycle. Brackets must be changed back into curly braces to create a valid JSON string. Return ONLY JSON string in plain text (no code blocks) without any additional commentaries."
    "Dialogue: {dialog}"
)

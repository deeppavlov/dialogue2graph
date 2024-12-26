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

# graph_example_1 = {
#     "edges": [
# #        {'source': 1, 'target': 2, 'utterances': ['I need to make an order']},
#         {'source': 1, 'target': 2, 'utterances': ['I want to order from you', "I need to make an order"]},
#         {'source': 2, 'target': 3, 'utterances': ['I would like to purchase Pale Fire and Anna Karenina, please', "One War and Piece in hard cover and one Pride and Prejudice"]},
#         {"source": 3, "target": 4, "utterances": ["With credit card, please", "Cash"]},
#         {"source": 4, "target": 2, "utterances": ["Start new order"]}
#     ],
#     'nodes':
#       [
#           {'id': 1, 'label': 'start', 'is_start': True, 'utterances': [ 'How can I help?', 'Hello']},
#           {'id': 2, 'label': 'ask_books', 'is_start': False, 'utterances': [ 'What books do you like?']},
#           {'id': 3, 'label': 'ask_payment_method', 'is_start': False, 'utterances': [ 'Please, enter the payment method you would like to use: cash or credit card.']},
#           {"id": 4, "label": "ask_to_redo", "is_start": False, "utterances": [ "Something is wrong, can you please use other payment method or start order again"]}
#       ],
#       'reason': ""
# }

graph_example_1 = {
    "edges": [
        {'source': 1, 'target': 2, 'utterances': ['I need to make an order', 'I want to order from you']},
        {'source': 2, 'target': 3, 'utterances': ['I would like to purchase Pale Fire and Anna Karenina, please', 'One War and Piece in hard cover and one Pride and Prejudice']},
        {"source": 3, "target": 4, "utterances": ["With credit card, please", "Cash"]},
        {"source": 4, "target": 2, "utterances": ["Start new order"]}
    ],
    'nodes':
      [
          {'id': 1, 'label': 'start', 'is_start': True, 'utterances': [ 'How can I help?', 'Hello']},
          {'id': 2, 'label': 'ask_books', 'is_start': False, 'utterances': [ 'What books do you like?']},
          {'id': 3, 'label': 'ask_payment_method', 'is_start': False, 'utterances': [ 'Please, enter the payment method you would like to use: cash or credit card.']},
          {"id": 4, "label": "ask_to_redo", "is_start": False, "utterances": [ "Something is wrong, can you please use other payment method or start order again"]}
      ],
      'reason': ""
}


dialogue_example_2 = [
            {
                "text": "How can I help?",
                "participant": "assistant"
            },
            {
                "text": "I need to make an order",
                "participant": "user"
            },
            {
                "text": "Which books would you like to order?",
                "participant": "assistant"
            },
            {
                "text": "One War and Piece in hard cover and one Pride and Prejudice",
                "participant": "user"
            },
            {
                "text": "Please, enter the payment method you would like to use: cash or credit card.",
                "participant": "assistant"
            },
            {
                "text": "With credit card, please",
                "participant": "user"
            },
            {
                "text": "Something is wrong, can you please use other payment method or start order again",
                "participant": "assistant"
            },
            {
                "text": "I will enter new payment method",
                "participant": "user"
            }
        ]



graph_example_2 = {
    "edges": [
                {
                    "source": 1,
                    "target": 2,
                    "utterances": [
                        "I need to make an order"
#                        "I want to order from you"
                    ]
                },
                {
                    "source": 2,
                    "target": 3,
                    "utterances": [
#                        "I would like to purchase 'Pale Fire' and 'Anna Karenina', please",
                        "One War and Piece in hard cover and one Pride and Prejudice"
                    ]
                },
                {
                    "source": 3,
                    "target": 4,
                    "utterances": [
                        # "Cash",
                        "With credit card, please"
                    ]
                },
                {
                    "source": 4,
                    "target": 3,
                    "utterances": [
                        "I will enter new payment method"
                    ]
                }
            ],
            "nodes": [
                {
                    "id": 1,
                    "label": "start",
                    "is_start": True,
                    "utterances": [
                        "How can I help?"
                    ]
                },
                {
                    "id": 2,
                    "label": "ask_item",
                    "is_start": False,
                    "utterances": [
                        "Which books would you like to order?"
                    ]
                },
                {
                    "id": 3,
                    "label": "ask_payment_method",
                    "is_start": False,
                    "utterances": [
                        "Please, enter the payment method you would like to use: cash or credit card."
                    ]
                },
                {
                    "id": 4,
                    "label": "ask_to_redo",
                    "is_start": False,
                    "utterances": [
                        "Something is wrong, can you please use other payment method or start order again"
                    ]
                }
            ]
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
    #"Never create nodes with user's utterances. "
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
    #"Never create nodes with user's utterances. "
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
    #"A cycle is a closed path in a graph, which means that it starts and ends at the same vertex and passes through a sequence of distinct vertices and edges."
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

#user's follow-up concern loops back to the problem elaboration stage, maintaining a logical and continuous support flo
#"Cycle starts at the 'elaborate_problem' node (id:2) because it acknowledges and addresses new user concerns when the user mentions another issue."
#"The cycle starts at the 'ask_membership_type' node as it represents the problem elaboration stage. This allows the user's intent to register another member to loop back to selecting a membership type, ensuring a continuous and logical registration flow."

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
    #"A cycle is a closed path in a graph, which means that it starts and ends at the same vertex and passes through a sequence of distinct vertices and edges."
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



prompts["options_graph_generation_prompt"] = PromptTemplate.from_template(
    "Your input is a list of dialogues from customer chatbot system. "
    "Your task is to create a cyclic dialogue graph corresponding to these dialogues. "
    "Next is an example of the graph (set of rules) how chatbot system looks like - it is "
    "a set of nodes with chatbot system utterances and a set of edges that are "
    "triggered by user requests: {graph_example_1} "
    "This is the end of the example."
    "Note that is_start field in the node is an entry point to the whole graph, not to the cycle. "

    "**Rules:**"
    "1) Nodes must be assistant's utterances, edges must be utterances from the user. "
    "2) Every assistance's utterance from the dialogue shall be present in one and only one node of a graph. " 
    "3) Every user's utterance from the dialogue shall be present in one and only one edge of a graph. "    
    "4) Use ony utterances from the dialogue. It is prohibited to create new utterances different from input ones. "
    "6) Never create nodes with user's utterances. "
    "8) Graph must be cyclic - shall contain cycle(s). "
    "9) Usially graph has branches, and different dialogues can present different branches. "
    "9) The cycle point(s) should make logical sense. "
    "10) The starting node of the cycle cannot be the beginning of a conversation with the user. "
    "It must be a continuation of the user's previous phrase, kind of problem elaboration stage. "
    "Typically it is clarifying question to previous users' phrase for example. "
    "So cycle start cannot be greeting (first) node of the whole graph, it shall be another one node. "
    "11) Number of nodes and edges cannot exceed number of utterances in a dialogue. "
    "12) You must always return valid JSON fenced by a markdown code block. Do not return any additional text. "
    "13) Add reason point to the graph with explanation how cycle start points have been chosen. "
    "I will give a list of dialogues, your task is to build a graph for this list according to the rules and examples above. "
    "List of dialogues: {dialog}"
)

# 9) Number of nodes is always equal to the amount of unique assistant's utterances in all the dialogues.
# However, close utterances according to point 4 above must remain in one node and thus constitute unique
# sets of utterances. Then the number of such sets must correspond to the number of nodes.
#or are different ways to answer to same or similar user's utterance,
# 6) Group of utterances from single node as indicated in point 5 above constitute unique
# set of utterances. Then the number of such sets must correspond to the number of nodes.
part_1 = """Your input is a list of dialogues from customer chatbot system.
Your task is to genearate set of nodes for the dialogue graph corresponding to these dialogues.
Next is an example of the graph (set of rules) how chatbot system looks like - it is
a set of nodes with assistant's utterances and a set of edges that are
triggered by user requests: """
# part_2 = """This is the end of the example.
# Note that is_start field in the node is an entry point to the whole graph, not to the cycle.
# **Rules:**
# 1) Nodes must be assistant's utterances, edges must be utterances from the user.
# 2) Every assistance's utterance from the dialogue shall be present in one and only one node of a graph.
# 3) Every user's utterance from the dialogue shall be present in at least one edge of a graph.
# 4) Never create nodes with user's utterances.
# 5) We allow multiple utterances in one node.
# So if two or more assistance's utterances from same or different dialogues are interchangeable
# and lead to interchangeable transitions, they must be grouped into single node for sure.
# Don't miss any assistant's utterance in any of the groups.
# 6) We allow multiple utterances in one edge.
# So if two or more user's utterances from same or different dialogues are interchangeable
# and follow interchangeable assistance's utterances, they must be grouped into one edge for sure.
# Don't miss any user's utterance in any of the groups.
# 7) Group of utterances from single node as indicated at point 6 above constitute unique
# set of utterances. Then the number of such sets must correspond to the number of nodes.
# 8) Usually graph has branches, and different dialogues can present different branches.
# 9) Cyclic graph means you connect new edge to one of previously created nodes.
# When you go to next user's utterance, first try to answer to that utterance with utterance from one of previously created nodes.
# If you see it is possible not to create new node with same or similar utterance,
# but instead create next edge connecting back to one of previous nodes, then it is place for a cycle here.
# 10) The starting node of a cycle cannot be the entry point to the whole graph.
# It means that starting node typically does not have label "start" where is_start is True.
# Instead it must be a continuation of the user's previous phrase, kind of problem elaboration stage.
# Typically it is clarifying question to previous user's phrase.
# So cycle start cannot be greeting (first) node of the whole graph, it shall be another one node.
# 11) Resulting graph shall not loop any node back to same node.
# 12) You must always return valid JSON fenced by a markdown code block. Do not return any additional text.
# 13) Add reason point to the graph with explanation why you didn't combine similar utterances in one edge or node.
# I will give a list of dialogues, your task is to build a graph for this list according to the rules and examples above.
# List of dialogues: """

# 7) Group of utterances from single node as indicated at point 5 above constitute unique
# set of utterances. Then the number of such sets must correspond to the number of nodes.
#The number of these groups must correspond to the number of nodes.

# Find all user's utterances from all the dialogues so, that:
# their following nodes have interchangeable utterances,
# their preceding nodes have interchangeable utterances,
# and group them.
#  Keep in mind that adjacent utterances from assistant cannot be included in the same group.

# part_2 = """This is the end of the example.
# Note that is_start field in the node is an entry point to the whole graph, not to the cycle.
# **Rules:**
# 1) Nodes must be assistant's utterances, edges must be utterances from the user.
# 2) Every assistance's utterance from the dialogue shall be present in one and only one node of a graph.
# 3) Every user's utterance from the dialogue shall be present in at least one edge of a graph.
# 4) Never create nodes with user's utterances.
# 5) We allow multiple utterances in one node.
# Group all the assistant's utterances from all the dialogues so, that:
# For any pair of items in a group the following is true:
# two sets of user's utterances immediately following the item in a pair (every entrance of the item builds pair) have at least one common utterance.  
# two sets of user's utterances immediately preceding the item in a pair (every entrance of the item builds pair) have at least one common utterance.
# At the same time, the interchangeability of the two utterances ensures commonality.
# When both items in a pair don't have following utterances, they shall be grouped together based on immediately preceding utterances.
# When both items in a pair don't have preceding utterances, they shall be grouped together based on immediately following utterances.
# Every assitant's utterance must be part of one of the groups for sure.
# Form nodes from the resulting groups.
# Remove duplicates inside any of the groups.
# 6) We allow multiple utterances in one edge.
# Group all the user's utterances from all the dialogues so, that:
# For any pair of items in a group the following is true:
# two sets of assistant's utterances immediately following the item in a pair (every entrance of the item builds pair) have at least one common utterance.
# two sets of assistant's utterances immediately preceding the item in a pair (every entrance of the item builds pair) have at least one common utterance.
# At the same time, the interchangeability of the two utterances ensures commonality.
# When both items in a pair don't have following utterances, they shall be grouped together based on immediately preceding utterances.
# When both items in a pair don't have preceding utterances, they shall be grouped together based on immediately following utterances.
# Every user's utterance must be part of one of the groups for sure.
# Form edges from the resulting groups. Remove duplicates inside any of the groups.
# 7) Usually graph has branches, and different dialogues can present different branches.
# 8) Cyclic graph means you connect new edge to one of previously created nodes.
# When you go to next user's utterance, first try to answer to that utterance with utterance from one of previously created nodes.
# If you see it is possible not to create new node with same or similar utterance,
# but instead create next edge connecting back to one of previous nodes, then it is place for a cycle here.
# 9) The starting node of a cycle cannot be the entry point to the whole graph.
# It means that starting node typically does not have label "start" where is_start is True.
# Instead it must be a continuation of the user's previous phrase, kind of problem elaboration stage.
# Typically it is clarifying question to previous user's phrase.
# So cycle start cannot be greeting (first) node of the whole graph, it shall be another one node.
# 10) Resulting graph shall not loop any node back to same node.
# 11) You must always return valid JSON fenced by a markdown code block. Do not return any additional text.
# 12) Add reason point to the graph with explanation why you didn't combine similar utterances in one edge or node.
# I will give a list of dialogues, your task is to build a graph for this list according to the rules and examples above.
# List of dialogues: """

# part_2 = """This is the end of the example.
# Note that is_start field in the node is an entry point to the whole graph, not to the cycle.
# **Rules:**
# 1) Nodes must be assistant's utterances, edges must be utterances from the user.
# 2) Every assistance's utterance from the dialogue shall be present in its original unmodified form in one and only one node of a graph.
# 3) Every user's utterance from the dialogue shall be present in at least one edge of a graph.
# 4) Never create nodes with user's utterances.
# 5) We allow more than one assistant's utterances in one node.
# Group all assistant's utterances in all dialogues according to the following scheme:
# a. Go through all assistant's utterances.
# b. For each assistant's utterance go through all its occurrences in all dialogues.
# c. For each assistant's utterance make up the first set of user's utterances immediately following its occurrences
# and the second set of user's utterances immediately preceding its occurrences.
# d. Two assistant's utterances are included in one group when the both intersections of the first two sets and the second two sets are not empty.
# For intersection, interchangeable user's utterances are considered to be the same. Opposite version of same utterance fits as well.
# e. If both first sets are empty, only the intersection of the second sets is considered.
# f. If both second sets are empty, only the intersection of the first sets is considered.
# g. Every unique assitant's utterance must be part of one of the groups for sure.
# h. Duplicates inside any of the groups must be removed.
# Form nodes from the resulting groups. The number of groups must correspond to the number of nodes.
# 6) We allow more than one user's utterances in one edge.
# Group all user's utterances in all dialogues according to the following scheme:
# a. Go through all user's utterances.
# b. For each user's utterance go through all its occurrences in all dialogues.
# c. For each user's utterance make up the first set of assistant's utterances immediately following its occurrences
# and the second set of assistant's utterances immediately preceding its occurrences.
# d. Two user's utterances are included in one group when the both intersections of the first two sets and the second two sets are not empty.
# For intersection, interchangeable assistant's utterances are considered to be the same.
# e. If both first sets are empty, only the intersection of the second sets is considered.
# f. If both second sets are empty, only the intersection of the first sets is considered.
# g. Every user's utterance must be part of one of the groups for sure.
# h. Duplicates inside any of the groups must be removed.
# Form edges from the resulting groups. The number of groups must correspond to the number of edges.
# 7) Usually graph has branches, and different dialogues can present different branches.
# 8) Cyclic graph means you connect new edge to one of previously created nodes.
# When you go to next user's utterance, first try to answer to that utterance with utterance from one of previously created nodes.
# If you see it is possible not to create new node with same or similar utterance,
# but instead create next edge connecting back to one of previous nodes, then it is place for a cycle here.
# 9) The starting node of a cycle cannot be the entry point to the whole graph.
# It means that starting node typically does not have label "start" where is_start is True.
# Instead it must be a continuation of the user's previous phrase, kind of problem elaboration stage.
# Typically it is clarifying question to previous user's phrase.
# So cycle start cannot be greeting (first) node of the whole graph, it shall be another one node.
# 10) Resulting graph shall not loop any node back to same node.
# 11) You must always return valid JSON fenced by a markdown code block. Do not return any additional text.
# 12) Add reason point to the graph with explanation why you didn't combine similar utterances in one edge or node.
# I will give a list of dialogues, your task is to build a graph for this list according to the rules and examples above.
# List of dialogues: """
# 7) Usually graph has branches, and different dialogues can present different branches.
# For the second set, user's utterances with opposite intent fit as well.
# 2) Every assistance's utterance from the dialogue shall be present in its original unmodified form in one and only one node of a graph.
# 3) Every user's utterance from the dialogue shall be present in at least one edge of a graph.
# 4) Never create nodes with user's utterances.
#b. For each assistant's utterance go through all its occurrences in all dialogues.
#For each occurence of every assistant's utterance
# 2) Group all assistant's utterances in all dialogues according to the following scheme:
# a. Go through all unique assistant's utterances.
# b. Make up the first set of unique user's utterances immediately following current assistant's utterance
# and the second set of unique user's utterances immediately preceding current assistant's utterance.
# d. Two assistant's utterances are included in one group then and only then when two following conditions are met:
# intersection of the two first sets is not empty,
# intersection of the two second sets is not empty.
# For intersection, user's utterances with same intent are considered to be the same as well.
# For the first set, user's utterances with opposite intent are considered to be the same as well.
# If both first sets are empty, only the intersection of the second sets is considered.
# If both second sets are empty, only the intersection of the first sets is considered.
# e. Of the three types of utterances:
# with a question mark at the end,
# with an excalamation mark at the end,
# affirmative without exclamation mark,
# each group can contain only one.
# f. If two utterances from assistant are separated with just one user's utterance in any of the dialogues, they must be in different groups.
# g. Every unique assistant's utterance must be part of only one group for sure.
# h. Duplicates inside any of the groups must be removed.
# i. All the nodes for the graph must be created from the resulting groups exclusively with assistant's utterances only.
# j. The number of groups must correspond to the number of nodes.

# 3) Group all user's utterances in all dialogues according to the following scheme:
# a. Go through all unique user's utterances.
# b. Make up the first set of unique assistant's utterances immediately following current user's utterance
# and the second set of unique assistant's utterances immediately preceding current user's utterance.
# c. Two user's utterances with same intent are included in one group then and only then when two following conditions are met:
# intersection of the two first sets is not empty,
# intersection of the two second sets is not empty.
# For intersection here, assistant's utterances are considered equal when exact match or with the exactly same intent, and only these.
# If both first sets are empty, only the intersection of the second sets is considered.
# If both second sets are empty, only the intersection of the first sets is considered.
# d. Two user's utterances with opposite intent must be in different groups.
# e. Of the three types of utterances:
# with a question mark at the end,
# with an excalamation mark at the end,
# affirmative without exclamation mark,
# each group can contain only one.
# f. Every unique user's utterance must be part of only one group for sure.
# g. Duplicates inside any of the groups must be removed.
# h. All the edges for the graph must be created from the resulting groups exclusively with user's utterances only.
# i. The number of groups must correspond to the number of edges.

# 4) Group all user's utterances in all dialogues according to the the following scheme:
# a. User's utterances outgoing from one node belong to one group.
# b. Then split user's utterances with opposite intents into two different groups.
# c. All the edges for the graph must be created from the resulting groups with exclusively user's utterances only.

# d. Of the three types of utterances:
# with a question mark at the end,
# with an excalamation mark at the end,
# affirmative without exclamation mark,
# each group can contain only one.
# e. If two utterances from assistant are separated with just one user's utterance in any of the dialogues, they must be in different groups.
# f. Every unique assistant's utterance must be part of only one group for sure.
# g. Duplicates inside any of the groups must be removed.

# не придумывай новые фразы пользователя
#Each user utterance is part of an edge that comes out of the node with the previous assistant utterance and enters the node with the next assistant utterance.

# a. Go through all nodes.
# b. Go through all utterances in node (call it "source node").
# c. Go through all occurences of the utterance in assistant's utterances in all the dialogues.
# d. Take user's utterance following current occurance of assistant's utterance.
# e. Find node with assistant's utterance following this user's utterance. Will call this node "target node".
# f. Create an edge with this user's utterance. Source for this edge will be source node, and target is target node.
# g. If edge with these source and target already exists,
# just add new user's utterance to this edge if different from utterances existing in the edge. 
# h. If user's utterance is last utterance in a dialogue, go to point 6) and create a cycling edge.
# i. Go to next occurence in point c.
# j. Go to next utterance in point b.
# k. Go to next node in point a.
# a. If in any of the dialogues between two utterances from assistant is just one user's utterance and nothing else,
# these utterances must be in different groups.
# i. Two assistant's utterances with different or opposite intents must be in different groups.
# j. Two assistant's utterances with different either subjects, topics or differing details must be in different groups.
# 6) Nodes with user's utterances must be removed.

# For example, different drinks, or different rooms, etc, these are different details and make utterances containing them 
# separated in different groups.
# g. two utteranceshave have same essential details,
# o. When one assistant's utterance contains some important detail that is missed in the other assistant's utterance,
# these two assistant's utterances must be in different groups.

part_2 = """This is the end of the example.
**Rules:**
1) is_start field in the node is an entry point to the whole graph.
2) Nodes must be assistant's utterances only.
3) All the nodes for the graph are created from the resulting groups in point 5) with exclusively assistant's utterances only.
4) Don't use user's utterances for grouping process in point 5).
5) Take all assistant's utterances in all dialogues, each in one copy, and group them according to the following scheme:
a. Go through all assistant's utterances, take every utterance just in one copy.
b. Make up the set of user's utterances immediately following current assistant's utterance from point 5a.
c. Adjacent assistant's utterances are those which have just one utterance between them and nothing more in at least one of the dialogues.
d. Two assistant's utterances with same intent are included in one group when conditions below are met:
e. two utterances are not adjacent,
f. two utterances have similar meanings,
g. and one of next two conditions are met as well:
h. Either intersection of their two sets is not empty,
i. Or both sets are empty.
j. For intersection in point 5h only, user's utterances with same or similar intents
and user's utterances with opposite intents are considered to be the same.
k. Adjacent assistant's utterances are directly connected to each other by the utterance between them
so they cannot be in one node, meaning adjacent assistant's utterances must be in different groups strictly.
l. Two assistant's utterances with different intents must be in different groups.
m. When one assistant's utterance is the negation of another, these two assistant's utterances must be in different groups.
n. Two assistant's utterances with different general meanings must be in different groups.
o. Don't miss any assistant's utterance in all the dialogues.
p. Of the three types of assistant's utterances:
with a question mark at the end,
with an exclamation mark at the end,
affirmative without exclamation mark,
each group can contain only one.
q. Duplicates inside any of the groups must be removed.
r. Adjacent assistant's utterances must be in different groups.
6) It is forbidden to create nodes from user's utterances.
7) There should not be any duplicated groups.
8) Duplicates in the resulting nodes must be removed.
9) You must always return valid JSON fenced by a markdown code block. Do not return any additional text.
10) Add reason point to the graph with your explanation why you didn't combine utterances in one node according to the rules.
I will give a list of dialogues, your task is to build a set of nodes for this list according to the rules and examples above.
List of dialogues: """

auto_nodes = {
    "nodes": [
                            {
                                "id": 1,
                                "label": "start",
                                "is_start": True,
                                "utterances": [
                                    "Welcome to AutoCare Service Center! How can I help you today?"
                                ]
                            },
                            {
                                "id": 2,
                                "label": "ask_car_info",
                                "is_start": False,
                                "utterances": [
                                    "Could you please provide your car's make, model and year?"
                                ]
                            },
                            {
                                "id": 3,
                                "label": "ask_service_type",
                                "is_start": False,
                                "utterances": [
                                    "What type of service do you need? We offer diagnostics, repairs, and routine maintenance."
                                ]
                            },
                            {
                                "id": 4,
                                "label": "check_time_slots",
                                "is_start": False,
                                "utterances": [
                                    "Let me check our available time slots for this service."
                                ]
                            },
                            {
                                "id": 5,
                                "label": "offer_urgent_slot",
                                "is_start": False,
                                "utterances": [
                                    "I notice this is a critical repair. We have an urgent slot available today at 4 PM with a 20% rush fee. Would you like this option, or prefer a regular appointment next week?"
                                ]
                            },
                            {
                                "id": 6,
                                "label": "offer_regular_slots",
                                "is_start": False,
                                "utterances": [
                                    "We have slots available tomorrow at 10 AM or Friday at 2 PM. Which would you prefer?"
                                ]
                            },
                            {
                                "id": 7,
                                "label": "schedule_appointment",
                                "is_start": False,
                                "utterances": [
                                    "I've scheduled your appointment. Would you like to add any additional services while your car is here?"
                                ]
                            },
                            {
                                "id": 8,
                                "label": "oil_recommendation",
                                "is_start": False,
                                "utterances": [
                                    "While your car is here, we recommend an oil change and filter replacement. Would you like to add these services?"
                                ]
                            },
                            {
                                "id": 9,
                                "label": "ask_proceed_appointment",
                                "is_start": False,
                                "utterances": [
                                    "The estimated cost for all services will be $X. Would you like to proceed with the appointment?"
                                ]
                            },
                            {
                                "id": 10,
                                "label": "confirm_appointment",
                                "is_start": False,
                                "utterances": [
                                    "Your appointment is confirmed. We'll see you then. Have a great day!"
                                ]
                            }
                        ]
}

# Your task is to create a set of edges for dialogue graph corresponding to these dialogues.

edges_1 = """Your input is a list of dialogues from customer chatbot system.
Next is an example of the graph (set of rules) how chatbot system looks like - it is
a set of nodes with assistant's utterances and a set of edges that are
triggered by user requests: """

edges_2 = """This is the end of the example. Next is set of nodes for the target graph: """

# There can't be other edges or made up utterances in edges.
# 3) Don't create edges with non existing or modified utterances.
# Each user utterance is part of an edge that comes out of the node with the previous assistant utterance and enters the node with the next assistant utterance.
# h. If user's utterance is last utterance in a dialogue, try to find an answer to the utterance with utterance from one of nodes.
# Then create next edge connecting back to that node.

# k. Don't miss any user's utterance in all the dialogues.
# l. Don't create edges with non existing or modified utterances.
# Don't try to modify edges based on your own uderstanding. Follow the instructions strictly.
#Don't use grouping of utterances based on intents. Don't use intents at all. Use letter by letter comparison of utterances only.
#  Don't modify source and target nodes, use them strictly as in dialogues.
#Don't modifiy existing utterances. Use only original utterances.
# Don't make up new utterances.
# Follow these instructions strictly.



# Go through all the specific occurencies of user's utterances in all the dialogues.
# In given nodes part find node with assistant's utterance immediately preceding current ocurrence. This node will be source. 
# In given nodes part find node with assistant's utterance immediately following current occurence. This node will be target.
# Create edge with current user's utterance, source and target.
# If edge with these source and target already exists,
# just add current user's utterance to this edge if it is different from utterances existing in the edge.
# Don't modify edge with these source and target which already has this utterance.
#3) Resulting graph shall not loop any node back to same node.




# Let's call a pair of user's utterance and previous assistant's utterance an occurence.
# User's utterance of each occurence in dialogues is part of an edge that comes out of the node with the previous assistant utterance
# and enters the node with the next assistant utterance.
# If occurence doesn't have following assistant's utterances in all dialogues, you must find a continuation of 
# this occurence with assistant's utterance from one of nodes and connect new edge with this user's utterance
# to that node. This node is kind of problem elaboration stage. Typically it is clarifying question to previous user's phrase.

# Go through all the specific occurencies of user's utterances in all the dialogues.
# In given nodes part find node with assistant's utterance previous to current ocurrence. This node will be source. 
# In given nodes part find node with assistant's utterance next to following current occurence. This node will be target.
# Create edge with current user's utterance, source and target.
# If edge with these source and target already exists,
# just add current user's utterance to this edge if it is different from utterances existing in the edge.


# Don't modify source and target in edges, use them strictly as in dialogues.
# Don't miss any occurence of user's utterances in all the dialogues.

# edges_3 = """This is the end of nodes part.
# So your task is to add edges part to these nodes.
# **Rules:**
# 1) Nodes must be assistant's utterances, edges must be utterances from the user.
# 2) Edges of the graph shall be created only as follows:
# User's utterance in dialogues is part of an edge that comes out of the node with the previous assistant utterance
# and enters the node with the next assistant utterance.
# Pay attention to the fact that same user's utterance can connect different pairs of assistant's utterances
# so can be part of several different edges.
# When several different utterances connect one pair of nodes, edge must contain all these utterances.
# In case of just one node match don't combine several utterances in one edge.
# If specific pair of user's utterance and previous assistant's utterance doesn't have following assistant's utterances in all dialogues, you must find a continuation of 
# this pair with assistant's utterance from one of nodes and connect new edge with this user's utterance
# to that node. This node is kind of problem elaboration stage.
# Typically it is clarifying question to previous user's phrase.
# 3) You must always return valid JSON fenced by a markdown code block. Do not return any additional text.
# 4) Add reason point to the graph with your explanation why you missed some occurences of user's utterances.
# I will give a list of dialogues, your task is to build a graph for this list and nodes part above according to the rules and examples above.
# List of dialogues: """
# b. Pay attention to the fact that same user's utterance can connect different pairs of assistant's utterances
# so can be part of several different edges.
# c. When several different utterances connect one pair of nodes, edge must contain all these utterances.
# d. In case of just one node (source or target) match don't combine several utterances in one edge,
# but instead build several edges going out of the source or entering the target, even if user's utterances match.
# b. Same user's utterance can connect different pairs of nodes
# so be an utterance in several different edges.


# f. If specific pair of user's utterance and previous assistant's utterance terminates a dialogue,
# you must find the most suitable continuation of a dialogue flow
# for this pair with assistant's utterance from one of nodes and create new edge connecting this user's utterance
# to that node.
# g. To find such a node for the closing user's utterance try to find direct or indirect answer
# to this user's query in some of dialogues and use node with that answer.
# It is necessary to choose the most suitable answer.
# h. In case of the terminal user's utterance, from two equal candidates choose:
# firstly - one with similar phrases, then one closest to the beginning of the graph.
# i. If nothing is found for our closing utterance, search for a node with current problem elaboration step.
# Typically it is a clarifying question to current terminal user's utterance.
# c. Don't try connecting nodes using logical reasoning, use rule 2a only.

# b. Don't connect nodes using logical reasoning, use rule 2a only.
# c. Different user's utterances are combined in one edge if only both their previous utterances match
# and both their next utterances match, at the same time.

# All user responses were categorized based on their corresponding assistant prompts. Special attention was given to user utterances that caused the dialogue flow to loop back
# or branch differently, such as changing the service type, ensuring that all possible paths taken in the dialogues are accurately represented in the graph


# a. Every user's utterance in dialogues is one of utterances of an edge that comes out of the node with the previous assistant utterance
# and enters the node with the next assistant utterance.


# 3) Not a single utterance can be missed in created edges.
# 4) Don't create edges with non-existent or modified utterances.

# f. Next 4 steps are for terminal utterances only.
# g. If user's utterance terminates a dialogue, you must find the most suitable continuation of a dialogue flow
# for the pair source-user's utterance, with assistant's utterance from one of nodes and create new edge connecting this user's utterance
# to that node.
# h. To find such a node for the closing user's utterance try to find direct or indirect answer
# to this user's query in some of dialogues and use node with that answer.
# It is necessary to choose the most suitable answer.
# i. In case of the terminal user's utterance, from two equal candidates choose:
# firstly - one with similar phrases, then one closest to the beginning of the graph.
# j. If nothing is found for our closing utterance, search for a node with current problem elaboration step.
# Typically it is a clarifying question to current terminal user's utterance.  

# 3) Don't modify source and target in edges, use them strictly as in dialogues.
# 4) Don't create edges with non-existent or modified utterances.

# 3) Don't modify source or target in edges, use them strictly as in dialogues
# 4) In case when two or more user's utterances have common source only, but not target, don't combine several utterances in one edge,
# but instead build several edges going out of the source, even if user's utterances match.
# 5) In case when two or more user's utterances have common target only, but not source, don't combine several utterances in one edge,
# but instead build several edges entering the target, even if user's utterances match.
# 3) No edges can be created except according to points 2a-2f. 

# h. In case of the terminal user's utterance, from two equal candidates choose:
# firstly - one with similar phrases, then one closest to the beginning of the graph

# f. If user's utterance terminates a dialogue, you must find the most suitable continuation of a dialogue flow
# for the pair source-user's utterance, with assistant's utterance from one of nodes and create new edge connecting this user's utterance
# to that node.
# g. To find such a node for the closing user's utterance try to find direct or indirect answer
# to this user's query in some of dialogues and use node with that answer.
# It is necessary to choose the most suitable answer.
# h. In case of the terminal user's utterance, from two equal candidates choose one with similar phrases.
# i. If nothing is found for our closing utterance, search for a node with current problem elaboration step.
# Typically it is a clarifying question to current terminal user's utterance.  
# 3) To create nodes in ways other than in point 2 is forbidden.

# 2) The edges are created according to the following cycle only:
# a. Take user's utterance from dialogue.
# b. Source of an edge is node with immediately previous assistant's utterance in dialogue.
# c. Target of an edge is node with next assistant's utterance in dialogue.
# d. If edge with source from point 2b and target from point 2c doesn't exist yet, create a new edge with parameters from 2a-2c.
# e. If edge with source from point 2b and target from point 2c already exists, just add user's utterance from 2a to the edge.
# f. Take next user's utterance and go back to step 2a until all user's utterances in dialogues are through.

# 5) In case when two or more user's utterances have common source only (targets are different), don't combine several utterances in one edge,
# but instead build several edges going out of the source, even if user's utterances match.
# 6) In case when two or more user's utterances have common target only (sources are different), don't combine several utterances in one edge,
# but instead build several edges entering the target, even if user's utterances match.

# 6) One of utterances of an edge is an user's utterance between node in point 4) and node in point 5).
# 7) There cannot be other utterances in any edge other than defined in point 6).
# 1) To generate edges of a dialogue don't use any other matching between utterances other than letter by letter comparison.
# 2) It is forbidden to use intents or any other fuzzy comparison.

# edges_3 = """This is the end of nodes part.
# So your task is to add edges part to these nodes.
# **Rules:**

# 1) Nodes are assistant's utterances, edges are utterances from the user.
# 2) Source of an edge is node with immediately previous assistant's utterance in a dialogue.
# 3) Target of an edge is node with next assistant's utterance in dialogue.
# 4) Triggering the transition from source to target is defined exclusively
# based on conversation flow from dialogues in your input.
# 5) There can be more than one utterance in one edge.
# 6) Not a single utterance from all dialogues to be missed in created edges.
# 7) You must always return valid JSON fenced by a markdown code block. Do not return any additional text.
# 8) Add reason point to the graph with your explanation why utterance
# "Okay, please check availability" triggers transition from node 4 to node 5.
# Why utterance "Yes, add the oil change please" triggers transition from node 7 to node 8.
# Why utterance  "Yes, I'll take the urgent slot" triggers transition from node 5 to node 6.
# Why utterance  "Actually, I'd like to change the service type" triggers transition from node 7 to node 3.
# I will give a list of dialogues, your task is to build a graph based on the list and set of nodes above according to the rules and examples above.
# List of dialogues: """

# 5) There can be more than one utterance in one edge.

# edges_3 = """This is the end of nodes part.
# **Rules:**

# 1) Nodes are assistant's utterances, edges are utterances from the user.
# 2) Source of an edge is node with immediately previous assistant's utterance in a dialogue.
# 3) Target of an edge is node with next assistant's utterance in dialogue.
# 4) Triggering the transition from source to target is defined exclusively
# based on conversation flow from dialogues in your input.

# Your task is to create set of edges based on rules 1-4 only. Don't use any other considerations to create edges.
# It is forbidden to create edges based on some other reasoning.

# 5) Not a single utterance from all dialogues to be missed in created edges.
# 6) Don't create edges with non existent or modified utterances.
# 7) You must always return valid JSON fenced by a markdown code block. Do not return any additional text.
# 8) Add reason point to the graph with your explanation if you ignore my task and answer arbitrarily.
# I will give a list of dialogues, your return is a ready dialogue graph (set of nodes with set of edges you generated above).
# List of dialogues: """


edges_3 = """This is the end of nodes part.

Please do the following:
Enumerate all the unique triplets (first assitant's utterance, next user's utterance, next assistant's utterance) from all the dialogues.
For every node find triplet with first assistant's utterance from that node, and create edge with source as the node,
utterance as next user's utterance from the triplet, and target as node with next assistant's utterance from triplet.
If edge with such source and target already exists, just add next user's utterance to the edge.
You must always return valid JSON fenced by a markdown code block. Do not return any additional text.
Add reason point to the graph with your explanation if you ignore my task and answer arbitrarily.
I will give a list of dialogues, your return is a ready dialogue graph (set of nodes with set of edges you generated above).
List of dialogues: """

three_1 = """Your input is a dialogue graph from customer chatbot system - it is
a set of nodes with assistant's utterances and a set of edges that are
triggered by user requests: """

three_2 = """This is the end of input graph.
**Rules:**
1) is_start field in the node is an entry point to the whole graph.
2) Nodes are assistant's utterances, edges are utterances from the user.
Please consider the graph above, list of dialogues below and do the following:

3) For every user's utterance not present in input graph, you must find the most suitable continuation of a dialogue flow
for the sequence of immediately previous assistant's utterance and user's utterance,
with assistant's utterance from one of nodes, and create new edge connecting this user's utterance to that node.
4) To find such a node try to find direct or indirect answer
to this user's query in some of dialogues and use node with that answer.
5) It is necessary to choose the most suitable answer.
From two equal candidates choose:
firstly - one with similar phrases, then one closest to the beginning of the graph.
6) If nothing is found, search for a node with current problem elaboration step.
7) Typically it is a clarifying question to current user's utterance.
8) So it is necessary to add edges to the input graph from utterances which exist in dialogues but absent in the graph. 
9) Add reason point to the graph with your explanation if you ignore my task and answer arbitrarily.
I will give a list of dialogues, your return is a fixed version of dialogue graph above according to the rules above.
List of dialogues: """




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

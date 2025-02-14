from langchain.prompts import PromptTemplate


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

# cycle_graph_generation_prompt_enhanced = PromptTemplate.from_template(
#     """
# Create a dialogue graph for a {topic} conversation that will be used for training data generation. The graph must follow these requirements:

# 1. Dialogue Flow Requirements:
#    - Each assistant message (node) must be a precise question or statement that expects a specific type of response
#    - Each user message (edge) must logically and directly respond to the previous assistant message
#    - All paths must maintain clear context and natural conversation flow
#    - Avoid any ambiguous or overly generic responses

# 2. Graph Structure Requirements:
#    - Must contain at least 2 distinct cycles (return paths)
#    - Each cycle should allow users to:
#      * Return to previous choices for modification
#      * Restart specific parts of the conversation
#      * Change their mind about earlier decisions
#    - Include clear exit points from each major decision path
   
# 3. Core Path Types:
#    - Main success path (completing the intended task)
#    - Multiple modification paths (returning to change choices)
#    - Early exit paths (user decides to stop)
#    - Alternative success paths (achieving goal differently)

# Example of a good cycle structure:
# Assistant: "What size coffee would you like?"
# User: "Medium please"
# Assistant: "Would you like that hot or iced?"
# User: "Actually, can I change my size?"
# Assistant: "Of course! What size would you like instead?"

# Format:
# {{
#     "edges": [
#         {{
#             "source": "node_id",
#             "target": "node_id",
#             "utterances": ["User response text"]
#         }}
#     ],
#     "nodes": [
#         {{
#             "id": "node_id",
#             "label": "semantic_label",
#             "is_start": boolean,
#             "utterances": ["Assistant message text"]
#         }}
#     ]
# }}

# Requirements for node IDs:
# - Must be unique integers
# - Start node should have ID 1
# - IDs should increment sequentially

# Return ONLY the valid JSON without any additional text, commentaries or explanations.
# """
# )

# That means modification path connects another node with one of preceding nodes, it cannot stay with same node.
#    - Each assistant message (node) must be a precise question or statement that expects a specific type of response  
#    - The conversation should look logical and shall not contain unnecessary repetitions
#    - All sources and targets in edges shall not be empty or null
#    - All utterances shall not be empty lists
#    - Each assistant message (node) must logically and directly respond to the immediately previous user message if such exists
# Nodes must be assistant's utterances only and never repeat user's inputs.
# Edges must be user's utterances only and never repeat previous assistant's utterances.
#    - Each user message (edge) must be relevant reaction to the previous assistant message
#    - All edges shall have IDs of existing nodes for source and target
# There shouldn't be more than 20 nodes.
# Target of any edge must be different from the edge's source.

# 2. Graph Structure Requirements:
#    - Must contain at least 2 distinct cycles (return paths)
#    - Each cycle may allow users from none to all of the possibilities as follows:
#      * Return to previous choices for modification
#      * Restart specific parts of the conversation
#      * Change their mind about earlier decisions (if only it is appropriate)
#    - Include clear exit points from each major decision path

cycle_graph_generation_prompt_enhanced = PromptTemplate.from_template(
    """
Create a dialogue graph for a {topic} conversation that will be used for training data generation. The graph must follow these requirements:

1. Dialogue Flow Requirements:
   - Each assistant message (node) must be coherent, reasonable and natural reaction to the immediately previous user message if such exists
   - Each user message (edge) must be coherent, reasonable and conscious reaction to the previous assistant message
   - All paths must maintain clear context and natural flow as in real conversation without unnecessary repetitions
   - Avoid any ambiguous or overly generic responses
   - Every dialogue shall begin with some introduction like greeting etc


2. Graph Structure Requirements:
   - Must contain at least 8 nodes
   - Must contain at least 2 distinct cycles (return paths)
   - Each cycle may allow users to do the folllowing in a natural way only:
     * Return to previous choices for modification
     * Restart specific parts of the conversation
     * Change their mind about earlier decisions
   - Include clear exit points from each major decision path
   
3. Core Path Types:
   - Main success path (completing the intended task)
   - Multiple modification paths (returning to change choices)
   - Early exit paths (user decides to stop)
   - Alternative success paths (achieving goal differently)


Example of a modification path:
Assistant: "What size coffee would you like?"
User: "Medium please"
Assistant: "Would you like that hot or iced?"
User: "Actually, can I change my size?"
Assistant: "Of course! What size would you like instead?"

In previous example of modification path pay attention that modificating question "Actually, can I change my size?"
takes place after first choice "Medium please" and next assistant phrase "Would you like that hot or iced?" have been already spoken.
The modificating question modifies previous user's choice so never immediately follows direct assistant's question
answer to which it modifies, but follows next assistant's question "Would you like that hot or iced?" instead.
After assistant's phrase first goes direct user's answer in any dialogue, "medium please" in the example above.
Further after modification question goes additional assistant's reaction "Of course! What size would you like instead?" where
assistant modifies their question two steps before: "What size coffee would you like?". 
Further follows user's alternative answer then dialogue flow returns to its standard way.

Format:
{{
    "edges": [
        {{
            "source": "node_id",
            "target": "node_id",
            "utterances": ["User response text"]
        }}
    ],
    "nodes": [
        {{
            "id": "node_id",
            "label": "semantic_label",
            "is_start": boolean,
            "utterances": ["Assistant message text"]
        }}
    ]
}}

is_start field is mandatory for all the nodes

Requirements for IDs:
- Must be unique integers
- Start node should have ID 1
- IDs should increment sequentially
- All edges shall have IDs of existing nodes for source and target
- You must remove all edges where target is null

Nodes must be assistant's utterances only and never repeat user's inputs.
Utterances in nodes must be unique, meaning there shall not be repeating utterances in nodes.
Edges must be user's utterances only and never repeat previous assistant's utterances.
Target of any edge must be different from the edge's source.
Just one of edges of the whole graph must have 2-3 different utterances meaning modified answers to the same user's utterance.
Just one of nodes of the whole graph must have 2-3 different utterances meaning fluctuations in formulation of thoughts.
So you need to rephrase utterance in those node while maintaining its general meaning and add rephrased utterances to the node. 

Return ONLY the valid JSON without any additional text, commentaries or explanations.
"""
)


cycle_graph_repair_prompt = PromptTemplate.from_template("""
Fix the invalid transitions in this dialogue graph while keeping its structure.

Current invalid transitions that need to be fixed:
{invalid_transitions}

Original graph structure:
{graph_json}

Requirements for the fix:
1. Keep all node IDs and structure the same
2. Fix ONLY the invalid transitions
3. Make sure the fixed transitions are logical and natural
4. Each user response must logically follow from the assistant's previous message
5. Each assistant response must properly address the user's input

Return ONLY the complete fixed graph JSON with the same structure.
""")

extra_edge_prompt = PromptTemplate.from_template("""
Add extra edges to this dialogue graph while keeping its structure.

Original graph structure:
{graph_json}

Requirements for the modification:
1. Find all nodes with no outgoing edges
2. Add an edge to every found node looping back to the start node of the graph
3. Make sure new transitions are logical and natural
4. Utterances of newly added edges should be such that the phrase in start node following them looks like a continuation of the conversation
5. Targets of new edges must properly address the user's input

Return ONLY the complete fixed graph JSON with the same structure.
""")

part_1 = """Your input is a list of dialogues from customer chatbot system.
Your task is to genearate set of nodes for the dialogue graph corresponding to these dialogues.
Next is an example of the graph (set of rules) how chatbot system looks like - it is
a set of nodes with assistant's utterances and a set of edges that are
triggered by user requests: """

part_2 = """This is the end of the example.
**Rules:**
1) is_start field in the node is an entry point to the whole graph.
2) Nodes must be assistant's utterances only.
3) All the nodes for the graph are created from the resulting groups in point 4) according to the rules 6,7,8,9
with exclusively assistant's utterances only in their original unmodified form.
4) The grouping process looks like follows:
a. Go over all the dialogues, take every assistant's utterance one by one.
b. Search for all assistant's utterances having common basic idea with current utterance,
and not adjacent with it. It is forbidden to select groups based on intents.
c. Form one group from current utterance and all utterances found in 4b.
d. Of the three types of assistant's utterances:
with a question mark at the end,
with an exclamation mark at the end,
affirmative without exclamation mark,
each group can contain only one.
So if two different types encounter in one group, you shall separate them into different groups.
e. Don't miss any assistant's utterance in all the dialogues.
f. Go to next utterance in step 4a, skip those present in existing groups. Don't miss any utterance.
5) Below are examples when two utterances have common basic idea:
if they ask about posession or obtaining of some entities, and these entities are close by in-context meaning to each other:
for example, one entity can be a particular case of the other in the dialogue conext;
when they both are requests without ending question mark and have common words or synonyms;
if they ask whether something is done or about status of something, and have common or synonymous words;
if they ask about accessability of something similar by in-context meaning to each other;
if they have common objects and the remainders of each utterance are close by meaning to each other.
6) If two utterances don't have common words or in-context synonyms,
then they must be separated into different groups.
If one entity is a particular case of the other in the dialogue conext, they are considered synonyms.
7) Don't use user's utterances for grouping process in point 4).
8) Duplicates inside any of the nodes must be removed.
9) If one utterance mentions a problem and the other does not imply any problem, then they shall be separated into different groups.
9) Empty groups shall be removed.
10) Don't use new or modified utterances in the nodes.
11) You must always return valid JSON fenced by a markdown code block. Do not return any additional text.
12) Add reason point to the graph with your explanation why you placed utterances with common basic idea in different groups.
I will give a list of dialogues, your task is to build a set of nodes for this list according to the rules and examples above.
List of dialogues: """

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
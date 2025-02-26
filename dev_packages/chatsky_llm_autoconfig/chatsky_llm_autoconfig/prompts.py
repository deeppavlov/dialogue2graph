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
So you need to paraphrase utterance in those node while maintaining its general meaning and add paraphrased utterances to the node. 

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

part_1i = """Your input is a dialogue graph from customer chatbot system.
Your task is to extend set of nodes for the dialogue graph with input dialogue.
Dialogue graph is a set of nodes with assistant's utterances and a set of edges that are
triggered by user requests. Here goes the input graph: """

# part_2 = """This is the end of the example.
# **Rules:**
# 1) is_start field in the node is an entry point to the whole graph.
# 2) Nodes must be assistant's utterances only.
# 3) All the nodes for the graph are created from the resulting groups in point 4) according to the rules 6,7,8,9
# with exclusively assistant's utterances only in their original unmodified form.
# 4) The grouping process looks like follows:
# a. Go over all the dialogues, take every assistant's utterance one by one.
# b. Search for all assistant's utterances having common basic idea with current utterance,
# and not adjacent with it. It is forbidden to select groups based on intents.
# c. Form one group from current utterance and all utterances found in 4b.
# d. Of the three types of assistant's utterances:
# with a question mark at the end,
# with an exclamation mark at the end,
# affirmative without exclamation mark,
# each group can contain only one.
# So if two different types encounter in one group, you shall separate them into different groups.
# e. Don't miss any assistant's utterance in all the dialogues.
# f. Go to next utterance in step 4a, skip those present in existing groups. Don't miss any utterance.
# 5) Below are examples when two utterances have common basic idea:
# if they ask about posession or obtaining of some entities, and these entities are close by in-context meaning to each other:
# for example, one entity can be a particular case of the other in the dialogue conext;
# when they both are requests without ending question mark and have common words or synonyms;
# if they ask whether something is done or about status of something, and have common or synonymous words;
# if they ask about accessability of something similar by in-context meaning to each other;
# if they have common objects and the remainders of each utterance are close by meaning to each other.
# 6) If two utterances don't have common words or in-context synonyms,
# then they must be separated into different groups.
# If one entity is a particular case of the other in the dialogue conext, they are considered synonyms.
# 7) Don't use user's utterances for grouping process in point 4).
# 8) Duplicates inside any of the nodes must be removed.
# 9) If one utterance mentions a problem and the other does not imply any problem, then they shall be separated into different groups.
# 9) Empty groups shall be removed.
# 10) Don't use new or modified utterances in the nodes.
# 11) You must always return valid JSON fenced by a markdown code block. Do not return any additional text.
# 12) Add reason point to the graph with your explanation why you placed utterances with common basic idea in different groups.
# I will give a list of dialogues, your task is to build a set of nodes for this list according to the rules and examples above.
# List of dialogues: """

# 7) Try not to duplicate edges.
# 7) If two nodes with their immediately following neighbours in the resulting graph overlap, they should be combined into one group.
# 8) Two assistant's utterances with different intents shall be in different groups.
# 10) Nodes shall be grouped so that source and target of any edge are different.
# 10) You must use all the assistant's utterances in resulting set of nodes, not a single utterance to be missed.
# 10) Any edge in the resultng graph connects only different nodes.
# If speaker refers to something absent in the previous context, such dialogues 
# 7) New dialogues which appear as a result of grouping process in step 5) shall be logical and following the context.
# If you see that groups cause such situations:
# new resulting dialogues look illogical,
# or speaker refers to a non-existent context,
# then such groups shall be ungrouped.

part_2i = """This is the end of the graph.
**Rules:**
1) is_start field is mandatory for all the nodes.
2) is_start=True field in the node is an entry point to the whole graph.
3) Nodes must be assistant's utterances from the dialogues in their original form.
4) Several assistant's utterances can belong to same node. To solve the nodes generation task you shall understand
which utterances can be combined into one node, and which can not.
5) Main goal of combining different assistant's utterances into one node is to make the graph structure more efficient and decrease number of nodes.
6) Below is an example when two assistant's utterances go to one node:
'Please, enter the payment method you would like to use: cash or credit card.', and
'How would you prefer to pay?'
7) Next is an example when two assistant's utterances are not necessarily combined in one node:
'I know good chinese restaurant in town.', and
'Do you like Italian food?'
They both talk about restaurant, but different types. And to combine or not depends on surrouding contexts.
If they mention Chinese (or Italian) restaurant names or other details, specific to discussion of exactly this type of restaurant,
these utterences go to different nodes.
But if the remaining context doesn't mention such specific details, for example user could ask for something else instead,
like restaurant with a nice view, these two assistant's utterances would go to one node.
8) Another example shows pair of assistant's utterances which go to different nodes anyway.
'What is your price preference?', and
'Do you like Italian food? I know good restaurant in your price range.'
First utterance asks about price, while second one refers to price mentioned before, that means dialogue flows are different and hence nodes are different.
9) Consider all the dialogue paths which appear as a result of combining utterances and make sure they follow current context.
If they don't, modify your nodes.
10) Example when dialogue doesn't follow current context:
User: I am looking for a cheap restaurant.
Assistant: I found 5 expensive restaurants for you.
11) You shall use every assistant's utterance from the dialogue and the input graph for one node only.
12) You mustn't remove assitant's utterances even if they are similar or synonymous. Make sure you keep all the assistant's utterances in the resulting set of nodes.
12) Doublecheck that all the assistant's utterances from the dialogue and the input graph are present in resulting set of nodes,
not a single assistant's utterance to be missed.
13) Don't modify utterances even slightly (even a single letter) before placing them into the nodes.
14) You must always return valid JSON fenced by a markdown code block. Do not return any additional text.
15) Add reason point to your answer with an explanation why you didn't use all the assistant's utterances from the dialogue and the input graph.
I will give you the dialogue, your task is to extend a set of nodes of the graph above from this dialogue according to the rules and examples above.
The dialogue: """

part_2 = """This is the end of the example.
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
h. Go to next utterance in step 5a. Don't miss any utterance.
6) Don't use user's utterances for grouping process.
7) Every assistant's utterance not included in any group shall be present in its own group of single utterance.
8) You must doublecheck that all the assistant's utterances are present in resulting set of nodes, not a single utterance to be missed.
9) Always place adjacent assistant's utterances separated by only one user's utterance into different groups.
10) You must always return valid JSON fenced by a markdown code block. Do not return any additional text.
11) Add reason point to your answer with an explanation why some of the assistant's utterances were not included in the resulting groups.
I will give a list of dialogues, your task is to build a set of nodes for this list according to the rules and examples above.
List of dialogues: """

# 7) Combining nodes into groups must not allow dialogue flow to be distorted.
# 8) New dialogues which appear as a result of grouping process shall be logical and following the context.
# 7) Every assistant's utterance not included in any group shall be present in its own group of single utterance.
# 8) You must doublecheck that all the assistant's utterances are present in resulting set of nodes, not a single utterance to be missed.
# 9) Don't use user's utterances for grouping process.
# 10) Always place adjacent assistant's utterances separated by only one user's utterance into different groups.
# 6) One possible way how to find groups is to compare contexts one assistant's utterance before and one assistant's utterance after.
# When these contexts are similar, it is a good reason to combine nodes in one group.
# 6) Dialogue flow resulting the grouping process shall remain natural and conscious.

# 7) Every assistant's utterance not included in any group shall be present in its own group of single utterance.
# 8) Use every assistant's utterance for one group only.
# 9) Don't use user's utterances for grouping process.
# 10) All the nodes for the graph are created from the resulting groups decribed above.
# 9) Make sure that none of the edges and nodes contain contradictory utterances.
# 10) Make sure that nodes in dialogue flow following combined node are coherent.

# 9) Make sure that combining assistant's utterances into nodes does not distort dialogue flow logics and context.
# 10) Make sure that dialogue flows don't have contradictory chains (according to point 4).
# 9) Make sure that all the assistant's utterances are present in resulting set of nodes, not a single utterance to be missed.
# 9) Make sure you keep balance between combining nodes to provide graph efficiency and saving graph consistency (when any path of the graph
# is consistent and doesn't contain contradictions).
# 10) Make sure that any pair of utterances in any single node or edge doesn't contain mutually exclusive essential concepts between each other.
# 9) Make sure not a single dialogue path has mutually exclusive essential concepts like cheap-expensive, or center-south, etc.
# 9) Make sure not a single dialogue path has mutually exclusive essential concepts between any node and edges connected to it.
# 13) Make sure that any two utterances having contradictory concepts are in different nodes and different edges.
# 12) Make sure you keep balance between combining nodes to provide graph efficiency and saving graph consistency (when any path of the graph
# is consistent).

part_2_v2 = """This is the end of the example.
**Rules:**
1) is_start field is mandatory for all the nodes.
2) is_start=True field in the node is an entry point to the whole graph.
3) Nodes must be assistant's utterances only.
4) Every node can have more than one utterance, in this case nodes with several utterances connected by edges with several
utterances mean that there are dialogue flows connecting all the combinations of them (every utterance from each node with
every utterance from the other nodes and edges lying in one graph path). 
5) Some assistant's utterances can belong to same node. To solve the nodes generation task you shall understand
which utterances can be combined into one node, and which can not.
6) Main goal of combining different utterances into one node is to avoid duplication of edges with similar utterances,
connecting duplicated similar nodes, to make the graph structure more efficient and decrease number of nodes.
7) To understand when two assistant's utterances can be combined in one node, for every assistant's utterance
you shall consider all its pairs with assistant's utterances of same intent in all the dialogues. If you can swap this pair members (along
with immediate contexts) between each other while sticking to the general meaning of the dialogue,
this pair shall be in one node if and only if all the points 11-15 are satisfied.
8) Below is an example when two assistant's utterances go to one node:
'Please, enter the payment method you would like to use: cash or credit card.', and
'How would you prefer to pay?'
9) Next is an example when two utterances are not necessarily combined in one node:
a. I know good chinese restaurant in town.
b. Do you like Italian food?
They both talk about restaurant, but different types. And to combine or not depends on surrouding contexts.
If they mention Chinese (or Italian) restaurant names or other details, specific to discussion of exactly this type of restaurant,
these utterences go to different nodes.
But if the remaining context doesn't mention such specific details, for example user could ask for something else instead,
like restaurant with a nice view, these two utterances would go to one node.
10) Another example shows pair of utterances which go to different nodes anyway.
a. I know good chinese restaurant in town. What price range do you mean?
b. Do you like Italian food? I know good restaurant in your price range.
Utterance 10a. asks about price, while utterance 10b. refers to price mentioned before, that means dialogue flows are different and can't be part of one node.
11) Always place adjacent assistant's utterances into different nodes.
12) Consider all the dialogue paths (according to point 4) which appear as a result of combining utterances and make sure they are coherent and following current context.
13) Any two utterances having contradictory concepts shall be in different nodes and different edges.
14) Make sure you use every assistant's utterance for one node only.
15) Don't use new or modified utterances in the nodes.
16) Doublecheck that all the assistant's utterances are present in resulting set of nodes, not a single utterance to be missed.
17) Make sure not a single user's utterance is used in nodes.
18) You must always return valid JSON fenced by a markdown code block. Do not return any additional text.
19) Add reason point to your answer with an explanation why you didn't use all the assistant's utterances.
I will give a list of dialogues, your task is to build a set of nodes for this list according to the rules and examples above.
List of dialogues: """

# 10) Make sure you use every assistant's utterance for one node only.
# 11) Don't use new or modified utterances in the nodes.
# 12) Doublecheck that all the assistant's utterances are present in resulting set of nodes, not a single utterance to be missed.
# 13) Make sure not a single user's utterance is used in nodes.
# 13) Make sure not a single user's utterance is used in nodes (except those duplicating assistant's utterances).

part_2_v3 = """This is the end of the example.
**Rules:**
1) is_start field is mandatory for all the nodes.
2) is_start=True field in the node is an entry point to the whole graph.
3) Nodes must be assistant's utterances from the dialogues in their original form.
4) Several assistant's utterances can belong to same node. To solve the nodes generation task you shall understand
which utterances can be combined into one node, and which can not.
5) Main goal of combining different assistant's utterances into one node is to make the graph structure more efficient and decrease number of nodes.
6) Below is an example when two assistant's utterances go to one node:
'Please, enter the payment method you would like to use: cash or credit card.', and
'How would you prefer to pay?'
7) Next is an example when two assistant's utterances are not necessarily combined in one node:
'I know good chinese restaurant in town.', and
'Do you like Italian food?'
They both talk about restaurant, but different types. And to combine or not depends on surrouding contexts.
If they mention Chinese (or Italian) restaurant names or other details, specific to discussion of exactly this type of restaurant,
these utterences go to different nodes.
But if the remaining context doesn't mention such specific details, for example user could ask for something else instead,
like restaurant with a nice view, these two assistant's utterances would go to one node.
8) Another example shows pair of assistant's utterances which go to different nodes anyway.
'What is your price preference?', and
'Do you like Italian food? I know good restaurant in your price range.'
First utterance asks about price, while second one refers to price mentioned before, that means dialogue flows are different and hence nodes are different.
9) Consider all the dialogue paths which appear as a result of combining utterances and make sure they follow current context.
If they don't, modify your nodes.
10) Example when dialogue doesn't follow current context:
User: I am looking for a cheap restaurant.
Assistant: I found 5 expensive restaurants for you.
11) You shall use every assistant's utterance from the list of the dialogues for one node only.
12) Doublecheck that all the assistant's utterances are present in resulting set of nodes, not a single utterance to be missed.
13) You mustn't remove assitant's utterances even if they are similar or synonymous. Make sure you keep all the assistant's utterances in the set of nodes.
14) Don't modify dialogue utterances even slightly (even a single letter) before placing them into the nodes.
15) You must always return valid JSON fenced by a markdown code block. Do not return any additional text.
16) Add reason point to your answer with an explanation why you didn't use all the assistant's utterances from all the dialogues.
I will give a list of dialogues, your task is to build a set of nodes containing all the utterances from this list according to the rules and examples above.
List of dialogues: """


# part_2 = """This is the end of the example.
# **Rules:**
# 1) is_start field is mandatory for all the nodes.
# 2) is_start=True field in the node is an entry point to the whole graph.
# 3) Nodes must be assistant's utterances only.
# 4) All the nodes for the graph are created from the resulting groups in point 5) according to the rules 6,7,8,9
# with exclusively assistant's utterances only in their original unmodified form.
# 5) The grouping process looks like follows:
# a. Go over all the dialogues, take every assistant's utterance one by one.
# b. Search for all assistant's utterances surrounded by utterances with similar intents: one pair of assistant's and user's utterances previous to
# and one pair of user's and assistant's utterances next to current assistants' utterance.
# c. Place utterances with similar surrounings from 5)b. into one group.
# d. Go to next utterance in step 5a. Don't miss any utterance.
# 6) Below is an example when two utterances have adjacent pairs with similar intents:
# They are:
# 'Please, enter the payment method you would like to use: cash or credit card.', and
# 'How would you prefer to pay?'
# Surrounding pairs for both of them are similar so they need to be united into one group of nodes.
# 7) Don't use user's utterances for grouping process in point 5).
# 8) You must use all the assistant's utterances in resulting set of nodes.
# 9) You must always return valid JSON fenced by a markdown code block. Do not return any additional text.
# 10) Add reason point to the graph with your explanation why you didn't group assistant's utterances and why you didn't use all the assistant's utterances in dialogues.
# I will give a list of dialogues, your task is to build a set of nodes for this list according to the rules and examples above.
# List of dialogues: """



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
          {'id': 3, 'label': 'ask_payment_method', 'is_start': False, 'utterances': [ 'Please, enter the payment method you would like to use: cash or credit card.', 'How would you prefer to pay?']},
          {"id": 4, "label": "ask_to_redo", "is_start": False, "utterances": [ "Something is wrong, can you please use other payment method or start order again"]}
      ],
      'reason': ""
}
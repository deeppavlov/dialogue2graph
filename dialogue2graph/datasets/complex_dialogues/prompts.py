# flake8: noqa
from langchain.prompts import PromptTemplate

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

cycle_graph_generation_prompt_enhanced = PromptTemplate.from_template(
    """
Create a dialogue graph for a {topic} conversation that will be used for training data generation. The graph must follow these requirements:

1. Dialogue Flow Requirements:
   - Each assistant message (node) must be a precise question or statement that expects a specific type of response
   - Each user message (edge) must logically and directly respond to the previous assistant message
   - All paths must maintain clear context and natural conversation flow
   - Avoid any ambiguous or overly generic responses

2. Graph Structure Requirements:
   - Must contain at least 2 distinct cycles (return paths)
   - Each cycle should allow users to:
     * Return to previous choices for modification
     * Restart specific parts of the conversation
     * Change their mind about earlier decisions
   - Include clear exit points from each major decision path
   
3. Core Path Types:
   - Main success path (completing the intended task)
   - Multiple modification paths (returning to change choices)
   - Early exit paths (user decides to stop)
   - Alternative success paths (achieving goal differently)

Example of a good cycle structure:
Assistant: "What size coffee would you like?"
User: "Medium please"
Assistant: "Would you like that hot or iced?"
User: "Actually, can I change my size?"
Assistant: "Of course! What size would you like instead?"

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

Requirements for node IDs:
- Must be unique integers
- Start node should have ID 1
- IDs should increment sequentially

Return ONLY the valid JSON without any additional text, commentaries or explanations.
"""
)

cycle_graph_generation_prompt_informal= PromptTemplate.from_template(
    """
Create a dialogue graph for real conversations between two people about {topic}. The graph must follow these requirements:

1. Dialogue Flow Requirements:
   - Each assistant message (node) must be coherent, reasonable and natural reaction to the immediately previous user message if such exists
   - Each user message (edge) must be coherent, reasonable and conscious reaction to the previous assistant message
   - All paths must maintain clear context and natural flow as in real conversation without unnecessary repetitions
   - Avoid any ambiguous or generic responses
   - Every dialogue shall begin with some introduction like greeting etc
   - Every dialogue must be logically complete

2. Graph Structure Requirements:
   - Nodes are assistant's utterances, edges are utterances from the user
   - Dialogue flow goes from source to target for all edges
   - Graph must contain at least 8 nodes
   - Graph must contain at least 2 distinct cycles (return paths)
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

Example of the graph:
{graph_example}

Please use informal language like in the example above. Conversation shall look like a real one,
with all the real details inherent in a real conversation on this topic: {topic}.
Some of nodes and edges shall have several different utterances meaning different details of same intent.
For examples see provide_contact_info and provide_recommendations nodes of the above graph.
So you need to add several different details per node/edge for some of nodes and edges.
Make sure that number of edges with several utterances doesn't exceed 5 edges.

is_start field is mandatory for all the nodes.

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

Return ONLY the valid JSON without any additional text, commentaries or explanations.
"""
)

cycle_graph_repair_prompt = PromptTemplate.from_template(
    """
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
"""
)
graph_example = {
    'edges':
        [{'source': 1,
  'target': 2,
  'utterances': ["I'm looking for an Indian restaurant, preferably in the centre of town."]},
 {'source': 2,
  'target': 5,
  'utterances': ['I would prefer cheap restaurants.']},
 {'source': 5,
  'target': 7,
  'utterances': ['Sure please book a table there fore 7 people at 12:15 on saturday']},
 {'source': 1,
  'target': 5,
  'utterances': ['I am looking for a restaurant. The restaurant should be in the moderate price range and should be in the east']},
 {'source': 5,
  'target': 10,
  'utterances': ['The restaurant should serve italian food.']},
 {'source': 6,
  'target': 7,
  'utterances': ['I will have 5 people and we would like 12:15 if possible. Thanks.']},
 {'source': 7,
  'target': 9,
  'utterances': ["No that's all I needed. Thank you!",
   'Thanks for you help. I only need the restaurant reservation. Goodbye.']},
 {'source': 10,
  'target': 11,
  'utterances': ['What other restaurants in that area serve Italian food?']},
 {'source': 11,
  'target': 6,
  'utterances': ['No, that will do. Can I book a table for monday?']},
 {'source': 1,
  'target': 3,
  'utterances': ["I'm looking for a place to dine on the south side of town. Please find a place that's in the expensive price range."]},
 {'source': 3,
  'target': 8,
  'utterances': ['Do you have a favorite you could recommend? I will need the phone and postcode and food type also please.']},
 {'source': 1,
  'target': 4,
  'utterances': ['I am looking for a cheap restaurant in the centre.']},
 {'source': 4,
  'target': 8,
  'utterances': ["Yes, may I have the address, postcode, and phone number for Golden House? I'll book it myself."]},
 {'source': 8,
  'target': 9,
  'utterances': ['No, that will be it. Thank you for your help.',
   "Thanks, that's all I need. Have a nice day."]}],
   'nodes':
        [{'id': 1,
  'label': 'start',
  'is_start': True,
  'utterances': ['Hello! How can I help you?']},
 {'id': 2,
  'label': 'ask_price_range',
  'is_start': False,
  'utterances': ['There are a number of options for Indian restaurants in the centre of town. What price range would you like?']},
 {'id': 3,
  'label': 'ask_cuisine_preference',
  'is_start': False,
  'utterances': ['I found five expensive restaurants on the south side of town. Would you prefer Chinese, Indian, Italian or Mexican?']},
 {'id': 4,
  'label': 'ask_interest_in_options',
  'is_start': False,
  'utterances': ['I have found many possibilities. Golden house is chinese and the river bar steakhouse and grill serves modern european. Are either of those of interest for you?']},
 {'id': 5,
  'label': 'provide_recommendations',
  'is_start': False,
  'utterances': ['Try curry prince or pizza hut fen ditton',
   'I was able to find three options in your price range, may I recommend The Gandhi?']},
 {'id': 6,
  'label': 'ask_reservation_details',
  'is_start': False,
  'utterances': ['Absolutely, how many people will you have and what time are you wanting the reservation?']},
 {'id': 7,
  'label': 'confirm_booking',
  'is_start': False,
  'utterances': ['I was able to book that for you. They will reserve your table for 15 minutes. Your reference number is 6EQ61SD9 . Is there anything more I can help with?',
   'You are booked, the reference number is AF2GJ7G6, may I assist with anything else?']},
 {'id': 8,
  'label': 'provide_contact_info',
  'is_start': False,
  'utterances': ['They are located at 12 Lensfield Road City Centre, postcode cb21eg, and phone number 01842753771.',
   'If you ask me, the Chiquito Restaurant Bar serves the best Mexican food around. Their postcode is cb17dy. You can reach them at 01223400170. Can I help with anything else?']},
 {'id': 9,
  'label': 'closing',
  'is_start': False,
  'utterances': ['Thank you for calling, enjoy!',
   "You're welcome. Thank you for contacting Cambridge TownInfo centre, and have a great day.",
   'Thank you. Have a nice day.',
   'Thank you for using our service. Have a great day.']},
 {'id': 10,
  'label': 'offer_reservation',
  'is_start': False,
  'utterances': ['Pizza hut fen ditton serves italian food in the east, would you like a reservation?']},
 {'id': 11,
  'label': 'confirm_no_other_areas',
  'is_start': False,
  'utterances': ['Pizza hut fen ditton is the only Italian restaurant, in the east, in the moderate price range. Do you want me to try other areas?']}
    ]
}
extending_prompt_part_1 = """Your input is a dialogue graph from customer chatbot system.
Your task is to extend set of nodes for the dialogue graph with list of input dialogues.
Dialogue graph is a set of nodes with assistant's utterances and a set of edges that are
triggered by user requests. Here goes the input graph: """

extending_prompt_part_2 = """This is the end of the graph.
**Rules:**
1) is_start field is mandatory for all the nodes.
2) is_start=True field in the node is an entry point to the whole graph.
3) Nodes are assistant's utterances.
4) Several assistant's utterances can belong to same node. To solve the nodes extension task you shall understand
which utterances can be combined into one node, and which can not.
5) When a node contains more than one utterance, it means that any of these utterances can be part of dialogue paths going thru this node,
so number of dialogue paths is multiplied and all the utterances combinations of all the nodes in the path form the dialogue paths.
6) Main goal of combining different assistant's utterances into one node is to make the graph structure more efficient and decrease number of nodes.
6) As a result of combining nodes there can appear new dialogue paths. And not all of them can be logical.
And the reason of this is a combined node generates combined contexts, meaning every utterance in a combined node will have
additional context from the other utterance in the node.
7) You shall track such cases of illogical paths and separate combined nodes if needed.
8) Consider all the dialogue paths which appear as a result of combining utterances and make sure they are coherent and logical.
If they aren't, separate your nodes when needed.
9) You mustn't remove assistant's utterances even if they are similar or synonymous. Make sure you keep all the assistant's utterances in the resulting set of nodes.
10) Doublecheck that all the assistant's utterances from the input dialogues are present in resulting set of nodes,
not a single assistant's utterance to be missed.
11) You must always return valid JSON fenced by a markdown code block. Do not return any additional text.
12) Here go examples for list of dialogues and corresponding set of nodes.
{examples}
This is the end of examples
I will give you the list of dialogues, your task is to extend a set of nodes of the graph above with assistant's utterances from the dialogues according to the rules and examples above.
The list of dialogues: """

dg_examples = [
    {
        "dialogs": [
            {
                "text": "Hey there! How can I help you today?",
                "participant": "assistant",
            },
            {"text": "I need to book a ride to the airport.", "participant": "user"},
            {
                "text": "Sure! I can help with that. When is your flight, and where are you departing from?",
                "participant": "assistant",
            },
            {"text": "Do you have any other options?", "participant": "user"},
            {
                "text": "If you'd prefer, I can send you options for ride-share services instead. Would you like that?",
                "participant": "assistant",
            },
            {"text": "No, I'll manage on my own.", "participant": "user"},
            {
                "text": "No worries! Feel free to reach out anytime.",
                "participant": "assistant",
            },
            {"text": "Alright, thanks anyway.", "participant": "user"},
            {
                "text": "You're welcome! Have a fantastic trip!",
                "participant": "assistant",
            },
        ],
        "nodes": [
            {
                "id": 1,
                "label": "",
                "is_start": True,
                "utterances": ["Hey there! How can I help you today?"],
            },
            {
                "id": 2,
                "label": "",
                "is_start": False,
                "utterances": [
                    "Sure! I can help with that. When is your flight, and where are you departing from?"
                ],
            },
            {
                "id": 3,
                "label": "",
                "is_start": False,
                "utterances": [
                    "If you'd prefer, I can send you options for ride-share services instead. Would you like that?"
                ],
            },
            {
                "id": 4,
                "label": "",
                "is_start": False,
                "utterances": ["No worries! Feel free to reach out anytime."],
            },
            {
                "id": 5,
                "label": "",
                "is_start": False,
                "utterances": ["You're welcome! Have a fantastic trip!"],
            },
        ],
    },
    {
        "dialogs": [
            [
                {
                    "text": "Hey there! How can I help you today?",
                    "participant": "assistant",
                },
                {
                    "text": "I need to book a ride to the airport.",
                    "participant": "user",
                },
                {
                    "text": "Sure! I can help with that. When is your flight, and where are you departing from?",
                    "participant": "assistant",
                },
                {"text": "Do you have any other options?", "participant": "user"},
                {
                    "text": "If you'd prefer, I can send you options for ride-share services instead. Would you like that?",
                    "participant": "assistant",
                },
                {"text": "Actually, never mind.", "participant": "user"},
                {
                    "text": "Alright, let me know if you need help later. Have a great day!",
                    "participant": "assistant",
                },
                {"text": "Okay, have a great day!", "participant": "user"},
                {"text": "Glad to help! Safe travels.", "participant": "assistant"},
            ],
            [
                {
                    "text": "Hey there! How can I help you today?",
                    "participant": "assistant",
                },
                {
                    "text": "I need to book a ride to the airport.",
                    "participant": "user",
                },
                {
                    "text": "Sure! I can help with that. When is your flight, and where are you departing from?",
                    "participant": "assistant",
                },
                {"text": "Do you have any other options?", "participant": "user"},
                {
                    "text": "If you'd prefer, I can send you options for ride-share services instead. Would you like that?",
                    "participant": "assistant",
                },
                {"text": "No, I'll manage on my own.", "participant": "user"},
                {
                    "text": "No worries! Feel free to reach out anytime.",
                    "participant": "assistant",
                },
                {"text": "Alright, thanks anyway.", "participant": "user"},
                {
                    "text": "You're welcome! Have a fantastic trip!",
                    "participant": "assistant",
                },
            ],
        ],
        "nodes": [
            {
                "id": 1,
                "label": "",
                "is_start": True,
                "utterances": ["Hey there! How can I help you today?"],
            },
            {
                "id": 2,
                "label": "",
                "is_start": False,
                "utterances": [
                    "Sure! I can help with that. When is your flight, and where are you departing from?"
                ],
            },
            {
                "id": 3,
                "label": "",
                "is_start": False,
                "utterances": [
                    "If you'd prefer, I can send you options for ride-share services instead. Would you like that?"
                ],
            },
            {
                "id": 4,
                "label": "",
                "is_start": False,
                "utterances": [
                    "Alright, let me know if you need help later. Have a great day!"
                ],
            },
            {
                "id": 5,
                "label": "",
                "is_start": False,
                "utterances": ["Glad to help! Safe travels."],
            },
            {
                "id": 6,
                "label": "",
                "is_start": False,
                "utterances": ["No worries! Feel free to reach out anytime."],
            },
            {
                "id": 7,
                "label": "",
                "is_start": False,
                "utterances": ["You're welcome! Have a fantastic trip!"],
            },
        ],
    },
]

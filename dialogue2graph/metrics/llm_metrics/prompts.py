from langchain.prompts import PromptTemplate

compare_graphs_prompt = PromptTemplate.from_template(
    "You will get two dialogue graphs in following format: {graph_example}. "
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

graph_example = {
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
            "utterances": ["Please, enter the payment method you would like to use: cash or credit card."],
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

"""
LLM Metrics.
------------

This module contains functions that checks Graphs and Dialogues for various metrics using LLM calls.
"""

import logging
import json
from typing import List, TypedDict
from chatsky_llm_autoconfig.graph import BaseGraph, Graph
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

# Set up logging
logging.basicConfig(level=logging.INFO)


def are_triplets_valid(G: Graph, model: BaseChatModel) -> dict[str]:
    """
    Validates dialogue graph structure and logical transitions between nodes.

    Parameters:
        G (BaseGraph): The dialogue graph to validate
        model (BaseChatModel): The LLM model to use for validation

    Returns:
        dict: {'value': bool, 'description': str}
    """

    # Define validation result model
    class TransitionValidationResult(BaseModel):
        isValid: bool = Field(description="Whether the transition is valid or not")
        description: str = Field(description="Explanation of why it's valid or invalid")

    # Create prompt template
    triplet_validate_prompt_template = """
    You are evaluating if dialog transitions make logical sense.
    
    Given this conversation graph in JSON:
    {json_graph}
    
    For the current transition:
    Source (Assistant): {source_utterances}
    User Response: {edge_utterances}
    Target (Assistant): {target_utterances}

    EVALUATE: Do these three messages form a logical sequence in the conversation?
    Consider:
    1. Does the assistant's first response naturally lead to the user's response?
    2. Does the user's response logically connect to the assistant's next message?
    3. Is the overall flow natural and coherent?


    Reply in JSON format:
    {{"isValid": true/false, "description": "Brief explanation of why it's valid or invalid"}}
    """

    triplet_validate_prompt = PromptTemplate(
        input_variables=["json_graph", "source_utterances", "edge_utterances", "target_utterances"], template=triplet_validate_prompt_template
    )

    parser = PydanticOutputParser(pydantic_object=TransitionValidationResult)

    # Convert graph to JSON string
    graph_json = json.dumps(G.graph_dict)

    # Create node mapping
    node_map = {node["id"]: node for node in G.graph_dict["nodes"]}

    overall_valid = True
    descriptions = []

    for edge in G.graph_dict["edges"]:
        source_id = edge["source"]
        target_id = edge["target"]

        if source_id not in node_map or target_id not in node_map:
            description = f"Invalid edge: missing node reference {source_id} -> {target_id}"
            overall_valid = False
            descriptions.append(description)
            continue

        # Get utterances
        source_utterances = node_map[source_id]["utterances"]
        target_utterances = node_map[target_id]["utterances"]
        edge_utterances = edge["utterances"]

        # Prepare input for validation
        input_data = {
            "json_graph": graph_json,
            "source_utterances": source_utterances,
            "edge_utterances": edge_utterances,
            "target_utterances": target_utterances,
        }

        # print(triplet_validate_prompt.format(**input_data))

        # Run validation
        triplet_check_chain = triplet_validate_prompt | model | parser
        response = triplet_check_chain.invoke(input_data)

        if not response.isValid:
            overall_valid = False
            description = f"Invalid transition: {response.description}"
            logging.info(description)
            descriptions.append(description)

    result = {"value": overall_valid, "description": " ".join(descriptions) if descriptions else "All transitions are valid."}

    return result


def find_graph_ends(G: Graph, model: BaseChatModel) -> dict[str]:
    """
    Validates dialogue graph structure and logical transitions between nodes.

    Parameters:
        G (BaseGraph): The dialogue graph to validate
        model (BaseChatModel): The LLM model to use for validation

    Returns:
        dict: {'value': bool, 'description': str}
    """

    # Define validation result model
    class GraphEndsResult(BaseModel):
        ends: list = Field(description="IDs of ending nodes")
        description: str = Field(description="Explanation of model's decision")

    # Create prompt template
    graph_ends_prompt_template = """
    Your task is to find IDs of all the nodes satisfying condition below:
    Let's consider node with id A.
    There is only edge with source=A in the whole graph, and target of this edge is located earlier in the dialogue flow.
    
    Given this conversation graph in JSON:
    {json_graph}
 
 

    Reply in JSON format:
    {{"ends": [id1, id2, ...], "description": "Brief explanation of your decision"}}
    """

    graph_ends_prompt = PromptTemplate(
        input_variables=["json_graph"], template=graph_ends_prompt_template
    )

    parser = PydanticOutputParser(pydantic_object=GraphEndsResult)

    # Convert graph to JSON string
    graph_json = json.dumps(G.graph_dict)
    
        # Prepare input for validation
    input_data = {
        "json_graph": graph_json,
    }

        # print(triplet_validate_prompt.format(**input_data))

        # Run validation
    find_ends_chain = graph_ends_prompt | model | parser
    response = find_ends_chain.invoke(input_data)
    result = {"value": response.ends, "description": response.description}

    return result


def is_theme_valid(G: BaseGraph, model: BaseChatModel, topic: str) -> dict[str]:
    """
    Validates if the dialog stays on theme/topic throughout the conversation.

    Parameters:
        G (BaseGraph): The dialog graph to validate
        model (BaseChatModel): The LLM model to use for validation
        topic (str): The expected topic of the dialog

    Returns:
        dict: {'value': bool, 'description': str}
    """

    class ThemeValidationResult(BaseModel):
        isValid: bool = Field(description="Whether the dialog stays on theme")
        description: str = Field(description="Explanation of why it's valid or invalid")

    theme_validate_prompt = PromptTemplate(
        input_variables=["utterances", "topic"],
        template="""
        You are given a dialog between assistant and a user.
        Analyze the following dialog and determine if it is connected to the topic below.
        
        Topic: {topic}
        
        Dialog utterances:
        {utterances}
        
        Provide your answer in the following JSON format:
        {{"isValid": true or false, "description": "Explanation of why it's valid or invalid."}}

        Your answer:
        """,
    )

    parser = PydanticOutputParser(pydantic_object=ThemeValidationResult)

    # Extract all utterances from the graph
    graph = G.graph_dict
    all_utterances = []

    # Get assistant utterances from nodes
    for node in graph["nodes"]:
        all_utterances.extend(node.get("utterances", []))

    # Get user utterances from edges
    for edge in graph["edges"]:
        all_utterances.extend(edge.get("utterances", []))

    # Format utterances for the prompt
    formatted_utterances = "\n".join([f"- {utterance}" for utterance in all_utterances])

    # Prepare input data
    input_data = {"utterances": formatted_utterances, "topic": topic}

    # Create and run the chain
    theme_check_chain = theme_validate_prompt | model | parser
    response = theme_check_chain.invoke(input_data)

    return {"value": response.isValid, "description": response.description}



class InvalidTransition(TypedDict):
    from_: List[str]  # Using from_ because 'from' is reserved
    user: List[str]
    to: List[str]
    reason: str


class GraphValidationResult(TypedDict):
    is_valid: bool
    invalid_transitions: List[InvalidTransition]

    # 3. Is the overall flow natural and coherent?
    # Consider:
    # 1. Does the user's response looks as a natural answer human can give?
    # 2. Does the assistant's response looks natural in a dialogue context?

# def graph_validation(G: BaseGraph, model: BaseChatModel) -> GraphValidationResult:
#     """
#     Проверяет валидность графа диалога
#     Возвращает:
#     {
#         "is_valid": bool,  # валиден ли граф в целом
#         "invalid_transitions": [  # список невалидных переходов
#             {
#                 "from": ["source utterance"],
#                 "user": ["user utterance"],
#                 "to": ["target utterance"],
#                 "reason": "причина невалидности"
#             },
#             ...
#         ]
#     }
#     """
#     # Define validation result model
#     class TransitionValidationResult(BaseModel):
#         isValid: bool = Field(description="Whether the transition is valid or not")
#         description: str = Field(description="Explanation of why it's valid or invalid")

#     # Create prompt template
#     triplet_validate_prompt = PromptTemplate(
#         input_variables=["json_graph", "source_utterances", "edge_utterances", "target_utterances"],
#         template="""
#     You are evaluating if dialog transitions make sense.
    
#     Given this conversation graph in JSON:
#     {json_graph}
    
#     For the current transition:
#     Source (Assistant): {source_utterances}
#     User Response: {edge_utterances}
#     Target (Assistant): {target_utterances}

#     EVALUATE: Do these three messages look like a real conversation?

#     Reply in JSON format:
#     {{"isValid": true/false, "description": "Brief explanation of why it's valid or invalid"}}
#     """
#     )

#     parser = PydanticOutputParser(pydantic_object=TransitionValidationResult)

#     # Convert graph to JSON string
#     graph_json = json.dumps(G.graph_dict)

#     # Create node mapping
#     node_map = {node["id"]: node for node in G.graph_dict["nodes"]}
#     invalid_transitions = []
#     is_valid = True

#     for edge in G.graph_dict["edges"]:
#         source_id = edge["source"]
#         target_id = edge["target"]

#         # Проверяем существование узлов
#         if source_id not in node_map or target_id not in node_map:
#             is_valid = False
#             continue

#         # Get utterances
#         source_node = node_map[source_id]
#         target_node = node_map[target_id]

#         # Prepare input for validation
#         input_data = {
#             "json_graph": graph_json,
#             "source_utterances": source_node["utterances"],
#             "edge_utterances": edge["utterances"],
#             "target_utterances": target_node["utterances"]
#         }

#         # Run validation
#         triplet_check_chain = triplet_validate_prompt | model | parser
#         result = triplet_check_chain.invoke(input_data)

#         if not result.isValid:
#             is_valid = False
#             invalid_transitions.append({
#                 "from": source_node["utterances"],
#                 "user": edge["utterances"],
#                 "to": target_node["utterances"],
#                 "reason": result.description
#             })
#     print("is_valid: ", is_valid)
#     return {
#         "is_valid": is_valid,
#         "invalid_transitions": invalid_transitions
#     }

def graph_validation(G: BaseGraph, model: BaseChatModel) -> GraphValidationResult:
    """
    Проверяет валидность графа диалога
    Возвращает:
    {
        "is_valid": bool,  # валиден ли граф в целом
        "invalid_transitions": [  # список невалидных переходов
            {
                "from": ["source utterance"],
                "user": ["user utterance"],
                "to": ["target utterance"],
                "reason": "причина невалидности"
            },
            ...
        ]
    }
    """
    # Define validation result model
    class TransitionValidationResult(BaseModel):
        isValid: bool = Field(description="Whether the transition is valid or not")
        description: str = Field(description="Explanation of why it's valid or invalid")

    # Create prompt template
    triplet_validate_prompt = PromptTemplate(
        input_variables=["source_utterances", "edge_utterances", "target_utterances"],
        template="""
    You are evaluating if dialog transitions make sense.

    For the current transition:
    Source (Assistant): {source_utterances}
    User Response: {edge_utterances}
    Target (Assistant): {target_utterances}

    EVALUATE: Do these three messages look like a real conversation?

    Reply in JSON format:
    {{"isValid": true/false, "description": "Brief explanation of why it's valid or invalid"}}
    """
    )

    parser = PydanticOutputParser(pydantic_object=TransitionValidationResult)

    # Create node mapping
    node_map = {node["id"]: node for node in G.graph_dict["nodes"]}
    invalid_transitions = []
    is_valid = True

    for edge in G.graph_dict["edges"]:
        source_id = edge["source"]
        target_id = edge["target"]

        # Проверяем существование узлов
        if source_id not in node_map or target_id not in node_map:
            is_valid = False
            continue

        # Get utterances
        source_node = node_map[source_id]
        target_node = node_map[target_id]

        # Prepare input for validation
        input_data = {
            "source_utterances": source_node["utterances"],
            "edge_utterances": edge["utterances"],
            "target_utterances": target_node["utterances"]
        }

        # Run validation
        triplet_check_chain = triplet_validate_prompt | model | parser
        result = triplet_check_chain.invoke(input_data)

        if not result.isValid:
            is_valid = False
            invalid_transitions.append({
                "from": source_node["utterances"],
                "user": edge["utterances"],
                "to": target_node["utterances"],
                "reason": result.description
            })
    print("is_valid: ", is_valid)
    return {
        "is_valid": is_valid,
        "invalid_transitions": invalid_transitions
    }
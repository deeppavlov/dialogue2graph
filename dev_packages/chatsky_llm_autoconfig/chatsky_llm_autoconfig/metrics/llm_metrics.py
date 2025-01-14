"""
LLM Metrics.
------------

This module contains functions that checks Graphs and Dialogues for various metrics using LLM calls.
"""

import logging
import json
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
        Analyze the following dialog and determine if it stays on the expected topic.
        
        Expected Topic: {topic}
        
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

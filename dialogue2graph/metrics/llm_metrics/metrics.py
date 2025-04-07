"""
LLM Metrics.
------------

This module contains functions that checks Graphs and Dialogues for various metrics using LLM calls.
"""

import logging
import json
from typing import List, TypedDict, Union
from pydantic import BaseModel, Field
import numpy as np

from dialogue2graph.pipelines.core.graph import BaseGraph, Graph
from dialogue2graph.metrics.similarity import get_similarity
from dialogue2graph.pipelines.core.schemas import CompareResponse
from .prompts import compare_graphs_prompt, graph_example

from langchain_core.language_models.chat_models import BaseChatModel
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

# Set up logging
logging.basicConfig(level=logging.INFO)


class InvalidTransition(TypedDict):
    from_: List[str]  # Using from_ because 'from' is reserved
    user: List[str]
    to: List[str]
    reason: str


class GraphValidationResult(TypedDict):
    is_valid: bool
    invalid_transitions: List[InvalidTransition]


def are_triplets_valid(
    G: Graph, model: BaseChatModel, return_type: str = "dict"
) -> Union[dict, GraphValidationResult]:
    """
    Validates dialogue graph structure and logical transitions between nodes.

    Parameters:
        G (BaseGraph): The dialogue graph to validate
        model (BaseChatModel): The LLM model to use for validation
        return_type (str): Type of return value - either "dict" or "detailed"

    Returns:
        Union[dict, GraphValidationResult]:
            If return_type == "dict": {'value': bool, 'description': str}
            If return_type == "detailed": GraphValidationResult
    """
    if return_type not in ["dict", "detailed"]:
        raise ValueError('return_type must be either "dict" or "detailed"')

    # Define validation result model
    class TransitionValidationResult(BaseModel):
        isValid: bool = Field(description="Whether the transition is valid or not")
        description: str = Field(description="Explanation of why it's valid or invalid")

    # Create prompt template
    triplet_validate_prompt = PromptTemplate(
        input_variables=[
            "json_graph",
            "source_utterances",
            "edge_utterances",
            "target_utterances",
        ],
        template="""
    You are evaluating if dialog transitions make logical sense.
    
    Given this conversation graph in JSON:
    {json_graph}
    
    For the current transition:
    Source (Assistant): {source_utterances}
    User Response: {edge_utterances}
    Target (Assistant): {target_utterances}

    EVALUATE: Do these three set of messages form a logical sequence in the conversation?
    Consider:
    1. Does any of the assistant's first responses naturally lead to one of the user's responses?
    2. Does one of the user's responses logically connect to one of the assistant's next messages?
    3. Is the overall flow natural and coherent?

    Reply in JSON format:
    {{"isValid": true/false, "description": "Brief explanation of why it's valid or invalid"}}
    """,
    )

    parser = PydanticOutputParser(pydantic_object=TransitionValidationResult)

    # Convert graph to JSON string
    graph_json = json.dumps(G.graph_dict)

    # Create node mapping
    node_map = {node["id"]: node for node in G.graph_dict["nodes"]}
    invalid_transitions = []
    is_valid = True
    descriptions = []

    for edge in G.graph_dict["edges"]:
        source_id = edge["source"]
        target_id = edge["target"]

        if source_id not in node_map or target_id not in node_map:
            description = (
                f"Invalid edge: missing node reference {source_id} -> {target_id}"
            )
            is_valid = False
            descriptions.append(description)
            if return_type == "detailed":
                invalid_transitions.append(
                    {"from_": [], "user": [], "to": [], "reason": description}
                )
            continue

        # Get utterances
        source_node = node_map[source_id]
        target_node = node_map[target_id]

        # Prepare input for validation
        input_data = {
            "json_graph": graph_json,
            "source_utterances": source_node["utterances"],
            "edge_utterances": edge["utterances"],
            "target_utterances": target_node["utterances"],
        }

        # Run validation
        triplet_check_chain = triplet_validate_prompt | model | parser
        result = triplet_check_chain.invoke(input_data)

        if not result.isValid:
            is_valid = False
            descriptions.append(result.description)
            if return_type == "detailed":
                invalid_transitions.append(
                    {
                        "from_": source_node["utterances"],
                        "user": edge["utterances"],
                        "to": target_node["utterances"],
                        "reason": result.description,
                    }
                )

    if return_type == "dict":
        return {
            "value": is_valid,
            "description": " ".join(descriptions)
            if descriptions
            else "All transitions are valid.",
        }
    else:  # return_type == "detailed"
        return {"is_valid": is_valid, "invalid_transitions": invalid_transitions}


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


def _compare_edge_lens(G1: BaseGraph, G2: BaseGraph, max: list) -> bool:
    """Helper that compares number of utterances in each pair of edges of two nodes.
    Mapping of edges is defined by max parameter, which is argmax of embeddings of nodes utterances.
    See compare_graphs.
    Returns True if numbers match, else False.
    """
    nodes_map = {}
    graph1 = G1.graph_dict
    graph2 = G2.graph_dict
    nodes1 = [n["id"] for n in graph1["nodes"]]
    nodes2 = [n["id"] for n in graph2["nodes"]]
    for idx, n in enumerate(nodes1):
        nodes_map[n] = nodes2[max[idx]]

    for node1, node2 in zip(nodes1, [nodes_map[n] for n in nodes1]):
        edges1 = G1.get_edges_by_source(node1)
        edges2 = G2.get_edges_by_source(node2)
        if len(edges1) != len(edges2):
            return False
        for edge1 in edges1:
            for edge2 in edges2:
                if nodes_map[edge1["target"]] == edge2["target"] and len(
                    edge1["utterances"]
                ) != len(edge2["utterances"]):
                    return False
    return True


def compare_graphs(
    G1: BaseGraph,
    G2: BaseGraph,
    embedder: str = "BAAI/bge-m3",
    sim_th: float = 0.93,
    llm_comparer: str = "gpt-4o",
    formatter: str = "gpt-3.5-turbo",
    device="cuda:0",
) -> CompareResponse:
    """
    Compares two graphs via utterance embeddings similarity. If similarity is lower than `sim_th` value LLM llm_comparer is used for additional comparison.
    LLM formatter is used to keep LLM answer in a required format.
    Returns dict with True or False value and a description.
    """

    g1 = G1.graph_dict
    g2 = G2.graph_dict

    # list of concatenations of all nodes utterances:
    nodes1_list = G1.get_list_from_nodes()
    nodes2_list = G2.get_list_from_nodes()

    if len(nodes1_list) != len(nodes2_list):
        return {
            "value": False,
            "description": f"Numbers of nodes do not match: {len(nodes1_list)} != {len(nodes2_list)}",
        }

    # g1_list, g2_list - concatenations of utterances of every node and its outgoing edges
    g1_list, n_edge_utts1 = G1.get_list_from_graph()
    g2_list, n_edge_utts2 = G2.get_list_from_graph()

    nodes_matrix = get_similarity(
        nodes1_list, nodes2_list, embedder, device=device
    )  # embeddings for utterances in nodes
    mix_matrix = get_similarity(
        g1_list, g2_list, embedder, device=device
    )  # embeddings for utterances in nodes+edges

    nodes_max = list(np.argmax(nodes_matrix, axis=1))
    mix_max = list(np.argmax(mix_matrix, axis=1))
    if nodes_max != mix_max:
        return {
            "value": False,
            "description": f"Mapping for nodes {nodes_max} doesn't match mapping for nodes+edges {mix_max}",
        }
    if len(set(nodes_max)) < len(nodes1_list):
        return {
            "value": False,
            "description": "At least one of nodes corresponds to more than one in another graph",
        }

    if n_edge_utts1 != n_edge_utts2:
        return {
            "value": False,
            "description": "Graphs have different number of user's utterances",
        }
    if len(set(mix_max)) < len(g1_list):
        return {
            "value": False,
            "description": "At least one of nodes concatenated with edges corresponds to more than one in another graph",
        }

    if not _compare_edge_lens(G1, G2, mix_max):
        return {
            "value": False,
            "description": "At least one pair of edges has different number of utterances",
        }

    nodes_min = np.min(np.max(nodes_matrix, axis=1))
    mix_min = np.min(np.max(mix_matrix, axis=1))

    full_min = min(nodes_min, mix_min)

    if full_min >= sim_th:
        return {
            "value": True,
            "description": f"Nodes similarity: {nodes_min}, Nodes+edges similarity: {mix_min}",
        }

    parser = PydanticOutputParser(pydantic_object=CompareResponse)
    format_model = ChatOpenAI(model=formatter)
    model = ChatOpenAI(model=llm_comparer)
    fixed_output_parser = OutputFixingParser.from_llm(parser=parser, llm=format_model)
    chain = model | fixed_output_parser
    query = compare_graphs_prompt.format(
        result_form=CompareResponse().model_dump(),
        graph_example=graph_example,
        graph_1=g1,
        graph_2=g2,
    )
    messages = [HumanMessage(content=query)]
    return chain.invoke(messages)

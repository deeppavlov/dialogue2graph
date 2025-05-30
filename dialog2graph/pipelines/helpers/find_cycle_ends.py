"""
Find cycle ends
----------------

The module provides graph auxilary method to find cycle ends.
"""

import json
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from dialog2graph.pipelines.core.graph import Graph
from langchain_core.language_models.chat_models import BaseChatModel


def find_cycle_ends(G: Graph, cycle_ends_model: BaseChatModel) -> dict[str]:
    """
    Find nodes in a dialog graph G using conditions in graph_ends_prompt_template with the help of model.

    Parameters:
        G (BaseGraph): The dialog graph
        cycle_ends_model (BaseChatModel): The LLM to be used

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
    There is only edge with source=A in the whole graph, and target of this edge is located earlier in the dialog flow.
    
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

    find_ends_chain = graph_ends_prompt | cycle_ends_model | parser
    response = find_ends_chain.invoke(input_data)
    result = {"value": response.ends, "description": response.description}

    return result

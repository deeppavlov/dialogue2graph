import json
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from dialogue2graph.pipelines.core.graph import Graph
from langchain_core.language_models.chat_models import BaseChatModel


def find_cycle_ends(G: Graph, cycle_ends_model: BaseChatModel) -> dict[str]:
    """
    To find nodes in dialogue graph G by condition in graph_ends_prompt_template with help of model.

    Parameters:
        G (BaseGraph): The dialogue graph
        cycle_ends_model (BaseChatModel): The LLM model to be used

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

    find_ends_chain = graph_ends_prompt | cycle_ends_model | parser
    response = find_ends_chain.invoke(input_data)
    result = {"value": response.ends, "description": response.description}

    return result

from chatsky_llm_autoconfig.settings import EnvSettings

from chatsky_llm_autoconfig.algorithms.base import GraphExtender
from chatsky_llm_autoconfig.dialogue import Dialogue
from chatsky_llm_autoconfig.graph import Graph, BaseGraph
from chatsky_llm_autoconfig.schemas import DialogueGraph

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from chatsky_llm_autoconfig.autometrics.registry import AlgorithmRegistry

from chatsky_llm_autoconfig.prompts import prompt_dialogs_and_graph
from chatsky_llm_autoconfig.utils import call_llm_api

from chatsky_llm_autoconfig.metrics.automatic_metrics import (
    is_same_structure,
    compare_graphs
)

env_settings = EnvSettings()


@AlgorithmRegistry.register(input_type=list[Dialogue], path_to_result=env_settings.GENERATION_SAVE_PATH, output_type=BaseGraph)
class AppendChain(GraphExtender):
    """
    Attaches an additional dialogue to an existing original dialogue graph.

    Parameters:
        dialogues (list[Dialogue]): The list of 2 dialogues, the first one is an original dialogue and the second one is an additional dialogue.
        graph (Graph): Original dialogue graph.

    Returns:
        graph
    """
    prompt: str = ""
    def __init__(self):
        super().__init__()
        self.prompt = PromptTemplate.from_template(prompt_dialogs_and_graph)

    def invoke(self, dialogues: list[Dialogue] = None, graph: Graph = None) -> BaseGraph:
        print("model:  ",env_settings.GENERATION_MODEL_NAME)
        base_model = ChatOpenAI(model=env_settings.GENERATION_MODEL_NAME, api_key=env_settings.OPENAI_API_KEY, base_url=env_settings.OPENAI_BASE_URL, temperature=0)
        model = base_model | PydanticOutputParser(pydantic_object=DialogueGraph)

        final_prompt = self.prompt.format(
            orig_dial=dialogues[0],
            orig_graph=graph.graph_dict,
            new_dial=dialogues[1]
        )

        result = call_llm_api(final_prompt, model, temp=0)
        if result is None:
            return Graph(graph_dict={})
        
        # try:
        #     result.reason = "Fixes: " + result.reason
        # except Exception as e:
        #     print(e)

        graph_dict = result.model_dump()
        
        if not all([e['target'] for e in graph_dict['edges']]):
            return Graph(graph_dict={}), []

        result_graph = Graph(graph_dict=graph_dict)
        return result_graph

    async def ainvoke(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)
    
    async def evaluate(self, dialogues, graph, target_graph):
        result_graph = self.invoke(dialogues, graph)
        report = {
            "is_same_structure": is_same_structure(result_graph, target_graph),
            "graph_match": compare_graphs(result_graph, target_graph),
        }
        return report

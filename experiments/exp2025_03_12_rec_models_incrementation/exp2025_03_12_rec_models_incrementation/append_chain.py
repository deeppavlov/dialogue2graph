from settings import EnvSettings

from dialog2graph.pipelines.core.algorithms import GraphExtender
from dialog2graph.pipelines.core.graph import BaseGraph, Graph
from dialog2graph.pipelines.core.schemas import DialogGraph
from dialog2graph.pipelines.core.dialog import Dialog

from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser

from prompts import prompt_dialogs_and_graph
from utils import call_llm_api

# from chatsky_llm_autoconfig.metrics.automatic_metrics import (
#     is_same_structure,
#     compare_graphs
# )

from dialog2graph.metrics.no_llm_metrics import is_same_structure
from dialog2graph.metrics.llm_metrics import compare_graphs

env_settings = EnvSettings()


# @AlgorithmRegistry.register(input_type=list[Dialog], path_to_result=env_settings.GENERATION_SAVE_PATH, output_type=BaseGraph)
class AppendChain(GraphExtender):
    """
    Attaches an additional dialog to an existing original dialog graph.

    Parameters:
        dialogs (list[Dialog]): The list of 2 dialogs, the first one is an original dialog and the second one is an additional dialog.
        graph (Graph): Original dialog graph.

    Returns:
        graph
    """

    prompt: str = ""

    def __init__(self):
        super().__init__()
        self.prompt = PromptTemplate.from_template(prompt_dialogs_and_graph)

    def invoke(self, dialogs: list[Dialog] = None, graph: Graph = None) -> BaseGraph:
        print("model:  ", env_settings.GENERATION_MODEL_NAME)
        base_model = ChatOpenAI(
            model=env_settings.GENERATION_MODEL_NAME,
            api_key=env_settings.OPENAI_API_KEY,
            base_url=env_settings.OPENAI_BASE_URL,
            temperature=0,
        )
        model = base_model | PydanticOutputParser(pydantic_object=DialogGraph)

        final_prompt = self.prompt.format(
            orig_dial=dialogs[0], orig_graph=graph.graph_dict, new_dial=dialogs[1]
        )

        result = call_llm_api(final_prompt, model, temp=0)
        if result is None:
            return Graph(graph_dict={})

        graph_dict = result.model_dump()

        if not all([e["target"] for e in graph_dict["edges"]]):
            return Graph(graph_dict={}), []

        result_graph = Graph(graph_dict=graph_dict)
        return result_graph

    async def ainvoke(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)

    async def evaluate(self, dialogs, graph, target_graph):
        result_graph = self.invoke(dialogs, graph)
        report = {
            "is_same_structure": is_same_structure(result_graph, target_graph),
            "graph_match": compare_graphs(result_graph, target_graph),
        }
        return report

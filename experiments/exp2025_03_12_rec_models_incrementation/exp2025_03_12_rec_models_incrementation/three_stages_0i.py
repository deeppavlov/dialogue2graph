import pandas as pd
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser

from dialogue2graph.pipelines.core.dialogue_sampling import RecursiveDialogueSampler
from dialogue2graph.pipelines.core.algorithms import GraphGenerator
from dialogue2graph.pipelines.core.graph import BaseGraph, Graph
from dialogue2graph.pipelines.core.schemas import DialogueGraph
from dialogue2graph.pipelines.core.dialogue import Dialogue
from utils import call_llm_api
from settings import EnvSettings
from missing_edges_prompt import three_1, three_2
from prompts import part_1i0, part_2i0

# from chatsky_llm_autoconfig.metrics.automatic_metrics import (
#     is_same_structure,
#     compare_graphs
# )
from dialogue2graph.metrics.no_llm_metrics import is_same_structure
from dialogue2graph.metrics.llm_metrics import compare_graphs

env_settings = EnvSettings()
dialogue_sampler = RecursiveDialogueSampler()


# @AlgorithmRegistry.register(input_type=list[Dialogue], path_to_result=env_settings.GENERATION_SAVE_PATH, output_type=BaseGraph)
class ThreeStagesGraphGenerator(GraphGenerator):
    """Graph generator based on list of diaolgues.
    Thee stages:
    1. Algorithmic grouping assistant utterances into nodes.
    2. Algorithmic connecting nodes by edges.
    3. If one of dialogues ends with user's utterance, ask LLM to add missing edges.
    """

    prompt_name: str = ""

    def __init__(self, prompt_name: str = ""):
        super().__init__()
        self.prompt_name = prompt_name

    def invoke(
        self, dialogue: list[Dialogue] = None, graph: Graph = None, topic: str = ""
    ) -> BaseGraph:
        partial_variables = {}
        partial_variables["var_0"] = dialogue[0].to_list()
        prompt_extra = part_2i0 + " Dialogue_0: {var_0}"
        prompt = PromptTemplate(
            template=part_1i0 + "{graph}. " + prompt_extra,
            input_variables=["graph"],
            partial_variables=partial_variables,
        )
        print("model:  ", env_settings.GENERATION_MODEL_NAME)

        base_model = ChatOpenAI(
            model=env_settings.GENERATION_MODEL_NAME,
            api_key=env_settings.OPENAI_API_KEY,
            base_url=env_settings.OPENAI_BASE_URL,
        )
        model = base_model | PydanticOutputParser(pydantic_object=DialogueGraph)
        graph_dict = call_llm_api(
            prompt.format(graph=graph.graph_dict), model
        ).model_dump()

        print("RES: ", graph_dict)

        last_user = False

        try:
            if not last_user:
                result_graph = Graph(graph_dict=graph_dict)
                print("SKIP")
                return result_graph

            partial_variables = {}
            prompt_extra = ""
            for idx, dial in enumerate(dialogue):
                partial_variables[f"var_{idx}"] = dial.to_list()
                prompt_extra += f" Dialogue_{idx}: {{var_{idx}}}"
            prompt = PromptTemplate(
                template=three_1 + "{graph_dict}. " + three_2 + prompt_extra,
                input_variables=["graph_dict"],
                partial_variables=partial_variables,
            )

            print("PROMPT: ", prompt)

            model = base_model | PydanticOutputParser(pydantic_object=DialogueGraph)

            result = call_llm_api(prompt.format(graph_dict=graph_dict), model)
            if result is None:
                return Graph(graph_dict={})
            result.reason = "Fixes: " + result.reason
            graph_dict = result.model_dump()
            if not all([e["target"] for e in graph_dict["edges"]]):
                return Graph(graph_dict={}), []
            result_graph = Graph(graph_dict=graph_dict)
            return result_graph
        except Exception as e:
            print(e)
            return Graph({})

    async def ainvoke(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)

    async def evaluate(self, dialogues, target_graph, report_type="dict"):
        graph = self.invoke(dialogues)
        report = {
            "is_same_structure": is_same_structure(graph, target_graph),
            "graph_match": compare_graphs(graph, target_graph),
        }
        if report_type == "dataframe":
            report = pd.DataFrame(report, index=[0])
        elif report_type == "dict":
            return report
        else:
            raise ValueError(f"Invalid report_type: {report_type}")

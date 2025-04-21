import logging
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from typing import List
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from dialogue2graph.pipelines.core.dialogue_sampling import RecursiveDialogueSampler

from dialogue2graph.pipelines.core.algorithms import GraphGenerator
from dialogue2graph.pipelines.core.graph import BaseGraph, Graph
from dialogue2graph.pipelines.core.schemas import DialogueGraph, Node
from dialogue2graph.pipelines.core.dialogue import Dialogue
from dialogue2graph.metrics.no_llm_metrics import is_same_structure
from dialogue2graph.metrics.llm_metrics import compare_graphs
from utils import call_llm_api, nodes2graph, dialogues2list
from settings import EnvSettings
from missing_edges_prompt import three_1, three_2
from prompts import part_1i, part_2i


class DialogueNodes(BaseModel):
    nodes: List[Node] = Field(description="List of nodes representing assistant states")
    reason: str = Field(description="explanation")


env_settings = EnvSettings()
logging.getLogger("langchain_core.vectorstores.base").setLevel(logging.ERROR)
dialogue_sampler = RecursiveDialogueSampler()
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3", model_kwargs={"device": env_settings.DEVICE}
)


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
        self,
        dialogue: list[Dialogue] = None,
        graph: Graph = None,
        model_name="chatgpt-4o-latest",
        temp=0,
    ) -> tuple[BaseGraph, list[Dialogue]]:
        partial_variables = {}
        partial_variables["var_0"] = dialogue[0].to_list()
        prompt_extra = part_2i + " Dialogue_0: {var_0}"
        prompt = PromptTemplate(
            template=part_1i + "{graph}. " + prompt_extra,
            input_variables=["graph"],
            partial_variables=partial_variables,
        )

        base_model = ChatOpenAI(
            model=model_name,
            api_key=env_settings.OPENAI_API_KEY,
            base_url=env_settings.OPENAI_BASE_URL,
            temperature=temp,
        )
        model = base_model | PydanticOutputParser(pydantic_object=DialogueNodes)
        nodes = call_llm_api(
            prompt.format(graph=graph.graph_dict), model, temp=temp
        ).model_dump()

        _, _, _, _, last_user = dialogues2list(dialogue)

        for idx in range(len(nodes["nodes"])):
            nodes["nodes"][idx]["utterances"] = list(
                set(nodes["nodes"][idx]["utterances"])
            )
        print("NODES: ", nodes)
        try:
            sampled_dialogues = dialogue_sampler.invoke(graph, 1)
            graph_dict = nodes2graph(
                nodes["nodes"], dialogue + sampled_dialogues, embeddings
            )
        except Exception as e:
            print(e)
            return Graph({}), []
        print("RESULT: ", graph_dict, "\n")
        graph_dict = {
            "nodes": graph_dict["nodes"],
            "edges": graph_dict["edges"],
            "reason": "",
        }

        try:
            if not last_user:
                result_graph = Graph(graph_dict=graph_dict)
                # print("SKIP")
                return result_graph, sampled_dialogues

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

            result = call_llm_api(prompt.format(graph_dict=graph_dict), model, temp=0)
            if result is None:
                return Graph(graph_dict={}), []
            result.reason = "Fixes: " + result.reason
            graph_dict = result.model_dump()
            if not all([e["target"] for e in graph_dict["edges"]]):
                return Graph(graph_dict={}), []
            result_graph = Graph(graph_dict=graph_dict)
            return result_graph, sampled_dialogues
        except Exception as e:
            print(e)
            return Graph({}), []

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

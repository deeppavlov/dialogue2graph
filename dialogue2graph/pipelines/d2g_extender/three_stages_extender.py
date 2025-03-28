import logging
import pandas as pd
from typing import List
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from dialogue2graph.pipelines.core.dialogue_sampling import RecursiveDialogueSampler
from dialogue2graph.pipelines.core.schemas import DialogueGraph
from dialogue2graph.pipelines.d2g_algo.three_stages_algo import ThreeStagesGraphGenerator as AlgoGenerator
from dialogue2graph.pipelines.core.algorithms import GraphExtender
from dialogue2graph.pipelines.core.graph import BaseGraph, Graph
from dialogue2graph.pipelines.core.schemas import ReasonGraph, Node
from dialogue2graph.pipelines.core.dialogue import Dialogue
from dialogue2graph.metrics.no_llm_metrics import is_same_structure
from dialogue2graph.metrics.llm_metrics import compare_graphs
from dialogue2graph.utils.dg_helper import connect_nodes, get_helpers
from dialogue2graph.pipelines.helpers.prompts.missing_edges_prompt import add_edge_prompt_1, add_edge_prompt_2
from .prompts import extending_prompt_part_1, extending_prompt_part_2


class DialogueNodes(BaseModel):
    nodes: List[Node] = Field(description="List of nodes representing assistant states")
    reason: str = Field(description="explanation")


logging.getLogger("langchain_core.vectorstores.base").setLevel(logging.ERROR)
dialogue_sampler = RecursiveDialogueSampler()


class ThreeStagesGraphGenerator(GraphExtender):
    """Graph generator which iteratively takes step dialogues and adds them to graph
    generated on the previous step. First step is done with AlgoGenerator
    Three stages:
    1. LLM extension of graph nodes.
    2. Algorithmic connecting nodes by edges.
    3. If one of dialogues ends with user's utterance, ask LLM to add missing edges.
    """

    extending_llm: BaseChatModel
    filling_llm: BaseChatModel
    formatting_llm: BaseChatModel
    sim_model: HuggingFaceEmbeddings
    step: int
    algo_generator: AlgoGenerator

    def __init__(
        self, extending_llm: BaseChatModel, filling_llm: BaseChatModel, formatting_llm: BaseChatModel, sim_model: HuggingFaceEmbeddings, step: int = 2
    ):
        super().__init__(
            extending_llm=extending_llm,
            filling_llm=filling_llm,
            formatting_llm=formatting_llm,
            sim_model=sim_model,
            algo_generator=AlgoGenerator(filling_llm, formatting_llm, sim_model),
            step=step,
        )

    def _add_step(self, dialogues: list[Dialogue], graph: DialogueGraph) -> DialogueGraph:

        partial_variables = {}
        prompt_extra = extending_prompt_part_2
        for idx, dial in enumerate(dialogues):
            partial_variables[f"var_{idx}"] = dial.to_list()
            prompt_extra += f" Dialogue_{idx}: {{var_{idx}}}"
        prompt = PromptTemplate(
            template=extending_prompt_part_1 + "{graph}. " + prompt_extra,
            input_variables=["graph"],
            partial_variables=partial_variables,
        )

        fixed_output_parser = OutputFixingParser.from_llm(parser=PydanticOutputParser(pydantic_object=DialogueNodes), llm=self.formatting_llm)
        chain = self.extending_llm | fixed_output_parser

        messages = [HumanMessage(content=prompt.format(graph=graph))]
        nodes = chain.invoke(messages).model_dump()

        for idx in range(len(nodes["nodes"])):
            nodes["nodes"][idx]["utterances"] = list(set(nodes["nodes"][idx]["utterances"]))
        try:
            sampled_dialogues = dialogue_sampler.invoke(Graph(graph), 15)
            graph_dict = connect_nodes(nodes["nodes"], sampled_dialogues + dialogues, self.sim_model)
        except Exception as e:
            print(e)
            return Graph({})
        graph_dict = {"edges": graph_dict["edges"], "nodes": graph_dict["nodes"]}
        return graph_dict

    def invoke(self, dialogues: list[Dialogue]) -> BaseGraph:

        cur_graph = self.algo_generator.invoke(dialogues[: self.step]).graph_dict
        # for seq in dialogues[self.step::self.step]:
        for point in range(self.step, len(dialogues) - 1, self.step):
            cur_graph = self._add_step(dialogues[point : point + self.step], cur_graph)

        _, _, last_user = get_helpers(dialogues)
        try:
            if not last_user:
                result_graph = Graph(graph_dict=cur_graph)
                return result_graph

            partial_variables = {}
            prompt_extra = ""
            for idx, dial in enumerate(dialogues):
                partial_variables[f"var_{idx}"] = dial.to_list()
                prompt_extra += f" Dialogue_{idx}: {{var_{idx}}}"
            prompt = PromptTemplate(
                template=add_edge_prompt_1 + "{graph_dict}. " + add_edge_prompt_2 + prompt_extra,
                input_variables=["graph_dict"],
                partial_variables=partial_variables,
            )
            messages = [HumanMessage(content=prompt.format(graph_dict=cur_graph))]

            fixed_output_parser = OutputFixingParser.from_llm(parser=PydanticOutputParser(pydantic_object=ReasonGraph), llm=self.formatting_llm)
            chain = self.filling_llm | fixed_output_parser
            result = chain.invoke(messages)

            if result is None:
                return Graph(graph_dict={})
            result.reason = "Fixes: " + result.reason
            graph_dict = result.model_dump()
            if not all([e["target"] for e in graph_dict["edges"]]):
                return Graph(graph_dict={})
            result_graph = Graph(graph_dict=graph_dict)
            return result_graph
        except Exception as e:
            print(e)
            return Graph({})

    async def ainvoke(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)

    def evaluate(self, dialogues, gt_graph, report_type="dict"):
        graph = self.invoke(dialogues)
        report = {
            "is_same_structure": is_same_structure(graph, gt_graph),
            "graph_match": compare_graphs(graph, gt_graph),
        }
        if report_type == "dataframe":
            report = pd.DataFrame(report, index=[0])
        elif report_type == "dict":
            return report
        else:
            raise ValueError(f"Invalid report_type: {report_type}")

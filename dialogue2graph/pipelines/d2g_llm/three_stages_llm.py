import logging
from typing import List
import pandas as pd
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import HumanMessage

from dialogue2graph import Dialogue, Graph
from dialogue2graph.pipelines.core.algorithms import GraphGenerator
from dialogue2graph.pipelines.core.graph import BaseGraph
from dialogue2graph.pipelines.core.schemas import ReasonGraph, Node
from dialogue2graph.metrics.llm_metrics import compare_graphs
from dialogue2graph.metrics.no_llm_metrics import is_same_structure

from dialogue2graph.utils.dg_helper import connect_nodes, get_helpers
from dialogue2graph.pipelines.helpers.prompts.missing_edges_prompt import add_edge_prompt_1, add_edge_prompt_2
from .prompts import graph_example_1, grouping_prompt_1, grouping_prompt_2


class DialogueNodes(BaseModel):
    nodes: List[Node] = Field(description="List of nodes representing assistant states")
    reason: str = Field(description="explanation")


logging.getLogger("langchain_core.vectorstores.base").setLevel(logging.ERROR)


class ThreeStagesGraphGenerator(GraphGenerator):
    """Graph generator based on list of dialogues.
    Three stages:
    1. LLM grouping assistant utterances into nodes.
    2. Algorithmic connecting nodes by edges.
    3. If one of dialogues ends with user's utterance, ask LLM to add missing edges.
    """

    grouping_llm: BaseChatModel
    filling_llm: BaseChatModel
    formatting_llm: BaseChatModel
    sim_model: HuggingFaceEmbeddings

    def __init__(self, grouping_llm: BaseChatModel, filling_llm: BaseChatModel, formatting_llm: BaseChatModel, sim_model: HuggingFaceEmbeddings):
        super().__init__(grouping_llm=grouping_llm, filling_llm=filling_llm, formatting_llm=formatting_llm, sim_model=sim_model)

    def invoke(self, dialogues: list[Dialogue] = None, graph: ReasonGraph = None) -> BaseGraph:

        partial_variables = {}
        prompt_extra = grouping_prompt_2
        for idx, dial in enumerate(dialogues):
            partial_variables[f"var_{idx}"] = dial.to_list()
            prompt_extra += f" Dialogue_{idx}: {{var_{idx}}}"
        prompt = PromptTemplate(
            template=grouping_prompt_1 + "{graph_example_1}. " + prompt_extra,
            input_variables=["graph_example_1"],
            partial_variables=partial_variables,
        )

        fixed_output_parser = OutputFixingParser.from_llm(parser=PydanticOutputParser(pydantic_object=DialogueNodes), llm=self.formatting_llm)
        chain = self.grouping_llm | fixed_output_parser

        messages = [HumanMessage(content=prompt.format(graph_example_1=graph_example_1))]
        nodes = chain.invoke(messages).model_dump()

        _, _, last_user = get_helpers(dialogues)

        try:
            graph_dict = connect_nodes(nodes["nodes"], dialogues, self.sim_model)
        except Exception as e:
            print(e)
            return Graph({})
        graph_dict = {"nodes": graph_dict["nodes"], "edges": graph_dict["edges"], "reason": ""}

        try:
            if not last_user:
                result_graph = Graph(graph_dict=graph_dict)
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

            fixed_output_parser = OutputFixingParser.from_llm(parser=PydanticOutputParser(pydantic_object=ReasonGraph), llm=self.formatting_llm)
            chain = self.filling_llm | fixed_output_parser

            messages = [HumanMessage(content=prompt.format(graph_dict=graph_dict))]

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

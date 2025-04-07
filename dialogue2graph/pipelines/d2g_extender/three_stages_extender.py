import logging
import pandas as pd
from typing import List
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate
from dialogue2graph.pipelines.core.dialogue_sampling import RecursiveDialogueSampler

from dialogue2graph.pipelines.core.algorithms import GraphExtender
from dialogue2graph.pipelines.core.graph import BaseGraph, Graph
from dialogue2graph.pipelines.core.schemas import DialogueGraph, Node
from dialogue2graph.pipelines.core.dialogue import Dialogue
from dialogue2graph.pipelines.model_storage import ModelStorage
from dialogue2graph.metrics.no_llm_metrics import is_same_structure
from dialogue2graph.metrics.llm_metrics import compare_graphs
from dialogue2graph.utils.dg_helper import connect_nodes, get_helpers
from dialogue2graph.pipelines.helpers.prompts.missing_edges_prompt import (
    add_edge_prompt_1,
    add_edge_prompt_2,
)
from .prompts import extending_prompt_part_1, extending_prompt_part_2


class DialogueNodes(BaseModel):
    nodes: List[Node] = Field(description="List of nodes representing assistant states")
    reason: str = Field(description="explanation")


logging.getLogger("langchain_core.vectorstores.base").setLevel(logging.ERROR)
dialogue_sampler = RecursiveDialogueSampler()


class ThreeStagesGraphGenerator(GraphExtender):
    """Graph generator based on graph and additional list of dialogues.
    Three stages:
    1. LLM extension of graph nodes.
    2. Algorithmic connecting nodes by edges.
    3. If one of dialogues ends with user's utterance, ask LLM to add missing edges.
    """

    model_storage: ModelStorage = Field(description="Model storage")
    extending_llm: str = Field(description="LLM for extending graph nodes")
    filling_llm: str = Field(description="LLM for adding missing edges")
    formatting_llm: str = Field(description="LLM for formatting output")
    sim_model: str = Field(description="Similarity model")

    def __init__(
        self,
        model_storage: ModelStorage,
        extending_llm: str,
        filling_llm: str,
        formatting_llm: str,
        sim_model: str,
    ):
        super().__init__(
            model_storage=model_storage,
            extending_llm=extending_llm,
            filling_llm=filling_llm,
            formatting_llm=formatting_llm,
            sim_model=sim_model,
        )

    def invoke(
        self, dialogues: list[Dialogue] = None, graph: Graph = None
    ) -> BaseGraph:
        try:
            partial_variables = {}
            dialogue = dialogues[0].to_list()
            prompt = PromptTemplate(
                template=extending_prompt_part_1
                + "{graph}. "
                + extending_prompt_part_2
                + "{dialogue}",
                input_variables=["graph", "dialogue"],
            )

            fixed_output_parser = OutputFixingParser.from_llm(
                parser=PydanticOutputParser(pydantic_object=DialogueNodes),
                llm=self.model_storage.storage[self.formatting_llm].model,
            )
            chain = (
                self.model_storage.storage[self.extending_llm].model
                | fixed_output_parser
            )

            messages = [
                HumanMessage(
                    content=prompt.format(graph=graph.graph_dict, dialogue=dialogue)
                )
            ]
            nodes = chain.invoke(messages).model_dump()

            _, _, last_user = get_helpers(dialogues)

            for idx in range(len(nodes["nodes"])):
                nodes["nodes"][idx]["utterances"] = list(
                    set(nodes["nodes"][idx]["utterances"])
                )
            try:
                sampled_dialogues = dialogue_sampler.invoke(graph, 15)
                graph_dict = connect_nodes(
                    nodes["nodes"],
                    sampled_dialogues,
                    self.model_storage.storage[self.sim_model].model,
                )
            except Exception as e:
                print(e)
                return Graph({})
            graph_dict = {
                "nodes": graph_dict["nodes"],
                "edges": graph_dict["edges"],
                "reason": "",
            }

            if not last_user:
                result_graph = Graph(graph_dict=graph_dict)
                return result_graph

            partial_variables = {}
            prompt_extra = ""
            for idx, dial in enumerate(dialogues):
                partial_variables[f"var_{idx}"] = dial.to_list()
                prompt_extra += f" Dialogue_{idx}: {{var_{idx}}}"
            prompt = PromptTemplate(
                template=add_edge_prompt_1
                + "{graph_dict}. "
                + add_edge_prompt_2
                + prompt_extra,
                input_variables=["graph_dict"],
                partial_variables=partial_variables,
            )
            messages = [HumanMessage(content=prompt.format(graph_dict=graph_dict))]

            fixed_output_parser = OutputFixingParser.from_llm(
                parser=PydanticOutputParser(pydantic_object=DialogueGraph),
                llm=self.model_storage.storage[self.formatting_llm].model,
            )
            chain = (
                self.model_storage.storage[self.filling_llm].model | fixed_output_parser
            )
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

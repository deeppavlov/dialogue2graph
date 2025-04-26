"""
Three Stage LightGraphGenerator
-------------------------------

The module provides three step algorithm aimed to generate dialog graph.
"""

import logging
from pydantic import Field
from typing import Callable
from pydantic import ConfigDict
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings


from dialogue2graph import metrics
from dialogue2graph import Graph
from dialogue2graph.pipelines.core.d2g_generator import DGBaseGenerator
from dialogue2graph.pipelines.core.schemas import ReasonGraph
from dialogue2graph.pipelines.core.graph import BaseGraph
from dialogue2graph.pipelines.model_storage import ModelStorage
from dialogue2graph.pipelines.d2g_light.group_nodes import group_nodes
from dialogue2graph.utils.dg_helper import connect_nodes, get_helpers
from dialogue2graph.pipelines.helpers.parse_data import PipelineDataType
from dialogue2graph.pipelines.helpers.prompts.missing_edges_prompt import (
    add_edge_prompt_1,
    add_edge_prompt_2,
)

logging.getLogger("langchain_core.vectorstores.base").setLevel(logging.ERROR)


class LightGraphGenerator(DGBaseGenerator):
    """Graph generator from list of dialogues. Based on algorithm with embedding similarity usage.

    Attributes:
        model_storage: Model storage
        filling_llm: Name of LLM for adding missing edges
        formatting_llm: Name of LLM for formatting other LLMs output
        sim_model: HuggingFace name for similarity model
        step2_evals: Evaluation metrics called after stage 2 with connecting nodes by edges
        end_evals: Evaluation metrics called at the end of generation process
        model_config: It's a parameter for internal use of Pydantic
    """

    model_storage: ModelStorage = Field(description="Model storage")
    filling_llm: str = Field(
        description="LLM for adding missing edges",
        default="three_stages_light_filling_llm:v1",
    )
    formatting_llm: str = Field(
        description="LLM for formatting output",
        default="three_stages_light_formatting_llm:v1",
    )
    sim_model: str = Field(
        description="Similarity model", default="three_stages_light_sim_model:v1"
    )
    step2_evals: list[Callable] = Field(
        default_factory=list, description="Metrics after stage 2"
    )
    end_evals: list[Callable] = Field(
        default_factory=list, description="Metrics at the end"
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        model_storage: ModelStorage,
        filling_llm: str = "three_stages_light_filling_llm:v1",
        formatting_llm: str = "three_stages_light_formatting_llm:v1",
        sim_model: str = "three_stages_light_sim_model:v1",
        step2_evals: list[Callable] | None = [],
        end_evals: list[Callable] | None = [],
    ):
        # if model is not in model storage put the default model there
        model_storage.add(
            key=filling_llm,
            config={"model_name": "chatgpt-4o-latest", "temperature": 0},
            model_type=ChatOpenAI,
        )

        model_storage.add(
            key=formatting_llm,
            config={"model_name": "gpt-4o-mini", "temperature": 0},
            model_type=ChatOpenAI,
        )

        model_storage.add(
            key=sim_model,
            config={"model_name": "BAAI/bge-m3", "model_kwargs": {"device": "cpu"}},
            # config={"model_name": "BAAI/bge-m3", "config_kwargs": {"load_in_8bit": True, "torch_dtype": torch.float16}},
            # config={"model_name": "BAAI/bge-m3", "model_kwargs": {"device": "cpu", "torch_dtype": torch.float16}}, "low_cpu_mem_usage": True,

            model_type=HuggingFaceEmbeddings,
        )

        super().__init__(
            model_storage=model_storage,
            filling_llm=filling_llm,
            formatting_llm=formatting_llm,
            sim_model=sim_model,
            step2_evals=step2_evals,
            end_evals=end_evals,
        )

    def invoke(
        self, pipeline_data: PipelineDataType, enable_evals: bool = False
    ) -> tuple[BaseGraph, metrics.DGReportType]:
        """Invoke efficient implementation of the three stages generation algorithm."""

        node_utts, start_utts, user_end = get_helpers(pipeline_data.dialogs)
        groups = group_nodes(pipeline_data.dialogs, node_utts)

        nodes = [
            {
                "id": idx + 1,
                "label": "",
                "is_start": any(gr in start_utts for gr in group),
                "utterances": group,
            }
            for idx, group in enumerate(groups)
        ]

        graph_dict = connect_nodes(
            nodes,
            pipeline_data.dialogs,
            self.model_storage.storage[self.sim_model].model,
        )
        graph_dict.update(reason="")

        result_graph = Graph(graph_dict=graph_dict)
        report = {}

        if enable_evals and pipeline_data.true_graph:
            report = self.evaluate(result_graph, pipeline_data.true_graph, "step2")

        if user_end:
            partial_variables = {
                f"var_{idx}": dial.to_list()
                for idx, dial in enumerate(pipeline_data.dialogs)
            }
            prompt_extra = " ".join(
                f"Dialogue_{idx}: {{var_{idx}}}"
                for idx in range(len(pipeline_data.dialogs))
            )
            prompt = PromptTemplate(
                template=f"{add_edge_prompt_1}{{graph_dict}}. {add_edge_prompt_2}{prompt_extra}",
                input_variables=["graph_dict"],
                partial_variables=partial_variables,
            )

            fixed_output_parser = OutputFixingParser.from_llm(
                parser=PydanticOutputParser(pydantic_object=ReasonGraph),
                llm=self.model_storage.storage[self.formatting_llm].model,
            )
            chain = (
                self.model_storage.storage[self.filling_llm].model | fixed_output_parser
            )

            messages = [HumanMessage(content=prompt.format(graph_dict=graph_dict))]
            result = chain.invoke(messages)
            if result:
                result.reason = "Fixes: " + result.reason
                graph_dict = result.model_dump()
                if all(e["target"] for e in graph_dict["edges"]):
                    result_graph = Graph(graph_dict=graph_dict)
                    if enable_evals and pipeline_data.true_graph:
                        report.update(
                            self.evaluate(result_graph, pipeline_data.true_graph, "end")
                        )
            else:
                return Graph(graph_dict={}), report

        return result_graph, report

    async def ainvoke(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)

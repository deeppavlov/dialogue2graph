"""
Three Stage LLMGraphGenerator
-------------------------------

The module provides three step algorithm aimed to generate dialog graph and based on LLMs.
"""

import logging
from datetime import datetime
from typing import List, Callable
from pydantic import ConfigDict
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


from dialog2graph import metrics
from dialog2graph import Graph
from dialog2graph.pipelines.core.d2g_generator import DGBaseGenerator
from dialog2graph.pipelines.core.graph import BaseGraph, Metadata
from dialog2graph.pipelines.core.schemas import ReasonGraph, Node
from dialog2graph.pipelines.model_storage import ModelStorage
from dialog2graph.utils.logger import Logger
from dialog2graph.utils.dg_helper import connect_nodes, get_helpers
from dialog2graph.pipelines.helpers.parse_data import PipelineDataType
from dialog2graph.pipelines.helpers.prompts.missing_edges_prompt import (
    add_edge_prompt_1,
    add_edge_prompt_2,
)
from dialog2graph.pipelines.d2g_llm.prompts import (
    graph_example_1,
    grouping_prompt_1,
    grouping_prompt_2,
)


class DialogNodes(BaseModel):
    """Class for dialog nodes"""

    nodes: List[Node] = Field(description="List of nodes representing assistant states")
    reason: str = Field(description="explanation")


logging.getLogger("langchain_core.vectorstores.base").setLevel(logging.ERROR)
logger = Logger(__file__)


class LLMGraphGenerator(DGBaseGenerator):
    """Graph generator from list of dialogs. Based on LLM.
    Three stages:

    1. LLM grouping assistant utterances into nodes.
    2. Algorithmic connecting nodes by edges.
    3. If one of dialogs ends with user's utterance, ask LLM to add missing edges.

    Attributes:
        model_storage: Model storage
        grouping_llm: Name of LLM for grouping assistant utterances into nodes
        filling_llm: Name of LLM for adding missing edges
        formatting_llm: Name of LLM for formatting other LLMs output
        sim_model: HuggingFace name for similarity model
        step2_evals: Evaluation metrics called after stage 2 with connecting nodes by edges
        end_evals: Evaluation metrics called at the end of generation process
        model_config: It's a parameter for internal use of Pydantic
    """

    model_storage: ModelStorage = Field(description="Model storage")
    grouping_llm: str = Field(
        description="LLM for grouping assistant utterances into nodes",
        default="three_stages_grouping_llm:v1",
    )
    filling_llm: str = Field(
        description="LLM for adding missing edges",
        default="three_stages_filling_llm:v1",
    )
    formatting_llm: str = Field(
        description="LLM for formatting output",
        default="three_stages_formatting_llm:v1",
    )
    sim_model: str = Field(
        description="Similarity model", default="three_stages_sim_model:v1"
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
        grouping_llm: str = "three_stages_llm_grouping_llm:v1",
        filling_llm: str = "three_stages_llm_filling_llm:v1",
        formatting_llm: str = "three_stages_llm_formatting_llm:v1",
        sim_model: str = "three_stages_llm_sim_model:v1",
        step2_evals: list[Callable] | None = None,
        end_evals: list[Callable] | None = None,
    ):
        if step2_evals is None:
            step2_evals = []
        if end_evals is None:
            end_evals = []

        # if model is not in model storage put the default model there
        model_storage.add(
            key=grouping_llm,
            config={"model_name": "chatgpt-4o-latest", "temperature": 0},
            model_type=ChatOpenAI,
        )

        model_storage.add(
            key=filling_llm,
            config={"model_name": "o3-mini", "temperature": 1},
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
            model_type=HuggingFaceEmbeddings,
        )

        super().__init__(
            model_storage=model_storage,
            grouping_llm=grouping_llm,
            filling_llm=filling_llm,
            formatting_llm=formatting_llm,
            sim_model=sim_model,
            step2_evals=step2_evals,
            end_evals=end_evals,
        )

    def invoke(
        self, pipeline_data: PipelineDataType, enable_evals: bool = False
    ) -> tuple[BaseGraph, metrics.DGReportType]:
        """Invoke primary method of the three stages generation algorithm:

        1. Grouping assistant utterances into nodes with LLM.
        2. Algorithmic connecting nodes by edges: connect_nodes.
        3. If one of dialogs ends with user's utterance, ask LLM to add missing edges.

        Args:
            pipeline_data:
            data for generation and evaluation:
                dialogs for generation, of List[Dialog] type
                true_graph for evaluation, of Graph type
            enable_evals: when true, evaluate method is called
        Returns:
            tuple of resulted graph of Graph type and report dictionary like in example below:
            {'value': False, 'description': 'Numbers of nodes do not match: 7 != 8'}
        """
        metadata = Metadata(
            generator_name="d2g_llm",
            models_config=self.model_storage.model_dump(),
            schema_version="v1",
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        try:
            dialogs = ""
            for idx, dial in enumerate(pipeline_data.dialogs):
                dialogs += f" Dialog_{idx}: {dial}"

            prompt = PromptTemplate.from_template(
                grouping_prompt_1 + "{graph_example_1}. " + grouping_prompt_2 + dialogs
            )
            fixed_output_parser = OutputFixingParser.from_llm(
                parser=PydanticOutputParser(pydantic_object=DialogNodes),
                llm=self.model_storage.storage[self.formatting_llm].model,
            )
            chain = (
                self.model_storage.storage[self.grouping_llm].model
                | fixed_output_parser
            )
            # Use LLM to group nodes
            llm_output = chain.invoke(
                [HumanMessage(content=prompt.format(graph_example_1=graph_example_1))]
            )
            nodes = [node.model_dump() for node in llm_output.nodes]
        except Exception as e:
            logger.error("Error in step1: %s", e)
            return Graph(graph_dict={}, metadata=metadata), {}
        try:
            # Connect nodes
            graph_dict = connect_nodes(
                nodes,
                pipeline_data.dialogs,
                self.model_storage.storage[self.sim_model].model,
            )
            graph_dict["reason"] = ""

            # Evaluate if needed
            result_graph = Graph(graph_dict=graph_dict, metadata=metadata)
            report = (
                self.evaluate(result_graph, pipeline_data.true_graph, "step2")
                if enable_evals and pipeline_data.true_graph
                else {}
            )
        except Exception as e:
            logger.error("Error in step2: %s", e)
            return Graph(graph_dict={}, metadata=metadata), {}
        try:
            # Handle user end dialogs
            if get_helpers(pipeline_data.dialogs)[2]:
                prompt = PromptTemplate.from_template(
                    add_edge_prompt_1 + "{graph_dict}. " + add_edge_prompt_2
                )

                fixed_output_parser = OutputFixingParser.from_llm(
                    parser=PydanticOutputParser(pydantic_object=ReasonGraph),
                    llm=self.model_storage.storage[self.formatting_llm].model,
                )
                chain = (
                    self.model_storage.storage[self.filling_llm].model
                    | fixed_output_parser
                )
                result = chain.invoke(
                    [HumanMessage(content=prompt.format(graph_dict=graph_dict))]
                )

                result.reason = "Fixes: " + result.reason
                graph_dict = result.model_dump()

                # Validate edges
                if not all(e.get("target") for e in graph_dict["edges"]):
                    return Graph(graph_dict={}, metadata=metadata), report

                result_graph = Graph(graph_dict=graph_dict, metadata=metadata)
                if enable_evals and pipeline_data.true_graph:
                    report.update(
                        self.evaluate(result_graph, pipeline_data.true_graph, "end")
                    )

            return result_graph, report
        except Exception as e:
            logger.error("Error in step3: %s", e)
            return Graph(graph_dict={}, metadata=metadata), {}

    async def ainvoke(self, *args, **kwargs):  # pragma: no cover
        return self.invoke(*args, **kwargs)

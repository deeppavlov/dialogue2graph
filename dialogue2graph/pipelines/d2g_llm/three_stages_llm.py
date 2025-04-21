"""
Three Stage LLMGraphGenerator
-------------------------------

The module provides three step algorithm aimed to generate dialog graph and based on LLMs.
"""

import logging
from typing import List, Callable
from pydantic import ConfigDict
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.schema import HumanMessage

from dialogue2graph import metrics
from dialogue2graph import Graph
from dialogue2graph.pipelines.core.algorithms import GraphGenerator
from dialogue2graph.pipelines.core.graph import BaseGraph
from dialogue2graph.pipelines.core.schemas import ReasonGraph, Node
from dialogue2graph.pipelines.model_storage import ModelStorage


from dialogue2graph.utils.dg_helper import connect_nodes, get_helpers
from dialogue2graph.pipelines.helpers.parse_data import PipelineDataType
from dialogue2graph.pipelines.helpers.prompts.missing_edges_prompt import (
    add_edge_prompt_1,
    add_edge_prompt_2,
)
from dialogue2graph.pipelines.d2g_llm.prompts import (
    graph_example_1,
    grouping_prompt_1,
    grouping_prompt_2,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DialogueNodes(BaseModel):
    """Class for dialog nodes"""

    nodes: List[Node] = Field(description="List of nodes representing assistant states")
    reason: str = Field(description="explanation")


logging.getLogger("langchain_core.vectorstores.base").setLevel(logging.ERROR)


class LLMGraphGenerator(GraphGenerator):
    """Graph generator from list of dialogues. Based on LLM.
    Three stages:

    1. LLM grouping assistant utterances into nodes.
    2. Algorithmic connecting nodes by edges.
    3. If one of dialogues ends with user's utterance, ask LLM to add missing edges.

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
        grouping_llm: str,
        filling_llm: str,
        formatting_llm: str,
        sim_model: str,
        step2_evals: list[Callable] | None = None,
        end_evals: list[Callable] | None = None,
    ):
        if step2_evals is None:
            step2_evals = []
        if end_evals is None:
            end_evals = []

        # check if models are in model storage
        # if model is not in model storage put the default model there
        if grouping_llm not in model_storage.storage:
            model_storage.add(
                key=grouping_llm,
                config={"model": "gpt-4o-latest", "temperature": 0},
                model_type="llm",
            )

        if filling_llm not in model_storage.storage:
            model_storage.add(
                key=filling_llm,
                config={"model": "o3-mini", "temperature": 1},
                model_type="llm",
            )

        if formatting_llm not in model_storage.storage:
            model_storage.add(
                key=formatting_llm,
                config={"model": "gpt-4o-mini", "temperature": 0},
                model_type="llm",
            )

        if sim_model not in model_storage.storage:
            model_storage.add(
                key=sim_model,
                config={"model_name": "BAAI/bge-m3", "device": "cpu"},
                model_type="emb",
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
        3. If one of dialogues ends with user's utterance, ask LLM to add missing edges.

        Args:
            pipeline_data:
            data for generation and evaluation:
                dialogs for generation, of List[Dialogue] type
                true_graph for evaluation, of Graph type
            enable_evals: when true, evaluate method is called
        Returns:
            tuple of resulted graph of Graph type and report dictionary like in example below:
            {'value': False, 'description': 'Numbers of nodes do not match: 7 != 8'}
        """
        try:
            dialogs = ""
            for idx, dial in enumerate(pipeline_data.dialogs):
                dialogs += f" Dialogue_{idx}: {dial}"

            prompt = PromptTemplate.from_template(
                grouping_prompt_1 + "{graph_example_1}. " + grouping_prompt_2 + dialogs
            )
            fixed_output_parser = OutputFixingParser.from_llm(
                parser=PydanticOutputParser(pydantic_object=DialogueNodes),
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
            # Connect nodes
            graph_dict = connect_nodes(
                nodes,
                pipeline_data.dialogs,
                self.model_storage.storage[self.sim_model].model,
            )
            graph_dict["reason"] = ""

            # Evaluate if needed
            result_graph = Graph(graph_dict=graph_dict)
            report = (
                self.evaluate(result_graph, pipeline_data.true_graph, "step2")
                if enable_evals and pipeline_data.true_graph
                else {}
            )

            # Handle user end dialogues
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
                    return Graph(graph_dict={}), report

                result_graph = Graph(graph_dict=graph_dict)
                if enable_evals and pipeline_data.true_graph:
                    report.update(
                        self.evaluate(result_graph, pipeline_data.true_graph, "end")
                    )

            return result_graph, report
        except Exception as e:
            logger.error("Error in step3: %s", e)
            return Graph({}), {}

    async def ainvoke(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)

    def evaluate(self, graph, gt_graph, eval_stage: str) -> metrics.DGReportType:
        report = {}
        for metric in getattr(self, eval_stage + "_evals"):
            report[metric.__name__ + ":" + eval_stage] = metric(graph, gt_graph)
        return report

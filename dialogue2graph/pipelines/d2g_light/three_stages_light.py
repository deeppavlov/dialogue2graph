import logging
from pydantic import Field
from pydantic import ConfigDict
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.schema import HumanMessage

from dialogue2graph import metrics
from dialogue2graph.pipelines.core.algorithms import GraphGenerator
from dialogue2graph.pipelines.core.schemas import ReasonGraph
from dialogue2graph import Graph
from dialogue2graph.pipelines.core.graph import BaseGraph
from dialogue2graph.pipelines.model_storage import ModelStorage
from .group_nodes import group_nodes
from dialogue2graph.utils.dg_helper import connect_nodes, get_helpers
from dialogue2graph.pipelines.helpers.parse_data import PipelineDataType
from dialogue2graph.pipelines.helpers.prompts.missing_edges_prompt import add_edge_prompt_1, add_edge_prompt_2

logging.getLogger("langchain_core.vectorstores.base").setLevel(logging.ERROR)


class LightGraphGenerator(GraphGenerator):
    """Graph generator from list of dialogues. Based on algorithm with embedding similarity usage.

    Attributes:
        model_config: It's a parameter for internal use of Pydantic
        model_storage: Model storage
        filling_llm: Name of LLM for adding missing edges
        formatting_llm: Name of LLM for formatting other LLMs output
        sim_model: HuggingFace name for similarity model
        step2_evals: Evaluation metrics called after stage 2 with connecting nodes by edges
        end_evals: Evaluation metrics called at the end of generation process")
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    model_storage: ModelStorage = Field(description="Model storage")
    filling_llm: str = Field(description="LLM for adding missing edges")
    formatting_llm: str = Field(description="LLM for formatting output")
    sim_model: str = Field(description="Similarity model")
    step2_evals: list[callable] = Field(default_factory=list, description="Metrics after stage 2")
    end_evals: list[callable] = Field(default_factory=list, description="Metrics at the end")

    def __init__(
        self,
        model_storage: ModelStorage,
        filling_llm: str,
        formatting_llm: str,
        sim_model: str,
        step2_evals: list[callable] | None = None,
        end_evals: list[callable] | None = None,
    ):
        if step2_evals is None:
            step2_evals = []
        if end_evals is None:
            end_evals = []
        super().__init__(
            model_storage=model_storage,
            filling_llm=filling_llm,
            formatting_llm=formatting_llm,
            sim_model=sim_model,
            step2_evals=step2_evals,
            end_evals=end_evals,
        )

    def invoke(self, pipeline_data: PipelineDataType, enable_evals: bool = False) -> tuple[BaseGraph, metrics.DGReportType]:
        """Primary method of the three stages generation algorithm:
        1. Algorithmic grouping assistant utterances into nodes: group_nodes.
        2. Algorithmic connecting nodes by edges: connect_nodes.
        3. If one of dialogues ends with user's utterance, ask LLM to add missing edges.

        Args:
          pipeline_data:
            data for generation and evaluation:
              dialogs for generation, of List[Dialogue] type
              true_graph for evaluation, of Graph type
        Returns:
          tuple of resulted graph of Graph type and report dictionary like in example below:
          {'value': False, 'description': 'Numbers of nodes do not match: 7 != 8'}
        Raises:

        """

        nodes, starts, last_user = get_helpers(pipeline_data.dialogs)

        groups = group_nodes(pipeline_data.dialogs, nodes)

        nodes = []
        for idx, group in enumerate(groups):
            if any([gr in starts for gr in group]):
                start = True
            else:
                start = False
            nodes.append({"id": idx + 1, "label": "", "is_start": start, "utterances": group})

        graph_dict = connect_nodes(nodes, pipeline_data.dialogs, self.model_storage.storage[self.sim_model].model)
        graph_dict = {"nodes": graph_dict["nodes"], "edges": graph_dict["edges"], "reason": ""}

        result_graph = Graph(graph_dict=graph_dict)
        if enable_evals and pipeline_data.true_graph is not None:
            report = self.evaluate(result_graph, pipeline_data.true_graph, "step2")
        else:
            report = {}

        if not last_user:
            return result_graph, report

        partial_variables = {}
        prompt_extra = ""
        for idx, dial in enumerate(pipeline_data.dialogs):
            partial_variables[f"var_{idx}"] = dial.to_list()
            prompt_extra += f" Dialogue_{idx}: {{var_{idx}}}"
        prompt = PromptTemplate(
            template=add_edge_prompt_1 + "{graph_dict}. " + add_edge_prompt_2 + prompt_extra,
            input_variables=["graph_dict"],
            partial_variables=partial_variables,
        )

        fixed_output_parser = OutputFixingParser.from_llm(
            parser=PydanticOutputParser(pydantic_object=ReasonGraph), llm=self.model_storage.storage[self.formatting_llm].model
        )
        chain = self.model_storage.storage[self.filling_llm].model | fixed_output_parser

        messages = [HumanMessage(content=prompt.format(graph_dict=graph_dict))]
        result = chain.invoke(messages)
        if result is None:
            return Graph(graph_dict={}), report
        result.reason = "Fixes: " + result.reason
        graph_dict = result.model_dump()
        if not all([e["target"] for e in graph_dict["edges"]]):
            return Graph(graph_dict={}), report
        result_graph = Graph(graph_dict=graph_dict)
        if enable_evals and pipeline_data.true_graph is not None:
            report.update(self.evaluate(result_graph, pipeline_data.true_graph, "end"))
        return result_graph, report

    async def ainvoke(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)

    def evaluate(self, graph, gt_graph, eval_stage: str) -> metrics.DGReportType:

        report = {}
        for metric in getattr(self, eval_stage + "_evals"):
            report[metric.__name__ + ":" + eval_stage] = metric(graph, gt_graph)
        return report

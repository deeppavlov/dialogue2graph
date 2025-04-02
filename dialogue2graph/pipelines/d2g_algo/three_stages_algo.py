import logging
from pydantic import ConfigDict
from langchain.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import HumanMessage

from dialogue2graph import metrics
from dialogue2graph.pipelines.core.algorithms import GraphGenerator
from dialogue2graph.pipelines.core.schemas import ReasonGraph
from dialogue2graph import Graph
from dialogue2graph.pipelines.core.graph import BaseGraph

from .group_nodes import group_nodes
from dialogue2graph.utils.dg_helper import connect_nodes, get_helpers
from dialogue2graph.pipelines.helpers.parse_data import PipelineDataType
from dialogue2graph.pipelines.helpers.prompts.missing_edges_prompt import add_edge_prompt_1, add_edge_prompt_2

logging.getLogger("langchain_core.vectorstores.base").setLevel(logging.ERROR)


class ThreeStagesGraphGenerator(GraphGenerator):
    """Graph generator based on list of dialogues.
    Three stages:
    1. Algorithmic grouping assistant utterances into nodes.
    2. Algorithmic connecting nodes by edges.
    3. If one of dialogues ends with user's utterance, ask LLM to add missing edges.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    filling_llm: BaseChatModel
    formatting_llm: BaseChatModel
    sim_model: HuggingFaceEmbeddings
    step2_evals: list[callable]
    end_evals: list[callable]

    def __init__(
        self,
        filling_llm: BaseChatModel,
        formatting_llm: BaseChatModel,
        sim_model: HuggingFaceEmbeddings,
        step2_evals: list[callable],
        end_evals: list[callable],
    ):
        super().__init__(filling_llm=filling_llm, formatting_llm=formatting_llm, sim_model=sim_model, step2_evals=step2_evals, end_evals=end_evals)

    def invoke(self, pipeline_data: PipelineDataType, enable_evals: bool = False) -> tuple[BaseGraph, metrics.DGReportType]:

        nodes, starts, last_user = get_helpers(pipeline_data.dialogs)

        groups = group_nodes(pipeline_data.dialogs, nodes)

        nodes = []
        for idx, group in enumerate(groups):
            if any([gr in starts for gr in group]):
                start = True
            else:
                start = False
            nodes.append({"id": idx + 1, "label": "", "is_start": start, "utterances": group})

        graph_dict = connect_nodes(nodes, pipeline_data.dialogs, self.sim_model)
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

        fixed_output_parser = OutputFixingParser.from_llm(parser=PydanticOutputParser(pydantic_object=ReasonGraph), llm=self.formatting_llm)
        chain = self.filling_llm | fixed_output_parser

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

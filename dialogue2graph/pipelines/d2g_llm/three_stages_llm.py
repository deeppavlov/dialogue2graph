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
    nodes: List[Node] = Field(description="List of nodes representing assistant states")
    reason: str = Field(description="explanation")


logging.getLogger("langchain_core.vectorstores.base").setLevel(logging.ERROR)


class LLMGraphGenerator(GraphGenerator):
    """Graph generator based on list of dialogues.
    Three stages:
    1. LLM grouping assistant utterances into nodes.
    2. Algorithmic connecting nodes by edges.
    3. If one of dialogues ends with user's utterance, ask LLM to add missing edges.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    model_storage: ModelStorage = Field(description="Model storage")
    grouping_llm: str = Field(
        description="LLM for grouping assistant utterances into nodes"
    )
    filling_llm: str = Field(description="LLM for adding missing edges")
    formatting_llm: str = Field(description="LLM for formatting output")
    sim_model: str = Field(description="Similarity model")
    step2_evals: list[Callable] = Field(
        default_factory=list, description="Metrics after stage 2"
    )
    end_evals: list[Callable] = Field(
        default_factory=list, description="Metrics at the end"
    )

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
        partial_variables = {}
        prompt_extra = grouping_prompt_2
        for idx, dial in enumerate(pipeline_data.dialogs):
            partial_variables[f"var_{idx}"] = dial.to_list()
            prompt_extra += f" Dialogue_{idx}: {{var_{idx}}}"
        prompt = PromptTemplate(
            template=grouping_prompt_1 + "{graph_example_1}. " + prompt_extra,
            input_variables=["graph_example_1"],
            partial_variables=partial_variables,
        )

        fixed_output_parser = OutputFixingParser.from_llm(
            parser=PydanticOutputParser(pydantic_object=DialogueNodes),
            llm=self.model_storage.storage[self.formatting_llm].model,
        )
        chain = (
            self.model_storage.storage[self.grouping_llm].model | fixed_output_parser
        )

        messages = [
            HumanMessage(content=prompt.format(graph_example_1=graph_example_1))
        ]
        nodes = chain.invoke(messages).model_dump()

        _, _, last_user = get_helpers(pipeline_data.dialogs)

        try:
            graph_dict = connect_nodes(
                nodes["nodes"],
                pipeline_data.dialogs,
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

        try:
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
                template=add_edge_prompt_1
                + "{graph_dict}. "
                + add_edge_prompt_2
                + prompt_extra,
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
            if result is None:
                return Graph(graph_dict={}), report
            result.reason = "Fixes: " + result.reason
            graph_dict = result.model_dump()
            if not all([e["target"] for e in graph_dict["edges"]]):
                return Graph(graph_dict={}), report
            result_graph = Graph(graph_dict=graph_dict)
            if enable_evals and pipeline_data.true_graph is not None:
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

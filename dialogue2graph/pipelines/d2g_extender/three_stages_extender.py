import logging
from typing import List, Callable
from pydantic import ConfigDict
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate

from dialogue2graph import metrics
from dialogue2graph.pipelines.core.dialogue_sampling import RecursiveDialogueSampler
from dialogue2graph.pipelines.d2g_light.three_stages_light import LightGraphGenerator
from dialogue2graph.pipelines.core.algorithms import GraphExtender
from dialogue2graph.pipelines.core.graph import BaseGraph, Graph
from dialogue2graph.pipelines.core.schemas import ReasonGraph, Node
from dialogue2graph.pipelines.core.dialogue import Dialogue
from dialogue2graph.pipelines.model_storage import ModelStorage
from dialogue2graph.utils.dg_helper import connect_nodes, get_helpers
from dialogue2graph.pipelines.helpers.parse_data import PipelineDataType
from dialogue2graph.pipelines.helpers.prompts.missing_edges_prompt import (
      add_edge_prompt_1, add_edge_prompt_2
)
from dialogue2graph.pipelines.d2g_extender.prompts import (
      extending_prompt_part_1, extending_prompt_part_2
)


class DialogueNodes(BaseModel):
    nodes: List[Node] = Field(description="List of nodes representing assistant states")
    # reason: str = Field(description="explanation")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("langchain_core.vectorstores.base").setLevel(logging.ERROR)
dialogue_sampler = RecursiveDialogueSampler()


class LLMGraphExtender(GraphExtender):
    """Graph generator which iteratively takes step dialogues and adds them to graph
    generated on the previous step. First step is done with AlgoGenerator
    Three stages:
    1. LLM extension of graph nodes.
    2. Algorithmic connecting nodes by edges.
    3. If one of dialogues ends with user's utterance, ask LLM to add missing edges.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    model_storage: ModelStorage = Field(description="Model storage")
    extending_llm: str = Field(description="LLM for extending graph nodes")
    filling_llm: str = Field(description="LLM for adding missing edges")
    formatting_llm: str = Field(description="LLM for formatting output")
    sim_model: str = Field(description="Similarity model")
    step: int
    graph_generator: LightGraphGenerator
    step1_evals: list[Callable]
    extender_evals: list[Callable]
    step2_evals: list[Callable]
    end_evals: list[Callable]

    def __init__(
        self,
        model_storage: ModelStorage,
        extending_llm: str,
        filling_llm: str,
        formatting_llm: str,
        sim_model: str,
        step1_evals: list[Callable],
        extender_evals: list[Callable],
        step2_evals: list[Callable],
        end_evals: list[Callable],
        step: int = 2,
    ):
        super().__init__(
            model_storage=model_storage,
            extending_llm=extending_llm,
            filling_llm=filling_llm,
            formatting_llm=formatting_llm,
            sim_model=sim_model,
            graph_generator=LightGraphGenerator(model_storage, filling_llm, formatting_llm, sim_model, step2_evals, end_evals),
            step1_evals=step1_evals,
            extender_evals=extender_evals,
            step2_evals=step2_evals,
            end_evals=end_evals,
            step=step,
        )

    def _add_step(self, dialogues: list[Dialogue], graph: Graph) -> Graph:

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

        fixed_output_parser = OutputFixingParser.from_llm(
            parser=PydanticOutputParser(pydantic_object=DialogueNodes), llm=self.model_storage.storage[self.formatting_llm].model
        )
        chain = self.model_storage.storage[self.extending_llm].model | fixed_output_parser

        messages = [HumanMessage(content=prompt.format(graph=graph.graph_dict))]
        nodes = chain.invoke(messages).model_dump()

        for idx in range(len(nodes["nodes"])):
            nodes["nodes"][idx]["utterances"] = list(set(nodes["nodes"][idx]["utterances"]))
        try:
            sampled_dialogues = dialogue_sampler.invoke(graph, 15)
            graph_dict = connect_nodes(nodes["nodes"], sampled_dialogues + dialogues, self.model_storage.storage[self.sim_model].model)
        except Exception as e:
            logger.error("Error in dialog sampler: %s", e)
            return Graph({})
        graph_dict = {"edges": graph_dict["edges"], "nodes": graph_dict["nodes"]}
        return Graph(graph_dict)

    def invoke(
            self, pipeline_data: PipelineDataType, enable_evals: bool = False
            ) -> tuple[BaseGraph, metrics.DGReportType]:

        if pipeline_data.supported_graph is not None:
            cur_graph = pipeline_data.supported_graph
            start_point = 0
            report = {}
        else:
            raw_data = PipelineDataType(dialogs=pipeline_data.dialogs[: self.step], true_graph=pipeline_data.true_graph)
            cur_graph, report = self.graph_generator.invoke(raw_data, enable_evals)
            report = {f"d2g_light:{k}": v for k, v in report.items()}
            start_point = self.step
        if enable_evals and pipeline_data.true_graph is not None:
            report.update(self.evaluate(cur_graph, pipeline_data.true_graph, "step1"))
        for point in range(start_point, len(pipeline_data.dialogs) - 1, self.step):
            cur_graph = self._add_step(pipeline_data.dialogs[point : point + self.step], cur_graph)
            if enable_evals and pipeline_data.true_graph is not None:
                report.update(self.evaluate(cur_graph, pipeline_data.true_graph, "extender"))

        _, _, last_user = get_helpers(pipeline_data.dialogs)
        try:
            if enable_evals and pipeline_data.true_graph is not None:
                report.update(self.evaluate(cur_graph, pipeline_data.true_graph, "step2"))
            if not last_user:
                return cur_graph, report

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
            messages = [HumanMessage(content=prompt.format(graph_dict=cur_graph.graph_dict))]

            fixed_output_parser = OutputFixingParser.from_llm(
                parser=PydanticOutputParser(pydantic_object=ReasonGraph), llm=self.model_storage.storage[self.formatting_llm].model
            )
            chain = self.model_storage.storage[self.filling_llm].model | fixed_output_parser

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
        except Exception as e:
            logger.error("Error in step3: %s", e)
            return Graph({}), report

    async def ainvoke(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)

    def evaluate(self, graph, gt_graph, eval_stage: str) -> metrics.DGReportType:

        report = {}
        for metric in getattr(self, eval_stage + "_evals"):
            report[metric.__name__ + ":" + eval_stage] = metric(graph, gt_graph)
        return report

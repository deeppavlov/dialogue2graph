"""
Three Stage D2GExtender
-----------------------

The module provides three step algorithm aimed to extend dialog graph by generating.
"""

import logging
from datetime import datetime
from typing import List, Callable
from pydantic import ConfigDict
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

from dialogue2graph.utils.logger import Logger
from dialogue2graph import metrics
from dialogue2graph.pipelines.core.dialogue_sampling import RecursiveDialogueSampler
from dialogue2graph.pipelines.d2g_llm.three_stages_llm import LLMGraphGenerator
from dialogue2graph.pipelines.core.d2g_generator import DGBaseGenerator
from dialogue2graph.pipelines.core.graph import BaseGraph, Graph, Metadata
from dialogue2graph.pipelines.core.schemas import ReasonGraph, Node
from dialogue2graph.pipelines.core.dialogue import Dialogue
from dialogue2graph.pipelines.model_storage import ModelStorage
from dialogue2graph.utils.dg_helper import connect_nodes, get_helpers
from dialogue2graph.pipelines.helpers.parse_data import PipelineDataType
from dialogue2graph.pipelines.helpers.prompts.missing_edges_prompt import (
    add_edge_prompt_1,
    add_edge_prompt_2,
)
from dialogue2graph.pipelines.d2g_extender.prompts import (
    extending_prompt_part_1,
    extending_prompt_part_2,
    dg_examples,
)


class DialogueNodes(BaseModel):
    """Class for dialog nodes"""

    nodes: List[Node] = Field(description="List of nodes representing assistant states")
    # reason: str = Field(description="explanation")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("langchain_core.vectorstores.base").setLevel(logging.ERROR)
logger = Logger(__file__)

dialogue_sampler = RecursiveDialogueSampler()


class LLMGraphExtender(DGBaseGenerator):
    """Graph generator which iteratively takes step dialogues and adds them to graph
    generated on the previous step. First step is done with LightGraphGenerator or taken from
    supported graph
    Generation stages:

    1.
        a. If supported graph is given, it is used as a start. Otherwise, graph is generated with LLMGraphGenerator from first step dialogs
        b. Algorithmic connecting nodes by edges.
    2. Iterative steps:
        a. LLM extension of graph nodes with next step dialogs.
        b. Algorithmic connecting nodes by edges.
    3. If one of dialogues ends with user's utterance, ask LLM to add missing edges.

    Attributes:
        model_storage: Model storage
        extending_llm: Name of LLM for extending graph nodes
        filling_llm: Name of LLM for adding missing edges
        dialog_llm: Name of LLM used in dialog sampler
        formatting_llm: Name of LLM for formatting other LLMs output
        sim_model: HuggingFace name for similarity model
        step: number of dialogs for one step
        graph_generator: graph generator for the first stage
        step1_evals: Evaluation metrics called after first stage
        extender_evals: Evaluation metrics called after each extension step
        step2_evals: Evaluation metrics called after stage 2
        end_evals: Evaluation metrics called at the end of generation process
        model_config: It's a parameter for internal use of Pydantic
    """

    model_storage: ModelStorage = Field(description="Model storage")
    extending_llm: str = Field(
        description="LLM for extending graph nodes", default="extender_extending_llm:v1"
    )
    filling_llm: str = Field(
        description="LLM for adding missing edges", default="extender_filling_llm:v1"
    )
    formatting_llm: str = Field(
        description="LLM for formatting output", default="extender_formatting_llm:v1"
    )
    dialog_llm: str = Field(
        description="LLM for dialog sampler", default="extender_dialog_llm:v1"
    )
    sim_model: str = Field(
        description="Similarity model", default="extender_sim_model:v1"
    )
    step: int
    graph_generator: LLMGraphGenerator
    step1_evals: list[Callable]
    extender_evals: list[Callable]
    step2_evals: list[Callable]
    end_evals: list[Callable]
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        model_storage: ModelStorage,
        grouping_llm: str = "extender_grouping_llm:v1",
        extending_llm: str = "extender_extending_llm:v1",
        filling_llm: str = "extender_filling_llm:v1",
        formatting_llm: str = "extender_formatting_llm:v1",
        dialog_llm: str = "extender_dialog_llm:v1",
        sim_model: str = "extender_sim_model:v1",
        step1_evals: list[Callable] | None = [],
        extender_evals: list[Callable] | None = [],
        step2_evals: list[Callable] | None = [],
        end_evals: list[Callable] | None = [],
        step: int = 2,
    ):
        # if model is not in model storage put the default model there
        model_storage.add(
            key=grouping_llm,
            config={"model_name": "chatgpt-4o-latest", "temperature": 0},
            model_type=ChatOpenAI,
        )

        model_storage.add(
            key=extending_llm,
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
            key=dialog_llm,
            config={"model_name": "o3-mini", "temperature": 1},
            model_type=ChatOpenAI,
        )

        model_storage.add(
            key=sim_model,
            config={"model_name": "BAAI/bge-m3", "model_kwargs": {"device": "cpu"}},
            model_type=HuggingFaceEmbeddings,
        )
        super().__init__(
            model_storage=model_storage,
            extending_llm=extending_llm,
            filling_llm=filling_llm,
            formatting_llm=formatting_llm,
            dialog_llm=dialog_llm,
            sim_model=sim_model,
            graph_generator=LLMGraphGenerator(
                model_storage,
                grouping_llm,
                filling_llm,
                formatting_llm,
                sim_model,
                step2_evals,
                end_evals,
            ),
            step1_evals=step1_evals,
            extender_evals=extender_evals,
            step2_evals=step2_evals,
            end_evals=end_evals,
            step=step,
        )

    def _add_step(self, dialogues: list[Dialogue], graph: Graph) -> Graph:
        dialogs = ""
        for idx, dial in enumerate(dialogues):
            dialogs += f"\nDialogue_{idx}: {dial}"

        prompt = PromptTemplate.from_template(
            extending_prompt_part_1 + "{graph}. " + extending_prompt_part_2 + dialogs
        )

        fixed_output_parser = OutputFixingParser.from_llm(
            parser=PydanticOutputParser(pydantic_object=DialogueNodes),
            llm=self.model_storage.storage[self.formatting_llm].model,
        )
        chain = (
            self.model_storage.storage[self.extending_llm].model | fixed_output_parser
        )

        messages = [
            HumanMessage(
                content=prompt.format(graph=graph.graph_dict, examples=dg_examples)
            )
        ]
        nodes = chain.invoke(messages).model_dump()

        new_nodes = []
        for idx, node in enumerate(nodes["nodes"]):
            node["utterances"] = list(set(node["utterances"]))
            if node["utterances"]:
                new_nodes.append(node)
        nodes["nodes"] = new_nodes

        try:
            sampled_dialogues = dialogue_sampler.invoke(
                graph,
                cycle_ends_model=self.model_storage.storage[self.dialog_llm].model,
                upper_limit=15,
            )
            graph_dict = connect_nodes(
                nodes["nodes"],
                sampled_dialogues + dialogues,
                self.model_storage.storage[self.sim_model].model,
            )
        except Exception as e:
            logger.error("Error in dialog sampler: %s", e)
            return Graph(graph_dict={}, metadata=graph.metadata)

        return Graph(
            graph_dict={"edges": graph_dict["edges"], "nodes": graph_dict["nodes"]},
            metadata=graph.metadata,
            )

    def invoke(
        self, pipeline_data: PipelineDataType, enable_evals: bool = False
    ) -> tuple[BaseGraph, metrics.DGReportType]:
        """Invoke primary method of the three stages generation algorithm.

        Args:
            pipeline_data: data for generation and evaluation.
            enable_evals: when true, evaluate method is called.

        Returns:
            A tuple containing the resulting graph and report dictionary.
        """
        report = {}
        cur_graph = pipeline_data.supported_graph or self._initial_graph(
            pipeline_data, enable_evals, report
        )
        cur_graph.metadata = Metadata(
            generator_name="d2g_extender",
            models_config=self.model_storage.model_dump(),
            schema_version="v1",
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        if enable_evals and pipeline_data.true_graph:
            report.update(self.evaluate(cur_graph, pipeline_data.true_graph, "step1"))

        for start in range(
            self.step if not pipeline_data.supported_graph else 0,
            len(pipeline_data.dialogs),
            self.step,
        ):
            cur_graph = self._add_step(
                pipeline_data.dialogs[start : start + self.step], cur_graph
            )
            if enable_evals and pipeline_data.true_graph:
                report.update(
                    self.evaluate(cur_graph, pipeline_data.true_graph, "extender")
                )

        try:
            if enable_evals and pipeline_data.true_graph:
                report.update(
                    self.evaluate(cur_graph, pipeline_data.true_graph, "step2")
                )

            if not get_helpers(pipeline_data.dialogs)[-1]:
                return cur_graph, report

            result_graph = self._finalize_graph(
                pipeline_data, cur_graph, enable_evals, report
            )
            return result_graph, report
        except Exception as e:
            logger.error("Error in step3: %s", e)
            return Graph(graph_dict={}, metadata=cur_graph.metadata), report

    def _initial_graph(self, pipeline_data, enable_evals, report):
        raw_data = PipelineDataType(
            dialogs=pipeline_data.dialogs[: self.step],
            true_graph=pipeline_data.true_graph,
        )
        cur_graph, initial_report = self.graph_generator.invoke(raw_data, enable_evals)
        report.update({f"d2g_light:{k}": v for k, v in initial_report.items()})
        return cur_graph

    def _finalize_graph(self, pipeline_data, cur_graph, enable_evals, report):
        dialogs = ""
        for idx, dial in enumerate(pipeline_data.dialogs):
            dialogs += f"\nDialogue_{idx}: {dial}"

        prompt = PromptTemplate.from_template(
            add_edge_prompt_1 + "{graph_dict}. " + add_edge_prompt_2 + dialogs
        )

        messages = [
            HumanMessage(content=prompt.format(graph_dict=cur_graph.graph_dict))
        ]

        fixed_output_parser = OutputFixingParser.from_llm(
            parser=PydanticOutputParser(pydantic_object=ReasonGraph),
            llm=self.model_storage.storage[self.formatting_llm].model,
        )
        chain = self.model_storage.storage[self.filling_llm].model | fixed_output_parser
        result = chain.invoke(messages)

        if result and all(e["target"] for e in result.model_dump()["edges"]):
            result_graph = Graph(
                graph_dict=result.model_dump(),
                metadata=cur_graph.metadata
                )
            if enable_evals and pipeline_data.true_graph:
                report.update(
                    self.evaluate(result_graph, pipeline_data.true_graph, "end")
                )
            return result_graph

        return Graph(graph_dict={}, metadata=cur_graph.metadata)

    async def ainvoke(self, *args, **kwargs): # pragma: no cover
        return self.invoke(*args, **kwargs)

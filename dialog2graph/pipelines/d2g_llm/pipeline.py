"""
D2GLLMPipeline
--------------

The module contains pipeline for graph generation using LLMs.
"""

# from dialog2graph.pipelines.core.pipeline import Pipeline as BasePipeline
from typing import Callable
from dotenv import load_dotenv

from dialog2graph import metrics
from dialog2graph.pipelines.core.pipeline import BasePipeline
from dialog2graph.pipelines.model_storage import ModelStorage

from dialog2graph.pipelines.d2g_llm.three_stages_llm import LLMGraphGenerator

load_dotenv()


class D2GLLMPipeline(BasePipeline):
    """
    D2GLLMPipeline is a pipeline class for generating graphs based on provided dialogs using LLMs.

    Attributes:
        name (str): The name of the pipeline.
        model_storage (ModelStorage): An object to manage and store models used in the pipeline.
        grouping_llm (str): The key for the grouping LLM model in the model storage. Defaults to "d2g_llm_grouping_llm:v1".
        filling_llm (str): The key for the filling LLM model in the model storage. Defaults to "d2g_llm_filling_llm:v1".
        formatting_llm (str): The key for the formatting LLM model in the model storage. Defaults to "d2g_llm_formatting_llm:v1".
        sim_model (str): The key for the similarity embedder model in the model storage. Defaults to "d2g_llm_sim_model:v1".
        step2_evals (list[Callable], optional): A list of evaluation functions to be applied at step 2 of the pipeline. Defaults to None.
        end_evals (list[Callable], optional): A list of evaluation functions to be applied at the end of the pipeline. Defaults to None.
    """

    def __init__(
        self,
        name: str,
        model_storage: ModelStorage,
        grouping_llm: str = "three_stages_llm_grouping_llm:v1",
        filling_llm: str = "three_stages_llm_filling_llm:v1",
        formatting_llm: str = "three_stages_llm_formatting_llm:v1",
        sim_model: str = "three_stages_llm_sim_model:v1",
        step2_evals: list[Callable] = metrics.DGEvalBase,
        end_evals: list[Callable] = metrics.DGEvalBase,
    ):
        super().__init__(
            model_storage=model_storage,
            sim_model=sim_model,
            name=name,
            steps=[
                LLMGraphGenerator(
                    model_storage,
                    grouping_llm,
                    filling_llm,
                    formatting_llm,
                    sim_model,
                    step2_evals,
                    end_evals,
                )
            ],
        )

    def _validate_pipeline(self):
        pass

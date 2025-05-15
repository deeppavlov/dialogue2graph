"""
D2GExtenderPipeline
-------------------

The module contains pipeline to extend dialog graphs.
"""

from typing import Callable
from dotenv import load_dotenv

from dialog2graph import metrics
from dialog2graph.pipelines.core.pipeline import BasePipeline
from dialog2graph.pipelines.model_storage import ModelStorage
from dialog2graph.pipelines.d2g_extender.three_stages_extender import LLMGraphExtender


load_dotenv()


class D2GExtenderPipeline(BasePipeline):
    """LLM graph extender pipeline"""

    def __init__(
        self,
        name: str,
        model_storage: ModelStorage,
        grouping_llm: str = "extender_grouping_llm:v1",
        extending_llm: str = "extender_extending_llm:v1",
        filling_llm: str = "extender_filling_llm:v1",
        formatting_llm: str = "extender_formatting_llm:v1",
        dialog_llm: str = "extender_dialog_llm:v1",
        sim_model: str = "extender_sim_model:v1",
        step1_evals: list[Callable] = metrics.PreDGEvalBase,
        extender_evals: list[Callable] = metrics.PreDGEvalBase,
        step2_evals: list[Callable] = metrics.DGEvalBase,
        end_evals: list[Callable] = metrics.DGEvalBase,
        step: int = 2,
    ):
        super().__init__(
            model_storage=model_storage,
            sim_model=sim_model,
            name=name,
            steps=[
                LLMGraphExtender(
                    model_storage,
                    grouping_llm,
                    extending_llm,
                    filling_llm,
                    formatting_llm,
                    dialog_llm,
                    sim_model,
                    step1_evals,
                    extender_evals,
                    step2_evals,
                    end_evals,
                    step,
                ),
            ],
        )

    def _validate_pipeline(self):
        pass

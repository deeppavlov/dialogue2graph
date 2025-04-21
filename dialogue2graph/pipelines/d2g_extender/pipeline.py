"""
D2GExtenderPipeline
-------------------

The module contains pipeline to extend dialog graphs.
"""

from typing import Callable
from dotenv import load_dotenv
from dialogue2graph.pipelines.core.pipeline import BasePipeline
from dialogue2graph.pipelines.model_storage import ModelStorage
from dialogue2graph.pipelines.d2g_extender.three_stages_extender import LLMGraphExtender

load_dotenv()


class D2GExtenderPipeline(BasePipeline):
    """LLM graph extender pipeline"""

    def __init__(
        self,
        name: str,
        model_storage: ModelStorage,
        extending_llm: str = "extender_extending_llm:v1",
        filling_llm: str = "extender_filling_llm:v1",
        formatting_llm: str = "extender_formatting_llm:v1",
        dialog_llm: str = "extender_dialog_llm:v1",
        sim_model: str = "extender_sim_model:v1",
        step1_evals: list[Callable] = None,
        extender_evals: list[Callable] = None,
        step2_evals: list[Callable] = None,
        end_evals: list[Callable] = None,
        step: int = 2,
    ):
        super().__init__(
            name=name,
            steps=[
                LLMGraphExtender(
                    model_storage,
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

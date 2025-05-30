"""
D2GLightPipeline
-----------------

The module contains pipeline for usual graph generation (so called light mode generation)
"""

from typing import Callable
from dotenv import load_dotenv

from dialog2graph import metrics
from dialog2graph.pipelines.core.pipeline import BasePipeline
from dialog2graph.pipelines.d2g_light.three_stages_light import LightGraphGenerator
from dialog2graph.pipelines.model_storage import ModelStorage


load_dotenv()


class D2GLightPipeline(BasePipeline):
    """Light graph generator pipeline"""

    def __init__(
        self,
        name: str,
        model_storage: ModelStorage,
        filling_llm: str = "three_stages_light_filling_llm:v1",
        formatting_llm: str = "three_stages_light_formatting_llm:v1",
        sim_model: str = "three_stages_light_sim_model:v1",
        step2_evals: list[Callable] = metrics.DGEvalBase,
        end_evals: list[Callable] = metrics.DGEvalBase,
    ):
        super().__init__(
            model_storage=model_storage,
            sim_model=sim_model,
            name=name,
            steps=[
                LightGraphGenerator(
                    model_storage,
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

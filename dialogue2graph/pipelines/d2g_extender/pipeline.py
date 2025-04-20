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
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()


class D2GExtenderPipeline(BasePipeline):
    """LLM graph extender pipeline"""

    def __init__(
        self,
        name: str,
        model_storage: ModelStorage,
        extending_llm: str = "d2g_extender_extending_llm:v1",
        filling_llm: str = "d2g_extender_filling_llm:v1",
        formatting_llm: str = "d2g_extender_formatting_llm:v1",
        dialog_llm: str = "d2g_extender_dialog_llm:v1",
        sim_model: str = "d2g_extender_sim_model:v1",
        step1_evals: list[Callable] = None,
        extender_evals: list[Callable] = None,
        step2_evals: list[Callable] = None,
        end_evals: list[Callable] = None,
        step: int = 2,
    ):
        # if model is not in model storage put the default model there
        model_storage.add(
            key=extending_llm,
            config={"model_name": "chatgpt-4o-latest", "temperature": 0},
            model_type="llm",
        )

        model_storage.add(
            key=filling_llm,
            config={"mode_name": "o3-mini", "temperature": 1},
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
            config={"model_name": "BAAI/bge-m3", "device": "cpu"},
            model_type=HuggingFaceEmbeddings,
        )

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

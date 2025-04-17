# from dialogue2graph.pipelines.core.pipeline import Pipeline as BasePipeline
from typing import Callable
from dotenv import load_dotenv
from dialogue2graph.pipelines.core.pipeline import BasePipeline
from dialogue2graph.pipelines.model_storage import ModelStorage

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from dialogue2graph.pipelines.d2g_llm.three_stages_llm import LLMGraphGenerator

load_dotenv()


class D2GLLMPipeline(BasePipeline):
    """LLM graph generator pipeline"""

    def __init__(
        self,
        name: str,
        model_storage: ModelStorage,
        grouping_llm: str = "d2g_llm_grouping_llm:v1",
        filling_llm: str = "d2g_llm_filling_llm:v1",
        formatting_llm: str = "d2g_llm_formatting_llm:v1",
        sim_model: str = "d2g_llm_sim_model:v1",
        step2_evals: list[Callable] = None,
        end_evals: list[Callable] = None,
    ):
        # if model is not in model storage put the default model there
        model_storage.add(
            key=grouping_llm,
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
            key=sim_model,
            config={"model_name": "BAAI/bge-m3", "device": "cpu"},
            model_type=HuggingFaceEmbeddings,
        )

        super().__init__(
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

from typing import Callable
from dotenv import load_dotenv
from dialogue2graph.pipelines.core.pipeline import BasePipeline
from dialogue2graph.pipelines.d2g_light.three_stages_light import LightGraphGenerator
from dialogue2graph.pipelines.model_storage import ModelStorage

load_dotenv()


class Pipeline(BasePipeline):
    """Light graph generator pipeline"""

    def __init__(
        self,
        name: str,
        model_storage: ModelStorage,
        filling_llm: str = "d2g_light_filling_llm:v1",
        formatting_llm: str = "d2g_light_formatting_llm:v1",
        sim_model: str = "d2g_light_sim_model:v1",
        step2_evals: list[Callable] = None,
        end_evals: list[Callable] = None,
    ):
        # check if models are in model storage
        # if model is not in model storage put the default model there
        if filling_llm not in model_storage.storage:
            model_storage.add(key=filling_llm, config={"model": "chatgpt-4o-latest", "temperature": 0}, model_type="llm")

        if formatting_llm not in model_storage.storage:
            model_storage.add(key=formatting_llm, config={"model": "gpt-4o-mini", "temperature": 0}, model_type="llm")

        if sim_model not in model_storage.storage:
            model_storage.add(key=sim_model, config={"model_name": "BAAI/bge-m3", "device": "cpu"}, model_type="emb")
        super().__init__(name=name, steps=[LightGraphGenerator(model_storage, filling_llm, formatting_llm, sim_model, step2_evals, end_evals)])

    def _validate_pipeline(self):
        pass

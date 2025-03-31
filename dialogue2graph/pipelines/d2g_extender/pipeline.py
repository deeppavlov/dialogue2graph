from dotenv import load_dotenv
from dialogue2graph.pipelines.core.pipeline import Pipeline as BasePipeline
from dialogue2graph.pipelines.core.dialogue import Dialogue
from dialogue2graph.pipelines.core.graph import Graph
from dialogue2graph.pipelines.d2g_algo.three_stages_algo import ThreeStagesGraphGenerator as AlgoGenerator
from dialogue2graph.pipelines.model_storage import ModelStorage
from .three_stages_extender import ThreeStagesGraphGenerator as Extender
from dialogue2graph.pipelines.helpers.parse_data import DataParser

load_dotenv()


class Pipeline(BasePipeline):
    """LLM graph extender pipeline"""

    def __init__(
        self, model_storage: ModelStorage, extending_llm: str = None, filling_llm: str = None, formatting_llm: str = None, sim_model: str = None
    ):
        # check if models are in model storage
        extending_llm = model_storage.storage.get(extending_llm, None)
        if not extending_llm:
            model_storage.add(key="d2g_extender_extending_llm:v1", config={"name": "gpt-4o-latest", "temperature": 0}, model_type="llm")
            extending_llm = model_storage.storage["d2g_extender_extending_llm:v1"].model

        filling_llm = model_storage.storage.get(filling_llm, None)
        if not filling_llm:
            model_storage.add(key="d2g_extender_filling_llm:v1", config={"name": "o3-mini", "temperature": 1}, model_type="llm")
            filling_llm = model_storage.storage["d2g_extender_filling_llm:v1"].model

        formatting_llm = model_storage.storage.get(formatting_llm, None)
        if not formatting_llm:
            model_storage.add(key="d2g_extender_formatting_llm:v1", config={"model": "gpt-4o-mini", "temperature": 0}, model_type="llm")
            formatting_llm = model_storage.storage["d2g_extender_formatting_llm:v1"].model

        sim_model = model_storage.storage.get(sim_model, None)
        if not sim_model:
            model_storage.add(key="d2g_extender_sim_model:v1", config={"model_name": "BAAI/bge-m3", "device": "cuda:0"}, model_type="emb")
            sim_model = model_storage.storage["d2g_extender_sim_model:v1"].model

        super().__init__(
            steps=[
                DataParser(),
                AlgoGenerator(filling_llm, formatting_llm, sim_model),
                Extender(extending_llm, filling_llm, formatting_llm, sim_model),
            ]
        )

    def _validate_pipeline(self):
        pass

    def invoke(self, data: Dialogue | list[Dialogue] | dict | list[list] | list[dict]) -> Graph:
        dialogues = self.steps[0].invoke(data)
        graph = self.steps[1].invoke(dialogues[:1])
        for idx in range(1, len(dialogues)):
            graph = self.steps[2].invoke(dialogues[idx : idx + 1], graph)

        return graph

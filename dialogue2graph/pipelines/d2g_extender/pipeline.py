from dotenv import load_dotenv
from dialogue2graph.pipelines.core.pipeline import BasePipeline
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
        self,
        model_storage: ModelStorage,
        extending_llm: str = "d2g_extender_extending_llm:v1",
        filling_llm: str = "d2g_extender_filling_llm:v1",
        formatting_llm: str = "d2g_extender_formatting_llm:v1",
        sim_model: str = "d2g_extender_sim_model:v1",
    ):
        # check if models are in model storage
        # if model is not in model storage put the default model there
        if extending_llm not in model_storage.storage:
            model_storage.add(key=extending_llm, config={"name": "gpt-4o-latest", "temperature": 0}, model_type="llm")

        if filling_llm not in model_storage.storage:
            model_storage.add(key=filling_llm, config={"name": "o3-mini", "temperature": 1}, model_type="llm")

        if formatting_llm not in model_storage.storage:
            model_storage.add(key=formatting_llm, config={"name": "gpt-4o-mini", "temperature": 0}, model_type="llm")

        if sim_model not in model_storage.storage:
            model_storage.add(key=sim_model, config={"model_name": "cointegrated/LaBSE-en-ru", "device": "cpu"}, model_type="emb")

        super().__init__(
            steps=[
                DataParser(),
                AlgoGenerator(model_storage, filling_llm, formatting_llm, sim_model),
                Extender(model_storage, extending_llm, filling_llm, formatting_llm, sim_model),
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

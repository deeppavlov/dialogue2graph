from dotenv import load_dotenv
from dialogue2graph.pipelines.core.pipeline import BasePipeline
from dialogue2graph.pipelines.helpers.parse_data import DataParser
from dialogue2graph.pipelines.model_storage import ModelStorage
from .three_stages_llm import ThreeStagesGraphGenerator as LLMGenerator

load_dotenv()


class Pipeline(BasePipeline):
    """LLM graph generator pipeline"""

    def __init__(
        self, model_storage: ModelStorage, grouping_llm: str = None, filling_llm: str = None, formatting_llm: str = None, sim_model: str = None
    ):

        # check if models are in model storage
        # if model is not in model storage put the default model there
        grouping_llm = model_storage.storage.get(grouping_llm, None)
        if not grouping_llm:
            model_storage.add(key="d2g_llm_grouping_llm:v1", config={"name": "gpt-4o-latest", "temperature": 0}, model_type="llm")
            grouping_llm = model_storage.storage["d2g_llm_grouping_llm:v1"].model

        filling_llm = model_storage.storage.get(filling_llm, None)
        if not filling_llm:
            model_storage.add(key="d2g_llm_filling_llm:v1", config={"name": "o3-mini", "temperature": 1}, model_type="llm")
            filling_llm = model_storage.storage["d2g_llm_filling_llm:v1"].model

        formatting_llm = model_storage.storage.get(formatting_llm, None)
        if not formatting_llm:
            model_storage.add(key="d2g_llm_formatting_llm:v1", config={"model": "gpt-4o-mini", "temperature": 0}, model_type="llm")
            formatting_llm = model_storage.storage["d2g_llm_formatting_llm:v1"].model

        sim_model = model_storage.storage.get(sim_model, None)
        if not sim_model:
            model_storage.add(key="d2g_llm_sim_model:v1", config={"model_name": "BAAI/bge-m3", "device": "cuda:0"}, model_type="emb")
            sim_model = model_storage.storage["d2g_llm_sim_model:v1"].model

        super().__init__(steps=[DataParser(), LLMGenerator(grouping_llm, filling_llm, formatting_llm, sim_model)])

    def _validate_pipeline(self):
        pass

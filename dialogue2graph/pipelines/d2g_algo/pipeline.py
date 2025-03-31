from dotenv import load_dotenv
from dialogue2graph.pipelines.core.pipeline import Pipeline as BasePipeline
from dialogue2graph.pipelines.helpers.parse_data import DataParser
from dialogue2graph.pipelines.model_storage import ModelStorage
from .three_stages_algo import ThreeStagesGraphGenerator as AlgoGenerator

load_dotenv()


class Pipeline(BasePipeline):
    """Algorithmic graph generator pipeline"""

    def __init__(
        self, model_storage: ModelStorage, filling_llm: str = None, formatting_llm: str = None, sim_model: str = None
    ):
        # check if models are in model storage
        filling_llm = model_storage.storage.get(filling_llm, None)
        if not filling_llm:
            model_storage.add(key="d2g_algo_filling_llm:v1", config={"name": "gpt-4o-latest", "temperature": 0}, model_type="llm")
            filling_llm = model_storage.storage["d2g_algo_filling_llm:v1"].model

        formatting_llm = model_storage.storage.get(formatting_llm, None)
        if not formatting_llm:
            model_storage.add(key="d2g_algo_formatting_llm:v1", config={"model": "gpt-4o-mini", "temperature": 0}, model_type="llm")
            formatting_llm = model_storage.storage["d2g_algo_formatting_llm:v1"].model

        sim_model = model_storage.storage.get(sim_model, None)
        if not sim_model:
            model_storage.add(key="d2g_algo_sim_model:v1", config={"model_name": "BAAI/bge-m3", "device": "cuda:0"}, model_type="emb")
            sim_model = model_storage.storage["d2g_algo_sim_model:v1"].model

        super().__init__(steps=[DataParser(), AlgoGenerator(filling_llm, formatting_llm, sim_model)])

    def _validate_pipeline(self):
        pass

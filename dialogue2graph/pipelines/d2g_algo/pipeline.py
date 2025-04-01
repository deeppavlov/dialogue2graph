from dotenv import load_dotenv
from dialogue2graph.pipelines.core.pipeline import BasePipeline
from dialogue2graph.pipelines.helpers.parse_data import DataParser
from dialogue2graph.pipelines.model_storage import ModelStorage
from .three_stages_algo import ThreeStagesGraphGenerator as AlgoGenerator

load_dotenv()


class Pipeline(BasePipeline):
    """Algorithmic graph generator pipeline"""

    def __init__(
        self,
        model_storage: ModelStorage,
        filling_llm: str = "d2g_algo_filling_llm:v1",
        formatting_llm: str = "d2g_algo_formatting_llm:v1",
        sim_model: str = "d2g_algo_sim_model:v1",
    ):
        # check if models are in model storage
        # if model is not in model storage put the default model there
        if filling_llm not in model_storage.storage:
            model_storage.add(key=filling_llm, config={"name": "gpt-4o-latest", "temperature": 0}, model_type="llm")

        if formatting_llm not in model_storage.storage:
            model_storage.add(key=formatting_llm, config={"name": "gpt-4o-mini", "temperature": 0}, model_type="llm")

        if sim_model not in model_storage.storage:
            model_storage.add(key=sim_model, config={"model_name": "cointegrated/LaBSE-en-ru", "device": "cpu"}, model_type="emb")

        super().__init__(steps=[DataParser(), AlgoGenerator(model_storage, filling_llm, formatting_llm, sim_model)])

    def _validate_pipeline(self):
        pass

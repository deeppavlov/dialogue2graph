from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from dialogue2graph.pipelines.core.pipeline import Pipeline as BasePipeline
from .three_stages_algo import ThreeStagesGraphGenerator as AlgoGenerator

load_dotenv()


class Pipeline(BasePipeline):
    """Algorithmic graph generator pipeline"""

    def __init__(
        self,
        name: str,
        filling_llm: BaseChatModel,
        formatting_llm: BaseChatModel,
        sim_model: HuggingFaceEmbeddings,
        step2_evals: list[callable],
        end_evals: list[callable],
    ):
        super().__init__(name=name, steps=[AlgoGenerator(filling_llm, formatting_llm, sim_model, step2_evals, end_evals)])

    def _validate_pipeline(self):
        pass

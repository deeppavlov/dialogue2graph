# from dialogue2graph.pipelines.core.pipeline import Pipeline as BasePipeline
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from dialogue2graph.pipelines.core.pipeline import Pipeline as BasePipeline
from dialogue2graph.pipelines.helpers.parse_data import DataParser
from .three_stages_llm import ThreeStagesGraphGenerator as LLMGenerator

load_dotenv()


class Pipeline(BasePipeline):
    """LLM graph generator pipeline"""

    def __init__(self, grouping_llm: BaseChatModel, filling_llm: BaseChatModel, formatting_llm: BaseChatModel, sim_model: HuggingFaceEmbeddings):
        super().__init__(steps=[DataParser(), LLMGenerator(grouping_llm, filling_llm, formatting_llm, sim_model)])

    def _validate_pipeline(self):
        pass

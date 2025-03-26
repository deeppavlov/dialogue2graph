# from dialogue2graph.pipelines.core.pipeline import Pipeline as BasePipeline
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from dialogue2graph.pipelines.core.pipeline import Pipeline as BasePipeline
from dialogue2graph.pipelines.core.dialogue import Dialogue
from dialogue2graph.pipelines.core.graph import Graph
from dialogue2graph.pipelines.helpers.parse_data import parse_data
from .three_stages_llm import ThreeStagesGraphGenerator as LLMGenerator

load_dotenv()


class Pipeline(BasePipeline):
    """LLM graph generator pipeline"""

    graph_generator: LLMGenerator

    def __init__(self, grouping_llm: BaseChatModel, filling_llm: BaseChatModel, sim_model: HuggingFaceEmbeddings):
        super().__init__(graph_generator=LLMGenerator(grouping_llm, filling_llm, sim_model))

    def _validate_pipeline(self):
        pass

    def invoke(self, data: Dialogue | list[Dialogue] | dict | list[list] | list[dict]) -> Graph:

        dialogues = parse_data(data)
        graph = self.graph_generator.invoke(dialogues)
        return graph

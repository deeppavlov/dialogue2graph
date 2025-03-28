from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from dialogue2graph.pipelines.core.pipeline import Pipeline as BasePipeline
from dialogue2graph.pipelines.core.dialogue import Dialogue
from dialogue2graph.pipelines.core.graph import Graph
from dialogue2graph.pipelines.d2g_algo.three_stages_algo import ThreeStagesGraphGenerator as AlgoGenerator
from .three_stages_extender import ThreeStagesGraphGenerator as Extender
from dialogue2graph.pipelines.helpers.parse_data import DataParser

load_dotenv()


class Pipeline(BasePipeline):
    """LLM graph extender pipeline"""

    def __init__(self, extending_llm: BaseChatModel, filling_llm: BaseChatModel, formatting_llm: BaseChatModel, sim_model: HuggingFaceEmbeddings):
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

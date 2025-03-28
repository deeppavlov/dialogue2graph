# import pandas as pd
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.embeddings import HuggingFaceEmbeddings

# from typing import Union
# from pydantic import BaseModel, Field
# from dialogue2graph.pipelines.core.algorithms import GraphGenerator, GraphExtender, InputParser
from dialogue2graph.pipelines.core.pipeline import Pipeline as BasePipeline

# from dialogue2graph import Dialogue, Graph
# from dialogue2graph.pipelines.d2g_algo.three_stages_algo import ThreeStagesGraphGenerator as AlgoGenerator
from .three_stages_extender import ThreeStagesGraphGenerator as Extender
from dialogue2graph.pipelines.helpers.parse_data import DataParser

load_dotenv()


class Pipeline(BasePipeline):
    # class Pipeline(BaseModel):
    #     steps: list[Union[InputParser, GraphGenerator, GraphExtender, list[Dialogue]]] = Field(default_factory=list)
    """LLM graph extender pipeline"""

    def __init__(
        self, extending_llm: BaseChatModel, filling_llm: BaseChatModel, formatting_llm: BaseChatModel, sim_model: HuggingFaceEmbeddings, step: int = 2
    ):
        super().__init__(
            steps=[
                DataParser(),
                Extender(extending_llm, filling_llm, formatting_llm, sim_model, step),
            ]
        )

    def _validate_pipeline(self):
        pass

    # def invoke(self, data: Dialogue | list[Dialogue] | dict | list[list] | list[dict], gt: Graph = None) -> Graph | dict | pd.DataFrame:

    #     n_invokes = len(self.steps)
    #     if gt:
    #         n_invokes = len(self.steps) - 1
    #         output = data

    #     dialogues = self.steps[0].invoke(data)
    #     output = self.steps[1].invoke(dialogues[:1])
    #     for idx in range(1, n_invokes):
    #         output = self.steps[2].invoke(dialogues[idx : idx + 1], output)
    #     if gt:
    #         output = self.steps[-1].evaluate(output, gt)
    #     return output

    # def invoke(self, data: Dialogue | list[Dialogue] | dict | list[list] | list[dict]) -> Graph:

    #     dialogues = self.steps[0].invoke(data)
    #     graph = self.steps[1].invoke(dialogues[:1])
    #     for idx in range(1, len(dialogues)):
    #         graph = self.steps[2].invoke(dialogues[idx : idx + 1], graph)

    #     return graph

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.embeddings import HuggingFaceEmbeddings

from dialogue2graph.pipelines.core.pipeline import Pipeline as BasePipeline

from dialogue2graph.pipelines.d2g_extender.three_stages_extender import ThreeStagesGraphGenerator as Extender


load_dotenv()


class Pipeline(BasePipeline):
    """LLM graph extender pipeline"""

    def __init__(
        self,
        name: str,
        extending_llm: BaseChatModel,
        filling_llm: BaseChatModel,
        formatting_llm: BaseChatModel,
        sim_model: HuggingFaceEmbeddings,
        step1_evals: list[callable],
        extender_evals: list[callable],
        step2_evals: list[callable],
        end_evals: list[callable],
        step: int = 2,
    ):
        super().__init__(
            name=name,
            steps=[
                Extender(extending_llm, filling_llm, formatting_llm, sim_model, step1_evals, extender_evals, step2_evals, end_evals, step),
            ],
        )

    def _validate_pipeline(self):
        pass

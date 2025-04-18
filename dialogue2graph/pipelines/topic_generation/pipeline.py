import os
import json
from pathlib import Path
from dotenv import load_dotenv
from dialogue2graph.pipelines.core.pipeline import BasePipeline
from dialogue2graph.pipelines.model_storage import ModelStorage
from dialogue2graph.datasets.complex_dialogues.generation import LoopedGraphGenerator

load_dotenv()


class TopicGenerationPipeline(BasePipeline):
    """Pipeline for generating topic-based dialogue graphs"""

    def __init__(
        self,
        model_storage: ModelStorage,
        generation_llm: str = "topic_generation_gen_llm:v1",
        validation_llm: str = "topic_generation_val_llm:v1",
        theme_validation_llm: str = "topic_generation_theme_val_llm:v1",
    ):
        
        super().__init__(
            steps=[
                LoopedGraphGenerator(
                    model_storage=model_storage,
                    generation_llm=generation_llm,
                    validation_llm=validation_llm,
                    theme_validation_llm=theme_validation_llm,
                )
            ]
        )

    def _validate_pipeline(self):
        pass

    def invoke(self, topic: str, output_path: str):
        for step in self.steps:
            output = step.invoke(topic=topic)

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        return output

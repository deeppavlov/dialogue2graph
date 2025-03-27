import os
import json
from pathlib import Path
from langchain_openai import ChatOpenAI
from dialogue2graph.pipelines.core import Pipeline
from dialogue2graph.datasets.complex_dialogues.generation import LoopedGraphGenerator


class TopicGenerationPipeline(Pipeline):

    def __init__(self):
        super().__init__()
        gen_model = ChatOpenAI(
            model="gpt-4o-latest",
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )

        val_model = ChatOpenAI(
            model="gpt-3.5-turbo",
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            temperature=0,
        )
        self.steps.append(
            LoopedGraphGenerator(
                generation_model=gen_model,
                validation_model=val_model,
            )
        )

    def invoke(self, topic: str, output_path: str):
        for step in self.steps:
            output = step.invoke(topic=topic)

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        return output

import json
import os
from pathlib import Path
from langchain_openai import ChatOpenAI
from dialogue2graph.pipelines.topic_generation.pipeline import TopicGenerationPipeline
from dialogue2graph.pipelines.model_storage import ModelStorage

ms = ModelStorage()

def generate_data(topic: str, config: dict, output_path: str):
    """Generate dialogue data for a given topic"""

    if config != {}:
        ms.load(config)

    pipeline = TopicGenerationPipeline(ms)

    result = pipeline.invoke(topic)
    print("Result:", result.graph_dict)

    result = pipeline.invoke(topic=topic)

    print("Result:", result)

    # Save results
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump([obj for obj in result], f, indent=2, ensure_ascii=False)

import json
from pathlib import Path
from dialog2graph.pipelines.topic_generation.pipeline import TopicGenerationPipeline
from dialog2graph.pipelines.model_storage import ModelStorage
from dialog2graph.utils.logger import Logger

ms = ModelStorage()
logger = Logger(__name__)


def generate_data(topic: str, config: dict, output_path: str):
    """Generate dialog data for a given topic"""

    if config != {}:
        ms.load(config)

    pipeline = TopicGenerationPipeline(ms)

    result = pipeline.invoke(topic)
    print("Result:", result.graph_dict)

    result = pipeline.invoke(topic=topic)

    logger.info("Result: %s" % result)

    # Save results
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump([obj for obj in result], f, indent=2, ensure_ascii=False)

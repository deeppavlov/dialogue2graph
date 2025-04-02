import json
from pathlib import Path
from dialogue2graph.pipelines.d2g_llm.pipeline import Pipeline
from dialogue2graph.pipelines.model_storage import ModelStorage

ms = ModelStorage()


def generate_llm(dialogues: str, config: Path, output_path: str):
    """Generates graph from dialogues via d2g_llm pipeline using parameters from config
    and saves graph dictionary to output_path"""

    if config != {}:
        ms.load(config)
    pipeline = Pipeline(ms)

    result = pipeline.invoke(dialogues)
    print("Result:", result.graph_dict)

    # Save results
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result.graph_dict, f, indent=2, ensure_ascii=False)

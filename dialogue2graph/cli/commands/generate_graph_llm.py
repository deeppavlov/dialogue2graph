import json
from pathlib import Path, PosixPath
from dialogue2graph.pipelines.d2g_llm.pipeline import Pipeline
from dialogue2graph.pipelines.helpers.parse_data import PipelineRawDataType
from dialogue2graph.pipelines.model_storage import ModelStorage

ms = ModelStorage()


def generate_llm(
    dialogues: PosixPath, tgraph: PosixPath, config: dict, output_path: str
):
    """Generates graph from dialogues via d2g_llm pipeline using parameters from config
    and saves graph dictionary to output_path"""

    if config != {}:
        ms.load(config)
    pipeline = Pipeline("d2g_llm", ms)

    raw_data = PipelineRawDataType(dialogs=dialogues, true_graph=tgraph)
    result, report = pipeline.invoke(raw_data)

    if output_path is not None:
        # Save results
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({"graph": result.graph_dict}, f, indent=2, ensure_ascii=False)
            json.dump({"report": report}, f, indent=2, ensure_ascii=False)
    else:
        print(json.dumps({"graph": result.graph_dict}, f, indent=2, ensure_ascii=False))
        print(json.dumps({"report": report}, f, indent=2, ensure_ascii=False))

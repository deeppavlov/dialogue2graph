import json
from pathlib import Path, PosixPath
from dialogue2graph.pipelines.d2g_extender.pipeline import Pipeline
from dialogue2graph.pipelines.model_storage import ModelStorage
from dialogue2graph.pipelines.helpers.parse_data import PipelineRawDataType


ms = ModelStorage()


def generate_extender(
    dialogues: PosixPath,
    graph: PosixPath,
    tgraph: PosixPath,
    config: dict,
    output_path: str,
):
    """Generates graph from dialogues via d2g_llm pipeline using parameters from config
    and saves graph dictionary to output_path"""

    if config != {}:
        ms.load(config)
    pipeline = Pipeline("d2g_ext", ms, step=1)

    raw_data = PipelineRawDataType(
        dialogs=dialogues, supported_graph=graph, true_graph=tgraph
    )
    result, _ = pipeline.invoke(raw_data)

    # Save results
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result.graph_dict, f, indent=2, ensure_ascii=False)

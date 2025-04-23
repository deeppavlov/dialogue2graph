import os
import json
from pathlib import Path, PosixPath
from datetime import datetime
from datasets import load_dataset

from dialogue2graph.utils.logger import Logger
from dialogue2graph.pipelines.core.pipeline import BasePipeline
from dialogue2graph.pipelines.d2g_light.pipeline import D2GLightPipeline
from dialogue2graph.pipelines.d2g_llm.pipeline import D2GLLMPipeline
from dialogue2graph.pipelines.d2g_extender.pipeline import D2GExtenderPipeline
from dialogue2graph.pipelines.model_storage import ModelStorage
from dialogue2graph.pipelines.helpers.parse_data import PipelineRawDataType

logger = Logger(__file__)
ms = ModelStorage()

metrics_path_name = "tests/metrics_results"
metrics_path = Path(metrics_path_name)

dataset = load_dataset(
    "DeepPavlov/d2g_generated_augmented", token=os.getenv("HUGGINGFACE_TOKEN")
)


def get_latest_file(path: PosixPath, pattern: str = "*"):
    try:
        return max(
            path.glob(pattern), key=lambda item: item.stat().st_ctime, default=None
        )
    except ValueError:
        return None


def compare_summaries(pipeline_name, last_summary, new_summary) -> bool:
    if not last_summary or not new_summary:
        return False
    last_avg, new_avg = last_summary[-1], new_summary[-1]
    if last_avg["graph"] != "average" or new_avg["graph"] != "average":
        logger.warning("Summary does not contain average graph")
        return False
    if last_avg["duration"] < new_avg["duration"]:
        logger.warning(
            "Pipeline %s got slower: %f -> %f",
            pipeline_name,
            last_avg["duration"],
            new_avg["duration"],
        )
    return last_avg["similarity"] <= new_avg["similarity"]


def test_d2g_pipeline(pipeline: BasePipeline) -> bool:
    """Check whether pipeline metrics against dataset is better than last in records"""

    new_summary = []
    for data in dataset["train"].select(range(1)):
        dialogs = data["augmented_dialogues"][0]["messages"]
        graph = data["graph"]

        raw_data = PipelineRawDataType(dialogs=dialogs, true_graph=graph)
        report = pipeline.invoke(
            ms, f"{pipeline.name}_sim_model:v1", raw_data, enable_evals=True
        )[1].model_dump()
        new_summary.append(
            {
                "graph": data["topic"],
                "duration": report["properties"]["time"],
                "similarity": report["properties"]["complex_graph_comparison"][
                    "similarity_avg"
                ],
            }
        )

    mean_duration = sum(item["duration"] for item in new_summary) / len(new_summary)
    mean_similarity = sum(item["similarity"] for item in new_summary) / len(new_summary)
    new_summary.append(
        {"graph": "average", "duration": mean_duration, "similarity": mean_similarity}
    )

    date = datetime.now().strftime("%d.%m.%Y")
    last_summary_file = get_latest_file(
        metrics_path, pipeline.name + "*.json"
    )
    last_summary_file_today = get_latest_file(
        metrics_path, f"{pipeline.name}_{date}*.json"
    )
    report_number = (
        int(last_summary_file_today.stem.split("_")[-1]) + 1
        if last_summary_file_today is not None
        else 1
    )
    metrics_path.mkdir(parents=True, exist_ok=True)
    with open(f"{metrics_path_name}/{pipeline.name}_{date}_{report_number}.json", "w") as f:
        json.dump(new_summary, f, ensure_ascii=False, indent=4)

    if last_summary_file is not None:
        with open(last_summary_file, "r") as f:
            last_summary = json.load(f)
        return compare_summaries(pipeline.name, last_summary, new_summary)
    return True


def test_d2g_pipelines():
    pipelines = [
        D2GLightPipeline(
            name="three_stages_light",
            model_storage=ms,
        ),
        D2GLLMPipeline(
            name="three_stages_llm",
            model_storage=ms,
        ),
        D2GExtenderPipeline(
            name="extender",
            model_storage=ms,
        ),
    ]

    for pipeline in pipelines:
        assert test_d2g_pipeline(pipeline), (
            "Pipeline %s results got worse: check %s/%s*.json for details"
            % (pipeline.name, metrics_path_name, pipeline.name)
        )

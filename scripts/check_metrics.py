import os
import json
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional
from datetime import datetime
from datasets import load_dataset
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLAlchemyMd5Cache
from sqlalchemy import create_engine


from dialogue2graph.utils.logger import Logger
from dialogue2graph.pipelines.core.pipeline import BasePipeline
from dialogue2graph.pipelines.d2g_light.pipeline import D2GLightPipeline
from dialogue2graph.pipelines.d2g_llm.pipeline import D2GLLMPipeline
from dialogue2graph.pipelines.d2g_extender import D2GExtenderPipeline
from dialogue2graph.pipelines.model_storage import ModelStorage
from dialogue2graph.pipelines.helpers.parse_data import PipelineRawDataType


load_dotenv()
engine = create_engine(os.getenv("SQLALCHEMY_DATABASE_URI"))
set_llm_cache(SQLAlchemyMd5Cache(engine=engine))

dataset = load_dataset(
    "DeepPavlov/d2g_generated_augmented", token=os.getenv("HUGGINGFACE_TOKEN")
)


logger = Logger(__file__)
ms = ModelStorage()
metrics_path_name = "tests/metrics_results"
metrics_path = Path(metrics_path_name)


def get_latest_file(directory_path: Path, pattern: str = "*") -> Optional[Path]:
    """
    Return the latest file from the given directory_path that matches the given pattern.

    Args:
        directory_path (Path): Path to look for the latest file in.
        pattern (str, optional): Pattern to use for globbing. Defaults to "*".

    Returns:
        Optional[Path]: The latest file found, or None if no files were found.
    """
    try:
        return max(
            directory_path.glob(pattern),
            key=lambda item: item.stat().st_ctime,
            default=None,
        )
    except ValueError:
        return None


def compare_summaries(
    pipeline_name: str, last_summary: list[dict], new_summary: list[dict]
) -> bool:
    """
    Compares the last summary of the pipeline with the new summary.

    Args:
        pipeline_name (str): The name of the pipeline.
        last_summary (list[dict]): The last summary of the pipeline.
        new_summary (list[dict]): The new summary of the pipeline.

    Returns:
        bool: Whether the pipeline improved.
    """
    if not last_summary or not new_summary:
        return False
    last_avg = last_summary[-1]
    new_avg = new_summary[-1]
    if last_avg["graph"] != "average" or new_avg["graph"] != "average":
        logger.warning("Summary does not contain average graph")
        return False
    duration_diff = new_avg["duration"] - last_avg["duration"]
    if duration_diff > 1:
        logger.warning(
            "Pipeline %s got slower for: %f",
            pipeline_name,
            duration_diff,
        )
    avg_diff = last_avg["similarity"] - new_avg["similarity"]
    if avg_diff >= 0.01:
        logger.warning(
            "Pipeline %s got worse for: %f",
            pipeline_name,
            avg_diff,
        )
    return avg_diff < 0.01


def test_d2g_pipeline(pipeline: BasePipeline) -> bool:
    """
    Check whether pipeline metrics against dataset is better than last in records

    Args:
        pipeline (BasePipeline): The pipeline to test.

    Returns:
        bool: Whether the pipeline improved.
    """
    # Run the pipeline on a single random item from the dataset
    new_summary = []
    # for data in dataset["train"].select(range(0, len(dataset["train"]), 30)):
    for data in dataset["train"].select(range(2)):
        dialogs = data["augmented_dialogues"][0]["messages"]
        graph = data["graph"]

        # Parse the raw data
        raw_data = PipelineRawDataType(dialogs=dialogs, true_graph=graph)
        report = pipeline.invoke(raw_data, enable_evals=True)[1].model_dump()
        # Extract the duration and similarity from the report
        new_summary.append(
            {
                "graph": data["topic"],
                "duration": report["properties"]["time"],
                "similarity": report["properties"]["complex_graph_comparison"][
                    "similarity_avg"
                ],
            }
        )

    # Calculate the mean duration and mean similarity
    mean_duration = sum(item["duration"] for item in new_summary) / len(new_summary)
    mean_similarity = sum(item["similarity"] for item in new_summary) / len(new_summary)
    # Add the mean values to the summary
    new_summary.append(
        {"graph": "average", "duration": mean_duration, "similarity": mean_similarity}
    )

    # Get the date and create a new file with the date and report number
    date = datetime.now().strftime("%d.%m.%Y")
    last_summary_file = get_latest_file(metrics_path, pipeline.name + "*.json")
    last_summary_file_today = get_latest_file(
        metrics_path, f"{pipeline.name}_{date}*.json"
    )
    report_number = (
        int(last_summary_file_today.stem.split("_")[-1]) + 1
        if last_summary_file_today is not None
        else 1
    )
    # Save the new summary to a file
    metrics_path.mkdir(parents=True, exist_ok=True)
    with open(
        f"{metrics_path_name}/{pipeline.name}_{date}_{report_number}.json", "w"
    ) as f:
        json.dump(new_summary, f, ensure_ascii=False, indent=4)

    # If there is a last summary, compare it with the new summary
    if last_summary_file is not None:
        with open(last_summary_file, "r") as f:
            last_summary = json.load(f)
        return compare_summaries(pipeline.name, last_summary, new_summary)
    # If there is no last summary, return True
    return True


def test_d2g_pipelines():
    """
    Tests all D2G pipelines. Checks if the results of pipelines are improving.
    Compares the results of the current run with the results of the last run.
    Asserts that the result is True (i.e., the pipeline's results are improving).

    Raises:
        If any pipeline's results are not improving, it raises an AssertionError
        with a message indicating which pipeline's results got worse
    """

    pipelines = [
        D2GLightPipeline(
            name="three_stages_light",
            model_storage=ms,
            sim_model="sim_model"
        ),
        D2GLLMPipeline(
            name="three_stages_llm", model_storage=ms, sim_model="sim_model"
        ),
        D2GExtenderPipeline(
            name="extender",
            model_storage=ms,
            sim_model="sim_model"
        ),
    ]

    pipeline_results = [test_d2g_pipeline(pipeline) for pipeline in pipelines]

    for idx in range(len(pipeline_results)):
        if not pipeline_results[idx]:
            logger.warning(
                "Pipeline %s results got worse: check %s/%s*.json for details",
                pipelines[idx].name,
                metrics_path_name,
                pipelines[idx].name,
            )

    assert all(pipeline_results), "Pipelines results got worse!"


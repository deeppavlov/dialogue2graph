""" Module for generating graphs using the D2GLLMPipeline from CLI.
"""
import os
import json
from pathlib import Path
import datetime
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLAlchemyMd5Cache
from langchain_community.cache import InMemoryCache
from sqlalchemy import create_engine

from dialog2graph import metrics
from dialog2graph.pipelines.d2g_llm.pipeline import D2GLLMPipeline
from dialog2graph.pipelines.helpers.parse_data import PipelineRawDataType
from dialog2graph.pipelines.model_storage import ModelStorage
from dialog2graph.utils.logger import Logger

logger = Logger(__name__)

try:
    engine = create_engine(os.getenv("SQLALCHEMY_DATABASE_URI"))
    set_llm_cache(SQLAlchemyMd5Cache(engine=engine))
except Exception:
    logger.warning("SQLAlchemyMd5Cache is not available")
    set_llm_cache(InMemoryCache())

ms = ModelStorage()


def generate_llm(
    dialogs: str,
    tgraph: str,
    enable_evals: bool,
    config: dict,
    graph_path: str,
    report_path: str,
):

    """
    Generates a graph from dialog data using the d2g_llm pipeline and saves the results.

    Args:
        dialogs (str): Input dialogs file.
        tgraph (str): Input true graph file.
        enable_evals (bool): Whether to enable evaluation metrics during the pipeline execution.
        config (dict): Configuration parameters for the pipeline.
        graph_path (str): Path to save the output graph file. If None, the graph is printed.
        report_path (str): Path to save the output report file. If None, a default path with a timestamp is used.

    Side Effects:
        Loads and applies configuration to the model storage if provided.
        Writes the resulting graph and report to specified paths, or prints them if not specified.
    """

    if config != {}:
        ms.load(config)

    pipeline = D2GLLMPipeline(
        "d2g_llm", ms, step2_evals=metrics.DGEvalBase, end_evals=metrics.DGEvalBase
    )

    raw_data = PipelineRawDataType(dialogs=dialogs, true_graph=tgraph)
    result, report = pipeline.invoke(raw_data, enable_evals=enable_evals)
    result_graph = {
        "nodes": result.graph_dict["nodes"],
        "edges": result.graph_dict["edges"],
    }

    if graph_path is not None:
        Path(graph_path).parent.mkdir(parents=True, exist_ok=True)
        with open(graph_path, "w", encoding="utf-8") as f:
            json.dump({"graph": result_graph}, f, indent=2, ensure_ascii=False)
    else:
        print(json.dumps({"graph": result_graph}, indent=2, ensure_ascii=False))
    if report_path is None:
        print(str(report))
        now = datetime.datetime.now()
        report_path = (
            f"./report_{pipeline.name}_{now.strftime('%Y-%m-%d_%H-%M-%S')}.json"
        )
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path.suffix == ".json":
        report.to_json(report_path)
    elif report_path.suffix == ".csv":
        report.to_csv(report_path)
    elif report_path.suffix == ".html":
        report.to_html(report_path)
    elif report_path.suffix == ".md":
        report.to_markdown(report_path)
    elif report_path.suffix == ".txt":
        report.to_text(report_path)
    print("report saved to ", report_path)

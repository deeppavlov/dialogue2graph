"""Module for running d2g_extender pipeline from CLI"""

import os
import json
from pathlib import Path
import datetime
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLAlchemyMd5Cache
from langchain_community.cache import InMemoryCache
from sqlalchemy import create_engine

from dialog2graph import metrics
from dialog2graph.pipelines.d2g_extender.pipeline import D2GExtenderPipeline
from dialog2graph.pipelines.model_storage import ModelStorage
from dialog2graph.pipelines.helpers.parse_data import PipelineRawDataType
from dialog2graph.utils.logger import Logger


logger = Logger(__name__)

try:
    engine = create_engine(os.getenv("SQLALCHEMY_DATABASE_URI"))
    set_llm_cache(SQLAlchemyMd5Cache(engine=engine))
except Exception:
    logger.warning("SQLAlchemyMd5Cache is not available")
    set_llm_cache(InMemoryCache())

ms = ModelStorage()


def generate_extender(
    dialogs: str,
    graph: str,
    tgraph: str,
    enable_evals: bool,
    config: dict,
    graph_path: str,
    report_path: str,
):
    """
    Generates graph from dialogs via d2g_extender pipeline using parameters from config
    and saves graph dictionary to output_path

    Args:
        dialogs (str): Input dialogs file
        graph (str): Input graph file
        tgraph (str): Input true graph file
        enable_evals (bool): Whether to call pipeline evals
        config (dict): Parameters for pipeline
        graph_path (str): Output graph file
        report_path (str): Output report file

    Side Effects:
        Loads and applies configuration to the model storage if provided.
        Writes the resulting graph and report to specified paths, or prints them if not specified.
    """

    if config != {}:
        ms.load(config)

    pipeline = D2GExtenderPipeline(
        "d2g_ext",
        ms,
        step1_evals=metrics.PreDGEvalBase,
        extender_evals=metrics.PreDGEvalBase,
        step2_evals=metrics.DGEvalBase,
        end_evals=metrics.DGEvalBase,
        step=1,
    )

    raw_data = PipelineRawDataType(
        dialogs=dialogs, supported_graph=graph, true_graph=tgraph
    )
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

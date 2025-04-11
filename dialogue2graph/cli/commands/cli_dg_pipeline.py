import logging
import json
from pathlib import Path
import datetime
import csv
import pandas as pd
import mimetypes


from dialogue2graph import metrics
from dialogue2graph.pipelines.d2g_light.pipeline import D2GLightPipeline
from dialogue2graph.pipelines.helpers.parse_data import PipelineRawDataType
from dialogue2graph.pipelines.model_storage import ModelStorage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ms = ModelStorage()


def parse_json(json_path):
    try:
        with open(str(json_path)) as f:
            return json.load(f)
    except ValueError as e:
        logger.error(f"Failed to load json file: {e}")
        return None  # or: raise


def parse_csv(csv_path):
    try:
        with open(str(csv_path)) as f:
            return csv.DictReader(f)
    except ValueError:
        return None  # or: raise


def parse_txt(txt_path):
    try:
        if mimetypes.guess_type(str(txt_path))[0] == "text/plain":
            return True
        else:
            return None
    except ValueError:
        return None  # or: raise


def parse_html(html_path):
    try:
        with open(str(html_path)) as f:
            return pd.read_html(f)[0]
    except ValueError:
        return None  # or: raise


def parse_md(md_path):
    try:
        with open(str(md_path)) as f:
            lines = f.readlines()
            if lines[0].startswith("# Report for "):
                return True
            else:
                return None
    except ValueError:
        return None  # or: raise


def test_dg_generation(
    dialogs: str,
    tgraph: str,
    enable_evals: bool,
    config: dict,
    graph_path: str,
    report_path: str,
):
    """Generates graph from dialogs via d2g_light pipeline using parameters from config
    and saves graph dictionary to output_path"""

    if config != {}:
        ms.load(config)
    pipeline = D2GLightPipeline(
        "d2g_light", ms, step2_evals=metrics.DGEvalBase, end_evals=metrics.DGEvalBase
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
        assert parse_json(report_path) is not None, "saved report is not valid json"
    elif report_path.suffix == ".csv":
        report.to_csv(report_path)
        assert parse_csv(report_path) is not None, "saved report is not valid csv"
    elif report_path.suffix == ".html":
        report.to_html(report_path)
        assert parse_html(report_path) is not None, "saved report is not valid html"
    elif report_path.suffix == ".md":
        report.to_markdown(report_path)
        assert parse_md(report_path) is not None, "saved report is not valid markdown"
    elif report_path.suffix == ".txt":
        report.to_text(report_path)
        assert parse_txt(report_path) is not None, "saved report is not valid text"
    print("report saved to ", report_path)

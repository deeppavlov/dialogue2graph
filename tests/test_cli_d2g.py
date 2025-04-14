import os
import json
import pathlib
import logging
import mimetypes
import csv
import pandas as pd
from langchain_community.document_loaders import UnstructuredMarkdownLoader


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)


filepath = pathlib.Path(__file__).resolve().parent


def parse_json(json_path):
    try:
        with open(str(json_path)) as f:
            return json.load(f)
    except ValueError as e:
        logger.error(f"Failed to load json file: {e}")
        return None


def parse_csv(csv_path):
    try:
        with open(str(csv_path)) as f:
            return csv.DictReader(f)
    except ValueError:
        return None


def parse_txt(txt_path):
    try:
        if mimetypes.guess_type(str(txt_path))[0] == "text/plain":
            return True
        else:
            return None
    except ValueError:
        return None


def parse_html(html_path):
    try:
        with open(str(html_path)) as f:
            return pd.read_html(f)[0]
    except ValueError:
        return None


def parse_md(md_path):
    try:
        with open(str(md_path)) as f:
            lines = f.readlines()
            if lines[0].startswith("# Report for "):
                return True
            else:
                return None
    except ValueError:
        return None

def check_json_subreports(report_path):
    with open(str(report_path)) as f:
        report = json.load(f)
    if "subreports" not in report:
        return False
    if "compare_graphs:step2" not in report["subreports"][0] or "is_same_structure:step2" not in report["subreports"][0]:
        return False
    return True


def check_md_subreports(report_path):
    loader = UnstructuredMarkdownLoader(str(report_path), mode="elements")
    data = loader.load()

    if len(data) < 9 or 'compare_graphs:step2' not in data[8].page_content \
        or 'is_same_structure:step2' not in data[8].page_content:
        return False

    return True

def check_csv_subreports(report_path):
    with open(str(report_path)) as f:
        csv_reader = csv.DictReader(f)
        first_row = next(csv_reader)

    if "compare_graphs:step2" not in first_row or "is_same_structure:step2" not in first_row:
        return False
    return True

def check_txt_subreports(report_path):
    with open(str(report_path)) as f:
        lines = f.readlines()

    if len(lines) < 9 or "compare_graphs:step2" not in lines[8] or "is_same_structure:step2" not in lines[8]:
        return False
    return True

def test_gen_light_json_positive():
    report_path = filepath.joinpath('d2g_light_test_report.json')
    exit_status = os.system(
        f"dialogue2graph gen-graph-light -c {filepath.joinpath('cfg.yml')} -d {filepath.joinpath('test_cli_no_graph.json')} -t {filepath.joinpath('test_cli_no_dialogs.json')} -r {report_path} -ev"
    )
    assert exit_status == 0, "dialogue2graph failed to generate report"
    assert parse_json(report_path) is not None, "saved report is not valid json"
    assert check_json_subreports(report_path), f"report file {report_path} doesn't have subreports"

def test_gen_light_json_no_eval():
    report_path = filepath.joinpath('d2g_light_test_report.json')
    os.system(
        f"dialogue2graph gen-graph-light -c {filepath.joinpath('cfg.yml')} -d {filepath.joinpath('test_cli_no_graph.json')} -t {filepath.joinpath('test_cli_no_dialogs.json')} -r {report_path}"
    )
    assert not check_json_subreports(report_path), f"report file {report_path} shall not have subreports"

def test_gen_light_md_positive():
    report_path = filepath.joinpath('d2g_light_test_report.md')
    exit_status = os.system(
        f"dialogue2graph gen-graph-light -c {filepath.joinpath('cfg.yml')} -d {filepath.joinpath('test_cli_no_graph.json')} -t {filepath.joinpath('test_cli_no_dialogs.json')} -r {report_path} -ev"
    )
    assert exit_status == 0, "dialogue2graph failed to generate report"
    assert parse_md(report_path) is not None, "saved report is not valid markdown"
    assert check_md_subreports(report_path), f"report file {report_path} doesn't have subreports"

def test_gen_light_md_no_eval():
    report_path = filepath.joinpath('d2g_light_test_report.md')
    os.system(
        f"dialogue2graph gen-graph-light -c {filepath.joinpath('cfg.yml')} -d {filepath.joinpath('test_cli_no_graph.json')} -t {filepath.joinpath('test_cli_no_dialogs.json')} -r {report_path}"
    )
    assert not check_md_subreports(report_path), f"report file {report_path} shall not have subreports"

def test_gen_light_csv_positive():
    report_path = filepath.joinpath('d2g_light_test_report.csv')
    exit_status = os.system(
        f"dialogue2graph gen-graph-light -c {filepath.joinpath('cfg.yml')} -d {filepath.joinpath('test_cli_no_graph.json')} -t {filepath.joinpath('test_cli_no_dialogs.json')} -r {report_path} -ev"
    )
    assert exit_status == 0, "dialogue2graph failed to generate report"
    assert parse_csv(report_path) is not None, "saved report is not valid csv"
    assert check_csv_subreports(report_path), f"report file {report_path} doesn't have subreports"

def test_gen_light_csv_no_eval():
    report_path = filepath.joinpath('d2g_light_test_report.csv')
    os.system(
        f"dialogue2graph gen-graph-light -c {filepath.joinpath('cfg.yml')} -d {filepath.joinpath('test_cli_no_graph.json')} -t {filepath.joinpath('test_cli_no_dialogs.json')} -r {report_path}"
    )
    assert not check_csv_subreports(report_path), f"report file {report_path} shall not have subreports"

def test_gen_light_txt_positive():
    report_path = filepath.joinpath('d2g_light_test_report.txt')
    exit_status = os.system(
        f"dialogue2graph gen-graph-light -c {filepath.joinpath('cfg.yml')} -d {filepath.joinpath('test_cli_no_graph.json')} -t {filepath.joinpath('test_cli_no_dialogs.json')} -r {report_path} -ev"
    )
    assert exit_status == 0, "dialogue2graph failed to generate report"
    assert parse_txt(report_path) is not None, "saved report is not valid csv"
    assert check_txt_subreports(report_path), f"report file {report_path} doesn't have subreports"

def test_gen_light_txt_no_eval():
    report_path = filepath.joinpath('d2g_light_test_report.txt')
    os.system(
        f"dialogue2graph gen-graph-light -c {filepath.joinpath('cfg.yml')} -d {filepath.joinpath('test_cli_no_graph.json')} -t {filepath.joinpath('test_cli_no_dialogs.json')} -r {report_path}"
    )
    assert not check_txt_subreports(report_path), f"report file {report_path} shall not have subreports"

def test_gen_light_nocfg():
    report_path = filepath.joinpath('d2g_light_test_report.json')
    exit_status = os.system(
        f"dialogue2graph gen-graph-light -c {filepath.joinpath('cfg1.yml')} -d {filepath.joinpath('test_cli.json')} -t {filepath.joinpath('test_cli.json')} -r {report_path}"
    )
    assert exit_status != 0, "dialogue2graph should fail without config file"


def test_gen_light_wrong_cfg():
    report_path = filepath.joinpath('d2g_light_test_report.json')
    exit_status = os.system(
        f"dialogue2graph gen-graph-light -c {filepath.joinpath('wrong_cfg.yml')} -d {filepath.joinpath('test_cli.json')} -t {filepath.joinpath('test_cli.json')} -r {report_path}"
    )
    assert exit_status != 0, "dialogue2graph should fail with config file where no model_name is specified"

def test_gen_light_no_dialog_file():
    report_path = filepath.joinpath('d2g_light_test_report.json')
    exit_status = os.system(
        f"dialogue2graph gen-graph-light -c {filepath.joinpath('cfg1.yml')} -d {filepath.joinpath('test_cli1.json')} -t {filepath.joinpath('test_cli.json')} -r {report_path}"
    )
    assert exit_status != 0, "dialogue2graph should fail without dialog file"

def test_gen_light_no_dialogs():
    report_path = filepath.joinpath('d2g_light_test_report.json')
    exit_status = os.system(
        f"dialogue2graph gen-graph-light -c {filepath.joinpath('cfg.yml')} -d {filepath.joinpath('test_cli_no_dialogs.json')} -t {filepath.joinpath('test_cli.json')} -r {report_path}"
    )
    assert exit_status != 0, "dialogue2graph should fail with input file without dialogs"


def test_gen_light_no_graph_file():
    report_path = filepath.joinpath('d2g_light_test_report.json')
    exit_status = os.system(
        f"dialogue2graph gen-graph-light -c {filepath.joinpath('cfg.yml')} -d {filepath.joinpath('test_cli.json')} -t {filepath.joinpath('test_cli_no_graph.json')} -r {report_path}"
    )
    assert exit_status != 0, "dialogue2graph should fail with input file without graph"

def test_gen_light_empty_dialogs():
    report_path = filepath.joinpath('d2g_light_test_report.json')
    exit_status = os.system(
        f"dialogue2graph gen-graph-light -c {filepath.joinpath('cfg.yml')} -d {filepath.joinpath('test_cli_empty_dialogs.json')} -t {filepath.joinpath('test_cli_no_graph.json')} -r {report_path}"
    )
    assert exit_status != 0, "dialogue2graph should fail with empty dialogs"

def test_gen_light_empty_graph():
    report_path = filepath.joinpath('d2g_light_test_report.json')
    exit_status = os.system(
        f"dialogue2graph gen-graph-light -c {filepath.joinpath('cfg.yml')} -d {filepath.joinpath('test_cli.json')} -t {filepath.joinpath('test_cli_empty_graph.json')} -r {report_path}"
    )
    assert exit_status != 0, "dialogue2graph should fail with empty graph"
import json
import pytest
import dotenv
from dialogue2graph import metrics
from dialogue2graph import Dialogue
from dialogue2graph.pipelines.d2g_extender.pipeline import D2GExtenderPipeline
from dialogue2graph.pipelines.helpers.parse_data import PipelineRawDataType
from dialogue2graph.pipelines.model_storage import ModelStorage

dotenv.load_dotenv()
if not dotenv.find_dotenv():
    pytest.skip("Skipping test as .env file is not found", allow_module_level=True)
ms = ModelStorage()


@pytest.fixture(scope="session")
def test_data():
    """
    Read JSON data once per pytest session
    (scope="session") to avoid re-reading the file.
    """
    with open("tests/test_metrics_data.json", encoding="utf-8") as f:
        data = json.load(f)
    return data


@pytest.fixture
def graph_positive_1(test_data):
    """
    Graph #1 from the positive scenario (data[0]).
    """
    return test_data[0]["graph"]


@pytest.fixture
def graph_negative(test_data):
    """
    Graph #3 from the negative scenario (data[2]).
    """
    return test_data[2]["graph"]


@pytest.fixture
def dialogues_positive(test_data):
    """
    Dialogues for the positive scenario (from data[0]).
    """
    raw_dialogues = test_data[0]["dialogues"]
    return [Dialogue(**dlg) for dlg in raw_dialogues]


@pytest.fixture
def dialogues_negative(test_data):
    """
    Dialogues for the negative scenario (from data[2]).
    """
    raw_dialogues = test_data[2]["dialogues"]
    return [Dialogue(**dlg) for dlg in raw_dialogues]


def test_d2g_extender_positive(dialogues_positive, graph_positive_1):
    """Test that d2g_algo pipeline returns True for GT=graph_positive_1
    and input=dialogues_positive"""

    ms.add(
        key="extending_llm",
        config={"model": "chatgpt-4o-latest", "temperature": 0},
        model_type="llm",
    )
    ms.add(
        key="filling_llm",
        config={"model": "o3-mini", "temperature": 1},
        model_type="llm",
    )
    ms.add(
        key="formatting_llm",
        config={"model": "gpt-4o-mini", "temperature": 0},
        model_type="llm",
    )
    ms.add(
        key="dialog_llm",
        config={"model": "os-mini", "temperature": 1},
        model_type="llm",
    )
    ms.add(
        key="sim_model",
        config={"model_name": "BAAI/bge-m3", "device": "cuda:0"},
        model_type="emb",
    )

    pipeline = D2GExtenderPipeline(
        name="d2g_ext",
        model_storage=ms,
        extending_llm="extending_llm",
        filling_llm="filling_llm",
        formatting_llm="formatting_llm",
        dialog_llm="dialog_llm",
        sim_model="sim_model",
        step1_evals=metrics.PreDGEvalBase,
        extender_evals=metrics.PreDGEvalBase,
        step2_evals=metrics.DGEvalBase,
        end_evals=metrics.DGEvalBase,
        step=1,
    )

    raw_data = PipelineRawDataType(
        dialogs=dialogues_positive, true_graph=graph_positive_1
    )
    _, report = pipeline.invoke(raw_data, enable_evals=True)

    assert report.properties["complex_graph_comparison"]["value"] is True, (
        f"Expected value=True, but got: {report.properties['complex_graph_comparison']['description']}"
    )


def test_d2g_extender_negative(dialogues_negative, graph_positive_1):
    """Test that d2g_algo pipeline returns False for GT=graph_negative
    and input=dialogues_negative"""

    ms.add(
        key="extending_llm",
        config={"model": "chatgpt-4o-latest", "temperature": 0},
        model_type="llm",
    )
    ms.add(
        key="filling_llm",
        config={"model": "o3-mini", "temperature": 1},
        model_type="llm",
    )
    ms.add(
        key="formatting_llm",
        config={"model": "gpt-4o-mini", "temperature": 0},
        model_type="llm",
    )
    ms.add(
        key="dialog_llm",
        config={"model": "os-mini", "temperature": 1},
        model_type="llm",
    )
    ms.add(
        key="sim_model",
        config={"model_name": "BAAI/bge-m3", "device": "cpu"},
        model_type="emb",
    )

    pipeline = D2GExtenderPipeline(
        name="d2g_ext",
        model_storage=ms,
        extending_llm="extending_llm",
        filling_llm="filling_llm",
        formatting_llm="formatting_llm",
        dialog_llm="dialog_llm",
        sim_model="sim_model",
        step1_evals=metrics.PreDGEvalBase,
        extender_evals=metrics.PreDGEvalBase,
        step2_evals=metrics.DGEvalBase,
        end_evals=metrics.DGEvalBase,
        step=1,
    )

    raw_data = PipelineRawDataType(
        dialogs=dialogues_negative, true_graph=graph_positive_1
    )
    _, report = pipeline.invoke(raw_data, enable_evals=True)

    assert report.properties["complex_graph_comparison"]["value"] is False, (
        "Expected value=False in the negative scenario."
    )

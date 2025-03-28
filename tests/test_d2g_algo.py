import json
import pytest
from dialogue2graph import Graph, Dialogue
from dialogue2graph.pipelines.d2g_algo.pipeline import Pipeline
from dialogue2graph.pipelines.models import ModelsAPI

models = ModelsAPI()


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
    return Graph(graph_dict=test_data[0]["graph"])


@pytest.fixture
def graph_negative(test_data):
    """
    Graph #3 from the negative scenario (data[2]).
    """
    return Graph(graph_dict=test_data[2]["graph"])


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


def test_d2g_algo_positive(dialogues_positive, graph_positive_1):
    """Test that d2g_algo pipeline returns True for GT=graph_positive_1
    and input=dialogues_positive"""

    filling_llm = models("llm", name="o3-mini", temp=1)
    formatting_llm = models("llm", name="gpt-4o-mini", temp=0)
    sim_model = models("similarity", name="BAAI/bge-m3", device="cuda:0")

    pipeline = Pipeline(filling_llm, formatting_llm, sim_model)

    result = pipeline.invoke(dialogues_positive, graph_positive_1)
    assert result["is_same_structure"] is True, f"Expected value=True, but got: {result['is_same_structure']}"
    assert result["graph_match"]["value"] is True, result["graph_match"]["description"]


def test_d2g_algo_negative(dialogues_negative, graph_negative):
    """Test that d2g_algo pipeline returns False for GT=graph_negative
    and input=dialogues_negative"""

    filling_llm = models("llm", name="o3-mini", temp=1)
    formatting_llm = models("llm", name="gpt-4o-mini", temp=0)
    sim_model = models("similarity", name="BAAI/bge-m3", device="cuda:0")

    pipeline = Pipeline(filling_llm, formatting_llm, sim_model)

    result = pipeline.invoke(dialogues_negative, graph_negative)
    assert result["is_same_structure"] is False, "Expected value=False in the negative scenario."
    assert result["graph_match"]["value"] is False, "Expected value=False in the negative scenario."

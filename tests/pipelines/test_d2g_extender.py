import os
import json
import pytest
import dotenv
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLAlchemyMd5Cache
from langchain_community.cache import InMemoryCache
from sqlalchemy import create_engine

from dialog2graph import Dialog
from dialog2graph.pipelines.d2g_extender.pipeline import D2GExtenderPipeline
from dialog2graph.pipelines.helpers.parse_data import PipelineRawDataType
from dialog2graph.pipelines.model_storage import ModelStorage
from dialog2graph.utils.logger import Logger

dotenv.load_dotenv()
if not dotenv.find_dotenv():
    pytest.skip("Skipping test as .env file is not found", allow_module_level=True)

logger = Logger(__file__)

try:
    engine = create_engine(os.getenv("SQLALCHEMY_DATABASE_URI"))
    set_llm_cache(SQLAlchemyMd5Cache(engine=engine))
except Exception:
    logger.warning("SQLAlchemyMd5Cache is not available")
    set_llm_cache(InMemoryCache())

ms = ModelStorage()


@pytest.fixture(scope="session")
def test_data():
    """
    Read JSON data once per pytest session
    (scope="session") to avoid re-reading the file.
    """
    with open("tests/test_pipelines_data.json", encoding="utf-8") as f:
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
def dialogs_positive(test_data):
    """
    Dialogs for the positive scenario (from data[0]).
    """
    raw_dialogs = test_data[0]["dialogs"]
    return [Dialog(**dlg) for dlg in raw_dialogs]


@pytest.fixture
def dialogs_negative(test_data):
    """
    Dialogs for the negative scenario (from data[2]).
    """
    raw_dialogs = test_data[2]["dialogs"]
    return [Dialog(**dlg) for dlg in raw_dialogs]


def test_d2g_extender_positive(dialogs_positive, graph_positive_1):
    """Test that d2g_extender pipeline returns True for GT=graph_positive_1
    and input=dialogs_positive"""

    pipeline = D2GExtenderPipeline(
        name="extender",
        model_storage=ms,
    )

    raw_data = PipelineRawDataType(
        dialogs=dialogs_positive, true_graph=graph_positive_1
    )
    _, report = pipeline.invoke(raw_data, enable_evals=True)

    assert report.properties["complex_graph_comparison"]["similarity_avg"] > 0.99, (
        f"Expected similarity_avg > 0.99, but got: {report.properties['complex_graph_comparison']['similarity_avg']}"
    )


def test_d2g_extender_negative(dialogs_negative, graph_negative):
    """Test that d2g_extender pipeline returns False for GT=graph_negative
    and input=dialogs_negative"""

    pipeline = D2GExtenderPipeline(
        name="extender",
        model_storage=ms,
    )

    raw_data = PipelineRawDataType(dialogs=dialogs_negative, true_graph=graph_negative)
    _, report = pipeline.invoke(raw_data, enable_evals=True)

    assert report.properties["complex_graph_comparison"]["similarity_avg"] <= 0.99, (
        "Expected similarity_avg <= 0.99 in the negative scenario."
    )

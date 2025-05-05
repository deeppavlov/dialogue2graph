# tests/test_automatic_metrics.py

import json
import pytest

# Remove the unused import to fix F401
# from dialog2graph.metrics.no_llm_metrics.metrics import match_graph_triplets  # <-- DELETED because it's unused

from dialog2graph import Graph, Dialog
from dialog2graph.metrics.no_llm_metrics.metrics import (
    is_same_structure,
    match_dg_triplets,
    are_paths_valid,
    match_roles,
    is_correct_length,
    all_utterances_present,
    triplet_match_accuracy,
    compute_graph_metrics,
)


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
def graph_positive_2(test_data):
    """
    Graph #2 from the positive scenario (data[1]).
    """
    return Graph(graph_dict=test_data[1]["graph"])


@pytest.fixture
def graph_negative(test_data):
    """
    Graph #3 from the negative scenario (data[2]).
    """
    return Graph(graph_dict=test_data[2]["graph"])


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


# -------------------------------
# Tests for the positive scenario
# -------------------------------


def test_all_utterances_present_positive(graph_positive_1, dialogs_positive):
    """
    Check that all utterances (from nodes and edges) of the first graph
    appear in the positive dialogs.
    """
    result = all_utterances_present(graph_positive_1, dialogs_positive)
    assert result is True, f"Expected value=True, but got: {result}"


def test_is_same_structure_positive(graph_positive_1, graph_positive_2):
    """
    Check that the two positive graphs are isomorphic (is_same_structure).
    """
    assert is_same_structure(graph_positive_1, graph_positive_2) is True, (
        "Expected the graphs to have the same structure."
    )


def test_match_dg_triplets_positive(graph_positive_1, dialogs_positive):
    """
    Check that all (assistant-user-assistant) triplets in the dialogs
    match the triplets in the graph (match_dg_triplets).
    """
    result = match_dg_triplets(graph_positive_1, dialogs_positive)
    assert result["value"] is True, f"Expected value=True, but got: {result}"


def test_are_paths_valid_positive(graph_positive_1, dialogs_positive):
    """
    Check that the dialogs form valid paths in the first graph (are_paths_valid).
    """
    result = are_paths_valid(graph_positive_1, dialogs_positive)
    assert result["value"] is True, f"Not all paths are valid: {result}"


def test_match_roles_positive(dialogs_positive):
    """
    Check that the first two positive dialogs have matching roles (assistant/user).
    """
    if len(dialogs_positive) < 2:
        pytest.skip("Not enough dialogs to test match_roles in positive data.")

    d1 = dialogs_positive[0]
    d2 = dialogs_positive[1]
    assert match_roles(d1, d2) is True, (
        "Expected the roles to match completely between dialogs."
    )


def test_is_correct_length_positive(dialogs_positive):
    """
    Check that the first two positive dialogs have the same number of messages.
    """
    if len(dialogs_positive) < 2:
        pytest.skip("Not enough dialogs to test is_correct_length in positive data.")

    d1 = dialogs_positive[0]
    d2 = dialogs_positive[1]
    assert is_correct_length(d1, d2) is True, (
        "Expected the dialogs to have the same length."
    )


def test_triplet_match_accuracy_positive(graph_positive_1, graph_positive_2):
    """
    Check the matching accuracy (triplet_match_accuracy) between two positive graphs.
    We expect perfect matching (1.0 for both nodes and edges).
    """
    acc = triplet_match_accuracy(graph_positive_1, graph_positive_2)
    assert acc["node_accuracy"] == 1.0, "Expected node_accuracy to be 1.0"
    assert acc["edge_accuracy"] == 1.0, "Expected edge_accuracy to be 1.0"


def test_compute_graph_metrics_positive(graph_positive_1, graph_positive_2):
    """
    Check basic metrics computed by compute_graph_metrics on a list of two positive graphs.
    The goal is to verify that the expected keys are present in the result.
    """
    results = compute_graph_metrics([graph_positive_1, graph_positive_2])
    expected_keys = {
        "with_cycles",
        "percentage_with_cycles",
        "average_edges_amount",
        "average_nodes_amount",
        "total_graphs",
        "total_edges",
        "total_nodes",
    }
    missing_keys = expected_keys - set(results.keys())
    assert not missing_keys, f"Missing keys in the result: {missing_keys}"


# -------------------------------
# Tests for the negative scenario
# -------------------------------


def test_all_utterances_present_negative(graph_negative, dialogs_negative):
    """
    Check that not all utterances of the graph are present in the negative dialogs.
    We expect the function to return NOT True but a set of missing phrases.
    """
    result = all_utterances_present(graph_negative, dialogs_negative)
    assert isinstance(result, set) and len(result) > 0, (
        "Expected some utterances to be missing in the dialogs."
    )


def test_is_same_structure_negative(graph_positive_1, graph_negative):
    """
    Check that the positive and negative graphs are not isomorphic.
    """
    assert is_same_structure(graph_positive_1, graph_negative) is False, (
        "Expected the graphs to have different structures."
    )


def test_match_dg_triplets_negative(graph_negative, dialogs_negative):
    """
    Check match_dg_triplets for the negative scenario.
    We expect the result to have value=False (since there are missing triplets).
    """
    result = match_dg_triplets(graph_negative, dialogs_negative)
    assert result["value"] is False, "Expected value=False in the negative scenario."
    assert "absent_triplets" in result, "Expected 'absent_triplets' key in the result."


def test_are_paths_valid_negative(graph_negative, dialogs_negative):
    """
    In the negative example, the paths might still be valid or partially invalid.
    """
    result = are_paths_valid(graph_negative, dialogs_negative)
    assert result["value"] is True, (
        "Expected value=True in negative data; adjust checks if needed."
    )


def test_is_correct_length_negative(dialogs_negative):
    """
    In the negative scenario, one of the tests shows that the dialogs have different lengths.
    """
    if len(dialogs_negative) < 2:
        pytest.skip("Not enough dialogs to test is_correct_length in negative data.")

    d1 = dialogs_negative[0]
    d2 = dialogs_negative[1]
    assert is_correct_length(d1, d2) is False, (
        "Expected the dialogs to have different lengths, so is_correct_length should be False."
    )


def test_triplet_match_accuracy_negative(graph_positive_1, graph_negative):
    """
    Check that when comparing a positive graph with a negative graph,
    the matching accuracy is less than 1.0.
    """
    acc = triplet_match_accuracy(graph_positive_1, graph_negative)
    # According to data: node_accuracy = 0.875, edge_accuracy = 0.9
    assert acc["node_accuracy"] < 1.0, (
        "Expected node_accuracy to be less than 1.0 in the negative scenario"
    )
    assert acc["edge_accuracy"] < 1.0, (
        "Expected edge_accuracy to be less than 1.0 in the negative scenario"
    )

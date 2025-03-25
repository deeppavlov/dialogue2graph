import pytest
from dialogue2graph.pipelines.core.graph import Graph
from dialogue2graph.metrics.no_llm_metrics.metrics import _get_jaccard_edges, _get_jaccard_nodes, _collapse_multiedges, _collapse_multinodes


@pytest.fixture
def sample_graph_dict():
    return {
        "nodes": [
            {"id": 1, "label": "start", "is_start": True, "utterances": ["Hello", "Hi"]},
            {"id": 2, "label": "response", "is_start": False, "utterances": ["How can I help?"]},
        ],
        "edges": [{"source": 1, "target": 2, "utterances": ["I need help"]}],
    }


def test_graph_metrics_initialization(sample_graph_dict):
    """Test that graph metrics functions can be called without errors"""
    graph = Graph(graph_dict=sample_graph_dict)

    # Test collapse functions
    edges = list(graph.graph.edges(data=True))
    nodes = list(graph.graph.nodes(data=True))

    collapsed_edges = _collapse_multiedges(edges)
    assert isinstance(collapsed_edges, dict)

    collapsed_nodes = _collapse_multinodes(nodes)
    assert isinstance(collapsed_nodes, dict)

    # Test Jaccard metrics
    max_values, max_indices = _get_jaccard_edges(edges, edges)
    assert isinstance(max_values, list)
    assert isinstance(max_indices, list)

    max_values, max_indices = _get_jaccard_nodes(nodes, nodes)
    assert isinstance(max_values, list)
    assert isinstance(max_indices, list)

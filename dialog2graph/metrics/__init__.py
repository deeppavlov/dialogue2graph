from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

from dialog2graph.pipelines.core.graph import Graph
from dialog2graph.pipelines.core.schemas import CompareResponse

# from dialog2graph.metrics import llm_metrics
from dialog2graph.metrics import no_llm_metrics
from dialog2graph.pipelines.helpers.parse_data import PipelineDataType
from dialog2graph.metrics.no_llm_metrics.keys2graph import graph_triplet_comparison

PreDGEvalBase = [no_llm_metrics.is_same_structure]
# DGEvalBase = [llm_metrics.compare_graphs, no_llm_metrics.is_same_structure]
DGEvalBase = [
    graph_triplet_comparison.compare_two_graphs,
    no_llm_metrics.is_same_structure,
]
DGReportType = dict


def compare_graphs_light(graph: Graph, data: PipelineDataType) -> bool:
    """
    Compares a generated Graph with the true Graph using two metrics:

    1. `match_dg_triplets`: checks if the generated graph matches the triplets from the dialogs.
    2. `is_same_structure`: checks if the generated graph has the same structure as the true graph.

    Args:
        graph (Graph): The generated graph to compare with the true graph.
        data (PipelineDataType): Contains the true graph and the dialogs.

    Returns:
        bool: True if both metrics return True, False otherwise.
    """
    if data.true_graph is None:
        return False
    return no_llm_metrics.match_dg_triplets(graph, data.dialogs)[
        "value"
    ] and no_llm_metrics.is_same_structure(graph, data.true_graph)


# def compare_graphs_full(graph: Graph, data: PipelineDataType) -> CompareResponse:
#     if data.true_graph is None:
#         return {"value": False, "description": "No true graph given"}
# return llm_metrics.compare_graphs(graph, data.true_graph)


def compare_graphs_full(
    model: HuggingFaceInferenceAPIEmbeddings | HuggingFaceEmbeddings,
    graph: Graph,
    data: PipelineDataType,
) -> CompareResponse:
    """
    Compares a generated Graph with the true Graph using the triplet comparison metric.

    Args:
        model (HuggingFaceEmbeddings): The model to use for computing the embeddings.
        graph (Graph): The generated graph to compare with the true graph.
        data (PipelineDataType): Contains the true graph and the dialogs.

    Returns:
        CompareResponse: A dictionary with a "value" key that is True if the graphs match, and a "description" key with a description of the comparison result.
    """
    if data.true_graph is None:
        return {"value": False, "description": "No true graph given"}
    return graph_triplet_comparison.compare_two_graphs(data.true_graph, graph, model)

from dialogue2graph.pipelines.core.graph import Graph
from dialogue2graph.pipelines.core.schemas import CompareResponse
from dialogue2graph.metrics import llm_metrics
from dialogue2graph.metrics import no_llm_metrics
from dialogue2graph.pipelines.helpers.parse_data import PipelineDataType

PreDGEvalBase = [no_llm_metrics.is_same_structure]
DGEvalBase = [llm_metrics.compare_graphs, no_llm_metrics.is_same_structure]
DGReportType = dict


def compare_graphs_light(graph: Graph, data: PipelineDataType) -> bool:
    if data.true_graph is None:
        return False
    return no_llm_metrics.match_dg_triplets(graph, data.dialogs)["value"] and no_llm_metrics.is_same_structure(graph, data.true_graph)


def compare_graphs_full(graph: Graph, data: PipelineDataType) -> CompareResponse:
    if data.true_graph is None:
        return {"value": False, "description": "No true graph given"}
    return llm_metrics.compare_graphs(graph, data.true_graph)

from dialogue2graph.metrics import llm_metrics
from dialogue2graph.metrics import no_llm_metrics

PreDGEvalBase = [no_llm_metrics.is_same_structure]
DGEvalBase = [llm_metrics.compare_graphs, no_llm_metrics.is_same_structure]
DGReportType = dict

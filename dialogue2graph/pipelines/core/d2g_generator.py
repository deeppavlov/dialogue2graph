from pydantic import Field

from dialogue2graph import metrics

from dialogue2graph.pipelines.core.algorithms import GraphGenerator


class DGBaseGenerator(GraphGenerator):
    """Dialog2graph base generator to reuse evaluate method in different algorithms
    Attributes:
        sim_model: str, model name in storage
    """
    sim_model: str = Field(description="Similarity model")

    def evaluate(self, graph, true_graph, eval_stage: str) -> metrics.DGReportType:
        """Call metrics and return report

        Args:
            graph: generated graph
            true_graph: expected graph
            eval_stage: string defining eval stage, like step2 or end
        Returns:
            dictionary with report like {"metric_name": result}
        """
        report = {}
        sim_model = self.model_storage.storage[self.sim_model].model
        for metric in getattr(self, eval_stage + "_evals"):
            report[metric.__name__ + ":" + eval_stage] = metric(
                true_graph, graph, sim_model
            )
        return report

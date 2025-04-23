from pydantic import Field

from dialogue2graph import metrics

from dialogue2graph.pipelines.core.algorithms import GraphGenerator


class DGBaseGenerator(GraphGenerator):
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
                sim_model, true_graph, graph
            )
        return report

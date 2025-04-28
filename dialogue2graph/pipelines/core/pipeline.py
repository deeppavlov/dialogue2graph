"""
Pipeline
---------

The module contains base pipeline class.
"""

import time
from typing import Union, Tuple, Any
from pydantic import BaseModel, Field
from dialogue2graph.pipelines.core.algorithms import (
    DialogAugmentation,
    DialogueGenerator,
    GraphGenerator,
    GraphExtender,
)
from dialogue2graph.pipelines.helpers.parse_data import (
    RawDGParser,
    PipelineRawDataType,
    PipelineDataType,
)
from dialogue2graph.pipelines.report import PipelineReport
from dialogue2graph.metrics import compare_graphs_full, compare_graphs_light
from dialogue2graph.pipelines.model_storage import ModelStorage


class BasePipeline(BaseModel):
    """Base class for pipelines
    Attributes:
        model_storage (ModelStorage): An object to manage and store models used in the pipeline.
        sim_model (str): The key for the similarity embedder model in the model storage.
    """

    model_storage: ModelStorage = Field(description="Model storage")
    sim_model: str = Field(description="Similarity model")
    name: str = Field(description="Name of the pipeline")
    steps: list[
        Union[DialogueGenerator, DialogAugmentation, GraphGenerator, GraphExtender]
    ] = Field(default_factory=list)

    def _validate_pipeline(self):
        pass

    def invoke(
        self,
        raw_data: PipelineRawDataType,
        enable_evals=False,
    ) -> Tuple[Any, PipelineReport]:
        """
        Invoke the pipeline to process the raw data and generate a report.

        This method processes the given raw data through each step in the pipeline,
        generating both output data (result of the pipeline) and a report detailing the pipeline's execution.
        It measures execution time, performs simple graph comparisons, and optionally
        evaluates the results with more detailed comparisons.

        Args:
            raw_data (PipelineRawDataType): The raw input data to be processed by the pipeline.
            enable_evals (bool, optional): If True, performs additional evaluations
                                           and adds more detailed comparisons to the report.

        Returns:
            Tuple[Any, PipelineReport]: A tuple containing the final output of the pipeline
                                        and a detailed report of the pipeline's execution.
        """

        data: PipelineDataType = RawDGParser().invoke(raw_data)
        report = PipelineReport(service=self.name)
        st_time = time.time()
        output = data
        for step in self.steps:
            output, subreport = step.invoke(output, enable_evals=enable_evals)
            report.add_subreport(subreport)
        end_time = time.time()
        report.add_property("time", end_time - st_time)
        report.add_property(
            "simple_graph_comparison",
            compare_graphs_light(output, data),
        )
        if enable_evals:
            report.add_property(
                "complex_graph_comparison",
                compare_graphs_full(
                    self.model_storage.storage[self.sim_model].model, output, data
                ),
            )

        return output, report

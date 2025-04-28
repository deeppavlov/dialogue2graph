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


class BasePipeline(BaseModel):
    """Base class for pipelines"""

    name: str = Field(description="Name of the pipeline")
    steps: list[
        Union[DialogueGenerator, DialogAugmentation, GraphGenerator, GraphExtender]
    ] = Field(default_factory=list)

    def _validate_pipeline(self):
        pass

    def invoke(
        self, raw_data: PipelineRawDataType, enable_evals=False
    ) -> Tuple[Any, PipelineReport]:
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
            "simple_graph_comparison", compare_graphs_light(output, data)
        )
        if enable_evals:
            report.add_property(
                "complex_graph_comparison", compare_graphs_full(output, data)
            )

        return output, report

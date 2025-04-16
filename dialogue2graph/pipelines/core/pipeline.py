"""
Pipeline
---------
This module contains base pipeline class.
"""

from typing import Union
from pydantic import BaseModel, Field
from dialogue2graph.pipelines.core.algorithms import (
    DialogAugmentation,
    DialogueGenerator,
    GraphGenerator,
    GraphExtender,
    InputParser,
)


class BasePipeline(BaseModel):
    # TODO: add docs
    steps: list[
        Union[
            InputParser,
            DialogueGenerator,
            DialogAugmentation,
            GraphGenerator,
            GraphExtender,
        ]
    ] = Field(default_factory=list)

    def _validate_pipeline(self):
        pass

    def invoke(self, data):
        for step in self.steps:
            output = step.invoke(data)
            data = output
        return output

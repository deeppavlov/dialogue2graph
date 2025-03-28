import pandas as pd
from typing import Union
from pydantic import BaseModel, Field
from dialogue2graph.pipelines.core.graph import Graph
from dialogue2graph.pipelines.core.algorithms import DialogAugmentation, DialogueGenerator, GraphGenerator, InputParser, GraphExtender


class Pipeline(BaseModel):
    steps: list[Union[InputParser, DialogueGenerator, DialogAugmentation, GraphGenerator, GraphExtender]] = Field(default_factory=list)

    def _validate_pipeline(self):
        pass

    def invoke(self, data, gt: Graph = None) -> Graph | dict | pd.DataFrame:
        n_invokes = len(self.steps)
        if gt:
            n_invokes = len(self.steps) - 1
            output = data
        for step in self.steps[:n_invokes]:
            output = step.invoke(data)
            data = output
        if gt:
            output = self.steps[-1].evaluate(output, gt)
        return output
